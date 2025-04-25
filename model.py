from transformers import AutoModelForCausalLM, AutoTokenizer
import torch # type: ignore
import torch.nn as nn # type: ignore
from typing import Optional
from peft import ( # type: ignore
    get_peft_model,
)
from safetensors.torch import load_file # type: ignore

    
def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class ICAE(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()

        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        self.icae = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", resume_download=True)
        
        # self.decompress_layer = nn.Linear(in_features=self.dim, out_features=self.icae.config.hidden_size, dtype=torch.bfloat16)

        self.decoder = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", resume_download=True)

        self.vocab_size = self.icae.config.vocab_size + 1    # [PAD] token
        self.pad_token_id = self.vocab_size - 1

        # tunable
        self.mem_size = self.training_args.fixed_mem_size
        self.vocab_size_with_mem = self.vocab_size + self.mem_size # so, the mem tokens are in the range [self.vocab_size, self.vocab_size + self.mem_size)

        # special tokens in addition to mem and length tokens
        num_special_tokens_added = 3
        self.ae_token_id = self.vocab_size_with_mem + 0
        self.boc_token_id = self.vocab_size_with_mem + 1
        self.eoc_token_id = self.vocab_size_with_mem + 2

        self.icae.resize_token_embeddings(self.vocab_size_with_mem + num_special_tokens_added) 
        
        # special tokens for Llama-2/Mistral tokenizer
        self.bos_id = 1
        self.eos_id = 2
        
        self.dim = self.icae.config.hidden_size
        self.memory_token_embed = nn.Embedding(self.mem_size + num_special_tokens_added, self.dim, padding_idx=None)
        self.icae = get_peft_model(self.icae, lora_config)
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        append_seq = torch.arange(
            self.vocab_size,
            self.vocab_size + self.mem_size,
            dtype=torch.long
        ).unsqueeze(0)   # shape: (1, mem_size)
        self.register_buffer("append_sequence", append_seq)

        self.latent_dim = self.dim   # or set this to some smaller dimension if desired
        self.fc_mu = nn.Linear(self.dim, self.latent_dim, dtype=torch.bfloat16)
        self.fc_logvar = nn.Linear(self.dim, self.latent_dim, dtype=torch.bfloat16)
        self.beta = model_args.h_noiser_ratio

    def train(self, mode: bool = True):
        # 1) flip training/eval flags as usual
        super().train(mode)
        
        # 2) when entering *training* mode, do your freeze + ckpting
        if mode:
            print("Freezing the decoder…")
            freeze_model(self.decoder)
            self.decoder.eval()  # keep it in eval inside train
            print_trainable_parameters(self)

            print("Enabling gradient checkpointing on decoder…")
            self.decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        return self

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # (B, T_enc)
        prompt_answer_ids: torch.LongTensor = None,  # (B, T_dec)
        labels: Optional[torch.LongTensor] = None,  # (B, T_dec_cond + T_labels); [-100] * T_dec_cond masked out
    ):
        # B: batch size
        # D: hidden size
        # mem_size: number of memory tokens
        # vocab_size: vocab size
        # T_labels (labels): T_reasoning + eos_token 
        # T_enc (encoder input sequence length): S_question + S_query + boc_token + T_labels
        # T_dec_cond (decoder input memory token prefix): mem_size + ae_token
        # T_dec (decoder input sequence length): T_dec_cond + T_reasoning + eos_token

        batch_size = input_ids.size(0)

        # encoder part
        prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer_ids)  # (B, T_dec, D)
            
        batch_append = self.append_sequence.expand(batch_size, -1)
        input_ids = torch.cat([input_ids, batch_append], dim=1)  # (B, T_enc + mem_size)
        mem_flag_enc = (input_ids >= self.vocab_size)  # (B, T_enc + mem_size)

        input_embedding = self.icae.get_base_model().model.embed_tokens(input_ids)  # (B, T_enc + mem_size, D)
        input_embedding[mem_flag_enc] = self.memory_token_embed(input_ids[mem_flag_enc] - self.vocab_size).to(input_embedding)
        
        # compress the input
        compress_outputs = self.icae(inputs_embeds=input_embedding, output_hidden_states=True)  # (B, T_enc + mem_size, D)
        compress_outputs = compress_outputs.hidden_states[-1]  # (B, T_enc + mem_size, D)

        # collect memory tokens
        mem_flag = (input_ids >= self.vocab_size) & (input_ids < self.vocab_size_with_mem)  # (B, T_enc + mem_size)
        compress_outputs = compress_outputs[mem_flag]    # (B*mem_size, D)
        compress_outputs = compress_outputs.view(batch_size, self.mem_size, -1)  # (B, mem_size, D)

        pooled = compress_outputs.mean(dim=1)   # (B, D)

        # Compute the latent parameters
        mu = self.fc_mu(pooled)       # (B, D)
        log_var = self.fc_logvar(pooled)   # (B, D)

        # Reparameterization trick to sample z
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)   # (B, D)

        z_mem_flat = (
            z
            .unsqueeze(1)                    # (B, 1, D)
            .expand(-1, self.mem_size, -1)   # (B, mem_size, D)
            .reshape(-1, z.size(-1))         # (B*mem_size, D)
        )

        # decoder part
        decoder_mem_flag = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size_with_mem)  # (B, T_dec)

        prompt_answer_embs[decoder_mem_flag] = z_mem_flat
        special_prompt = prompt_answer_ids >= self.vocab_size_with_mem  # (B, T_dec); NOTE: This works because boc/eoc tokens aren't included in prompt_answer_ids (decoder input).
        special_offsets = (prompt_answer_ids - self.vocab_size)[special_prompt]  # (B*T_dec,)
        special_embs_flat = self.memory_token_embed(special_offsets)  # (B*T_dec, D)
        special_embs_flat = special_embs_flat.to(prompt_answer_embs)
        prompt_answer_embs[special_prompt] = special_embs_flat
        
        if self.training:
            decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
        else:
            with self.icae.disable_adapter():
                decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, output_hidden_states=True)

        logits = decoder_outputs.logits  # (B, T_dec, vocab_size)

        effective_logits = logits[:,:-1,:].reshape(-1, logits.size(-1))  # (B*T_dec, vocab_size)
        target_ids = labels[:,1:].reshape(-1)  # (B*T_dec,)

        loss_recon = self.loss_fct(effective_logits, target_ids)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = loss_recon + self.beta * kl_loss

        return {"loss": loss, "logits": logits}
    
    def compress(self, input_ids: torch.LongTensor):
        batch_size = input_ids.size(0)

        batch_append = self.append_sequence.expand(batch_size, -1)  # (B, mem_size)
        input_ids = torch.cat([input_ids, batch_append], dim=1)  # (B, T_enc + mem_size)
        mem_flag_enc = (input_ids >= self.vocab_size)  # (B, T_enc + mem_size)
        input_emb = self.icae.get_base_model().model.embed_tokens(input_ids)
        input_emb[mem_flag_enc] = self.memory_token_embed(
            input_ids[mem_flag_enc] - self.vocab_size
        ).to(input_emb)

        encoder_out = self.icae(
            inputs_embeds=input_emb,
            output_hidden_states=True
        )  # (B, T_enc + mem_size, D)
        last_hidden = encoder_out.hidden_states[-1]  # (B, T_enc + mem_size, D)

        mem_flag = (input_ids >= self.vocab_size) & (input_ids < self.vocab_size_with_mem)  # (B, T_enc + mem_size)
        flat_mem = last_hidden[mem_flag]  # (B*mem_size, D)
        compress_outputs = flat_mem.view(batch_size, self.mem_size, -1)  # (B, mem_size, D)

        return compress_outputs

    def get_decoder_input_embeds(
        self,
        input_ids: torch.LongTensor,
        prompt_ids: torch.LongTensor,
        use_mean: bool = True
    ) -> torch.Tensor:
        mem_outputs = self.compress(input_ids)        # (B, mem_size, D)
        pooled = mem_outputs.mean(dim=1)               # (B, D)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        if use_mean:
            z = mu
        else:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)

        flat_z = z.unsqueeze(1).expand(-1, self.mem_size, -1).reshape(-1, z.size(-1))

        prompt_embs = self.icae.get_base_model().model.embed_tokens(prompt_ids)

        dec_mem_flag = (prompt_ids >= self.vocab_size) & (prompt_ids < self.vocab_size_with_mem)
        prompt_embs[dec_mem_flag] = flat_z

        special = prompt_ids >= self.vocab_size_with_mem
        offs = (prompt_ids - self.vocab_size)[special]
        se = self.memory_token_embed(offs)
        prompt_embs[special] = se.to(prompt_embs)

        return prompt_embs

    @torch.no_grad()
    def inference(
        self,
        input_ids: torch.LongTensor,
        prompt_ids: torch.LongTensor,
        max_steps: int = 512,
        use_mean: bool = True
    ) -> list[str]:
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        prompt_ids = prompt_ids.to(device)

        dec_embs = self.get_decoder_input_embeds(input_ids, prompt_ids, use_mean)
        out_emb = dec_embs.clone()
        past = None
        batch_size = input_ids.size(0)
        gen_ids: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_steps):
            with self.icae.disable_adapter():
                res = self.icae(inputs_embeds=out_emb, past_key_values=past, use_cache=True)
            logits = res.logits[:, -1, :self.vocab_size-1]
            past = res.past_key_values
            next_ids = torch.argmax(logits, dim=-1)  # (B,)

            for i in range(batch_size):
                if not finished[i]:
                    token_id = next_ids[i].item()
                    if token_id == self.eos_id:
                        finished[i] = True
                    else:
                        gen_ids[i].append(token_id)

            if finished.all():
                break

            next_emb = self.icae.get_base_model().model.embed_tokens(next_ids)
            out_emb = next_emb.unsqueeze(1)  # (B,1,D)

        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
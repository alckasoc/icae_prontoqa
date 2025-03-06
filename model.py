# ICAE that supports multi span concat

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from typing import Optional
from peft import (
    get_peft_model,
)
import math
from safetensors.torch import load_file

    
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
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        self.icae = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", resume_download=True)
        
        self.training = self.model_args.train    
        
        if self.training:    # independent model for gradient checkpointing
            self.decoder = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", resume_download=True)

        self.vocab_size = self.icae.config.vocab_size + 1    # [PAD] token
        self.pad_token_id = self.vocab_size - 1
        self.mean_compression_rate = training_args.mean_compression_rate

        # tunable
        self.mem_size = self.training_args.fixed_mem_size
        self.vocab_size_with_mem = self.vocab_size + self.mem_size # so, the mem tokens are in the range [self.vocab_size, self.vocab_size + self.mem_size)

        # special tokens in addition to mem and length tokens
        self.ae_token_id = self.vocab_size_with_mem + 0
        self.lm_token_id = self.vocab_size_with_mem + 1
        self.ft_token_id = self.vocab_size_with_mem + 2        

        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 3) 
        
        # special tokens for Llama-2/Mistral tokenizer
        self.bos_id = 1
        self.eos_id = 2
        
        self.dim = self.icae.config.hidden_size
        self.icae = get_peft_model(self.icae, lora_config)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory_token_embed = nn.Embedding(self.mem_size + 3, self.dim, padding_idx=None)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.append_sequence = torch.arange(self.vocab_size, self.vocab_size + self.mem_size, dtype=torch.long, device=self.device).unsqueeze(0)    # mem tokens
        
        if self.training:
            self.init()


    def init(self):
        print("Freezing the decoder...")
        freeze_model(self.decoder)
        self.decoder.eval()
        print_trainable_parameters(self)
        if self.training_args.restore_from is not None and self.training_args.restore_from != "":
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")
        print("Enabling gradient checkpointing...")
        # self.icae.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                
        
    def compute_num_segments(self, total_length):
        assert total_length > 0
        num_segments = math.ceil(total_length / (self.mem_size * self.mean_compression_rate))  # 128 * 4 -> 1 * (128*4)
        return num_segments


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        prompt_answer_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # encoder part
        print("input_ids shape: ", input_ids.size())
        print("prompt_answer_ids shape: ", prompt_answer_ids.size())
        print("labels shape: ", labels.size())
        
        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        print("num_segments: ", num_segments)
        segment_length = math.ceil(total_length / num_segments)
        print("segment_length: ", segment_length)

        prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer_ids)
        print("prompt_answer_embs shape: ", prompt_answer_embs.size())
        
        max_compressed_length = num_segments * self.mem_size
        print("max_compressed_length: ", max_compressed_length)
        
        compress_outputs = torch.zeros((max_compressed_length, self.dim)).to(prompt_answer_embs)
        print("compress_outputs shape: ", compress_outputs.size())
        
        for segment_idx in range(num_segments):
            print(f"===============Segment {segment_idx}=======================")
            
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            print(f"start_idx: {start_idx} | end_idx: {end_idx}")
            segment_input_ids = input_ids[:, start_idx:end_idx]
            print("segment_input_ids shape: ", segment_input_ids.size())
            print("append_sequence shape: ", self.append_sequence.size())
            segment_input_ids = torch.cat([segment_input_ids, self.append_sequence], dim=1)
            print("segment_input_ids shape after concat: ", segment_input_ids.size())
            mem_flag = segment_input_ids >= self.vocab_size
            print("mem_flag shape: ", mem_flag.size())

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            print("segment_input_embedding shape: ", segment_input_embedding.size())
            segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding)
            print("Populated segment_input_embedding memory tokens")
            
            # compress the current segment
            segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]
            print("segment_compress_outputs (last hidden state) shape: ", segment_compress_outputs.size())

            # collect memory tokens
            compress_outputs[segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = segment_compress_outputs[mem_flag]
            print(f"Filled in compressed memory for memory segment {segment_idx}.")
            
            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()

            print(f"===============Segment {segment_idx} END=======================")
        
        # decoder part
        decoder_mem_flag = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size + self.mem_size)   # only mem tokens
        print("decoder_mem_flag shape: ", decoder_mem_flag.size())

        prompt_answer_embs[decoder_mem_flag] = compress_outputs  # replace memory slots
        print("Populated decoder memory tokens with compressed outputs.")
        special_prompt = prompt_answer_ids >= self.vocab_size_with_mem
        prompt_answer_embs[special_prompt] = self.memory_token_embed(prompt_answer_ids[special_prompt] - self.vocab_size).to(prompt_answer_embs)    # replace special token's embedding from self.memory_token_embed
        print("Populated decoder special memory tokens.")
        
        if self.training:   # has an independent se.f.decoder
            decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
        else:
            with self.icae.disable_adapter():   # no independent decoder; use self.icae
                decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, output_hidden_states=True)

        logits = decoder_outputs.logits
        print("decoder_outputs logits shape: ", logits.size())

        effective_logits = logits[:,:-1,:].reshape(-1, logits.size(-1))  # Why are we skipping the last generated logit? It's probably the eos token.
        print("effective_logits shape: ", effective_logits.size())
        target_ids = labels[:,1:].reshape(-1)  # Why does it take from the first index onwards?
        print("target_ids shape: ", target_ids.size())

        loss = self.loss_fct(effective_logits, target_ids)
        return {"loss": loss, "logits": logits}

    def tokens_to_embeddings(self, token_ids):   # input_tokens can be either normal tokens and special tokens
        embeddings = self.icae.get_base_model().model.embed_tokens(token_ids)
        special_flags = token_ids >= self.vocab_size
        embeddings[special_flags] = self.memory_token_embed(token_ids[special_flags] - self.vocab_size).to(embeddings)    # replace special token's embedding from self.memory_token_embed
        return embeddings
    
    def _compress(
        self,
        input_ids: torch.LongTensor = None
    ):  # for inference; compress a fixed length of input into memory slots

        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)
        
        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((max_compressed_length, self.dim))
        
        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_input_ids = torch.cat([segment_input_ids, self.append_sequence], dim=1)
            mem_flag = segment_input_ids >= self.vocab_size

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding)

            # compress the current segment
            segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]

            # collect memory tokens
            compress_outputs[segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = segment_compress_outputs[mem_flag]
            
            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()
        
        return compress_outputs
    
    def run_inference(self, text: str):
        self.eval()
        with torch.no_grad():
            tokenized_text = self.tokenizer(text, truncation=True,
                                          max_length=5120, padding=False,
                                          return_attention_mask=False)
            # Generate compressed outputs
            input_ids = torch.LongTensor([tokenized_text['input_ids']]).to(self.device)
            memory_slots = self._compress(input_ids)
            prompt_ids = torch.LongTensor([[self.ae_token_id]]).to(self.device)

            prompt_answer_embs = self.tokens_to_embeddings(prompt_ids)
            memory_slots = memory_slots.to(prompt_answer_embs)
            decoder_input_embeddings = torch.cat((memory_slots.unsqueeze(0), prompt_answer_embs), dim=1)
            output = decoder_input_embeddings.clone()

            generate_text = []
            past_key_values = None

            # Generate text output
            for i in range(512):
                with self.icae.disable_adapter():   # no independent decoder; use self.icae
                    out = self.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                logit = out.logits[:, -1, :self.vocab_size-1]
                past_key_values = out.past_key_values

                next_token_id = torch.argmax(logit, dim=-1)
                # print(next_token_id)
                
                if next_token_id.item() == 2:   # eos
                    break

                output = self.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(self.device)
                generate_text.append(next_token_id.item())

            generated_text = self.tokenizer.decode(generate_text)

        return generated_text

    def encode_inference(self, text):
        self.eval()
        with torch.no_grad():
            tokenized_text = self.tokenizer(text, truncation=True,
                                          max_length=5120, padding=False,
                                          return_attention_mask=False)
            # Generate compressed outputs
            input_ids = torch.LongTensor([tokenized_text['input_ids']]).to(self.device)
            memory_slots = self._compress(input_ids)

        return memory_slots
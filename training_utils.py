from transformers import Trainer
import os
import torch # type: ignore
import random
import gc

import wandb
from peft import ( # type: ignore
    LoraConfig,
)
from model import ICAE


def run_batch_inference(
    model: ICAE,
    lines: list[dict],
    device: torch.device,
    max_steps: int = 512,
    use_mean: bool = True
) -> list[str]:
    """
    Batch inference wrapper for ICAE model.

    Args:
        model: Trained ICAE instance.
        lines: List of dicts with keys 'input' and 'reasoning'.
        device: torch device to run on.
        max_steps: Maximum decoding steps.
        use_mean: Whether to use deterministic latent (mu) or sample.

    Returns:
        List of generated strings, one per line.
    """
    model.eval()
    model.to(device)

    # Tokenize and prepare batches
    inputs, prompts = [], []
    for line in lines:
        txt = model.tokenizer(
            line['input'], truncation=True, max_length=5120,
            padding=False, return_attention_mask=False
        )
        chain = model.tokenizer(
            line['reasoning'], truncation=True, max_length=5120,
            padding=False, return_attention_mask=False
        )
        enc_ids = (
            txt['input_ids']
            + [model.boc_token_id]
            + chain['input_ids'][1:]  # Remove padding token from beginning because we treat this as the entire sequence.
            + [model.eoc_token_id]
        )
        inputs.append(enc_ids)
        prompts.append(model.append_sequence[0].tolist() + [model.ae_token_id])

    # Pad encoder inputs
    pad_id = model.pad_token_id
    max_len = max(len(ids) for ids in inputs)
    inputs_padded = [ids + [pad_id] * (max_len - len(ids)) for ids in inputs]

    input_ids_tensor = torch.LongTensor(inputs_padded).to(device)
    prompt_ids_tensor = torch.LongTensor(prompts).to(device)

    # Invoke model inference
    outputs = model.inference(
        input_ids=input_ids_tensor,
        prompt_ids=prompt_ids_tensor,
        max_steps=max_steps,
        use_mean=use_mean
    )
    return outputs


def train_model(args, notes, model, train_dataset, eval_dataset, model_args, training_args, lines, data_collator=None):
    if max(training_args.per_device_train_batch_size, training_args.per_device_eval_batch_size) == 1:
        data_collator = None
        
    print("LENGTH OF TRAIN DATASET: ", len(train_dataset))
    print("LENGTH OF EVAL DATASET: ", len(eval_dataset))

    # print training_args at local_rank 0
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(training_args)
    
    run = wandb.init(
        project="icae_prontoqa",
        tags=["new"],
        config={
            "model_args": vars(model_args),
            "training_args": vars(args)
        },
        notes=notes
    )

    training_args.output_dir = os.path.join(training_args.output_dir, run.name)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    checkpoint = None
    
    print(f"Loaded from the checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=None)
    # trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    torch.save(trainer.model.state_dict(), f"{training_args.output_dir}/model_weights.pth")

    #### 🚀 **Free Up GPU Memory BEFORE Re-Loading Model** ####
    print("🚀 Clearing VRAM before inference...")

    # Move model to CPU first (prevents GPU tensors lingering)
    model.to("cpu")
    del model  # Delete model object

    # Delete Trainer & Free Memory
    del trainer
    gc.collect()  # Garbage collection
    torch.cuda.empty_cache()  # Clear VRAM

    #### 🚀 **Now Reload the Model for Inference** ####
    print("🚀 Re-loading model for inference...")

    # EVALUATION
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = ICAE(model_args, training_args, lora_config)
    print(f"Loading trained checkpoint from {training_args.output_dir}")
    model.load_state_dict(torch.load(f"{training_args.output_dir}/model_weights.pth"), strict=False)
    model = model.to(args.device)

    #### 🚀 **Batch Inference** ####
    print("🚀 Running batch inference with run_batch_inference…")
    outputs = run_batch_inference(
        model=model,
        lines=lines,
        device=args.device,
        max_steps=512,
        use_mean=True
    )

    table_data = []
    for inp, out in zip(lines, outputs):
        print("========================================")
        print("INPUT: ", inp)
        print("OUTPUT:", out)
        print("========================================")
        table_data.append([inp, out])

    my_table = wandb.Table(
        columns=["input", "output"],
        data=table_data
    )
    run.log({"icae_new": my_table})
    run.finish()


def text_extraction(input_ids, length):
    
    input_len = len(input_ids)
    assert input_len >= 1, f"Error: invalid input length ({input_len})"
    
    if input_len <= length: # if shorter, keep the complete text
        return input_ids
    else:
        last_start = input_len - length
        random_start = random.randint(0, last_start)
        return input_ids[random_start: random_start+length]
    
def pretrain_tokenize_function(
    examples, 
    tokenizer,
    model_max_length,
    ae_token_id,
    eos_id,
    boc_token_id,
    eoc_token_id,
    mem,
):
    # 1) Build all the raw text inputs in one pass
    texts = [
        f"Question: {q} {query}"
        for q, query in zip(examples["question"], examples["query"])
    ]
    chains = examples["chain_of_thought"]

    # 2) Tokenize both lists in batch
    tokenized_text = tokenizer(
        texts,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    tokenized_chain = tokenizer(
        chains,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    # 3) Now build your three outputs in a single comprehension
    all_input_ids       = []
    all_prompt_ans_ids  = []
    all_labels          = []

    for qt_ids, chain_ids in zip(tokenized_text["input_ids"],
                                 tokenized_chain["input_ids"]):
        # 3a) trim the question if needed
        question_ids = text_extraction(qt_ids, model_max_length)

        # 3b) assemble encoder inputs
        input_ids = question_ids + [boc_token_id] + chain_ids[1:] + [eoc_token_id]

        # 3c) assemble decoder prompt+answer
        prompt_ids       = mem + [ae_token_id]
        answer_ids       = chain_ids[1:] + [eos_id]
        prompt_ans_ids   = prompt_ids + answer_ids

        # 3d) labels: mask prompt with -100
        labels = [-100] * len(prompt_ids) + answer_ids

        all_input_ids      .append(input_ids)
        all_prompt_ans_ids .append(prompt_ans_ids)
        all_labels         .append(labels)

    return {
        "input_ids":       all_input_ids,
        "prompt_answer_ids": all_prompt_ans_ids,
        "labels":          all_labels,
    }

class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(self, examples):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
        prompt_answer_ids = [torch.tensor(example["prompt_answer_ids"], dtype=torch.long) for example in examples]
        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        prompt_answer_ids = self.dynamic_padding(prompt_answer_ids, fill_value=self.pad_token_id)
        labels = self.dynamic_padding(labels)
        batch = {"input_ids": input_ids, "labels": labels, "prompt_answer_ids": prompt_answer_ids}
        return batch
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences)
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences
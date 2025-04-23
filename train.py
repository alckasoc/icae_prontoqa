import transformers

from dataclasses import dataclass, field
from typing import Optional
from peft import ( # type: ignore
    LoraConfig,
)
import argparse
import json

from training_utils import pretrain_tokenize_function, DataCollatorForDynamicPadding, train_model
from model import ICAE
from datasets import Dataset # type: ignore

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    lora_r: int = field(
        default=128,
        metadata={"help": "lora rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "lora alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "lora dropout"}
    )
    train: bool = field(
        default=True,
        metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"}
    )
    h_noiser_ratio: float = field(
        default=0.3,
        metadata={"help": "h_noiser ratio"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    fixed_mem_size: int = field(
        default=1,
        metadata={"help": "Enabling the fixed mem size."},
    )
    restore_from: str = field(
        default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    report_to: str = field(
        default="wandb"
    )
    max_steps: int = field(
        default=5000
    )
    save_strategy: str = field(
        default="no"
    )
    logging_strategy: str = field(
        default="steps"
    )
    logging_steps: int = field(
        default=1
    )
    learning_rate: float = field(
        default=2.5e-5
    )
    lr_scheduler_type: str = field(
        default="cosine"
    )
    lr_scheduler_kwargs: dict = field(default_factory=dict)
    warmup_steps: int = field(
        default=0
    )
    weight_decay: float = field(
        default=0.0
    )
    eval_strategy: str = field(
        default="epoch"
    )
    num_train_epochs: int = field(
        default=1
    )

def main(model_args, training_args, args, notes):    
    print("Loading dataset...")
    processed_data = []
    
    with open("345hop_random_true.json") as f:
        data = json.load(f)
    
    for _, example in data.items():
        for _, sample in example.items():
            processed_data.append({
                "question": sample["question"],
                "query": sample["query"],
                "chain_of_thought": " ".join(sample["chain_of_thought"]),
                "answer": sample["answer"]
            })
    
    hf_dataset = Dataset.from_list(processed_data)
    split_dataset = hf_dataset.train_test_split(test_size=args.test_size)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.shuffle(seed=42)
    print("Dataset loaded successfully...")

    # Inference examples.
    with open("train_10_inference_examples.jsonl", "r") as f:
        train_10_inference_examples = [json.loads(line) for line in f]

    with open("eval_10_inference_examples.jsonl", "r") as f:
        eval_10_inference_examples = [json.loads(line) for line in f]

    train_10_inference_examples = [{"input": f"Question: {i['question']} {i['query']}", "reasoning": i['chain_of_thought']} for i in train_10_inference_examples]
    eval_10_inference_examples = [{"input": f"Question: {i['question']} {i['query']}", "reasoning": i['chain_of_thought']} for i in eval_10_inference_examples]

    lines = train_10_inference_examples + eval_10_inference_examples
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("Loading model...")
    model = ICAE(model_args, training_args, lora_config).to(args.device)
    print("Model loaded successfully...")
    
    memory_size = training_args.fixed_mem_size
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

    print("Tokenizing train/eval datasets...")

    train_fn_kwargs = {
        "tokenizer": model.tokenizer, 
        "model_max_length": model.training_args.model_max_length,
        "ae_token_id": model.ae_token_id,
        "eos_id": model.eos_id,
        "boc_token_id": model.boc_token_id,
        "eoc_token_id": model.eoc_token_id,
        "mem": MEM_TOKENS, 
    }

    eval_fn_kwargs = {
        "tokenizer": model.tokenizer, 
        "model_max_length": model.training_args.model_max_length,
        "ae_token_id": model.ae_token_id,
        "eos_id": model.eos_id,
        "boc_token_id": model.boc_token_id,
        "eoc_token_id": model.eoc_token_id,
        "mem": MEM_TOKENS, 
    }
    
    train_dataset = train_dataset.map(pretrain_tokenize_function, batched=True, batch_size=100, fn_kwargs=train_fn_kwargs)
    eval_dataset = eval_dataset.map(pretrain_tokenize_function, batched=True, fn_kwargs=eval_fn_kwargs)
    print("Finished tokenizing train/eval datasets...")

    train_dataset = train_dataset.select([0])
    eval_dataset = train_dataset

    data_collator = DataCollatorForDynamicPadding(model.pad_token_id)

    print("Training model...")
    train_model(
        args,
        notes,
        model, 
        train_dataset, 
        eval_dataset, 
        model_args,
        training_args, 
        # lines,
        [{"input": f"Question: {train_dataset[0]['question']} {train_dataset[0]['query']}", "reasoning": train_dataset[0]['chain_of_thought']}],
        data_collator,
    )
    print("Finished training...")

def parse_args():
    """Parses command-line arguments and initializes Model & Training arguments dynamically."""
    parser = argparse.ArgumentParser(description="Fine-tune a LLaMA model with LoRA.")

    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training.")

    # Add arguments dynamically
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Pretrained model path.")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate.")
    parser.add_argument("--h_noiser_ratio", type=float, default=0.3, help="Ratio of hidden size to noise size.")

    # Training Arguments (Automatically Converted to `TrainingArguments`)
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test dataset split ratio.")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum training steps.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--lr_scheduler_kwargs", type=str, default="{}", help="JSON string of LR scheduler kwargs.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps.")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer type.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay coefficient.")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="Evaluation strategy.")
    parser.add_argument("--notes", type=str, help="Additional notes for the run.")

    args = parser.parse_args()

    # Convert JSON string arguments
    args.lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)

    skip_list = ['test_size', 'notes', 'device']
    
    model_args = ModelArguments(**{k: v for k, v in vars(args).items() if k in ModelArguments.__annotations__})
    training_args = TrainingArguments(**{k: v for k, v in vars(args).items() if k not in ModelArguments.__annotations__ and k not in skip_list})

    return model_args, training_args, args
    
if __name__ == "__main__":
    model_args, training_args, args = parse_args()
    main(model_args, training_args, args, args.notes)
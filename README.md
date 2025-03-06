# Running ICAE on ProntoQA

This README is a quick recap on the files in this repository.

```
icea_prontoqa/
├── 345hop_random_true.json             <- The un-preprocessed dataset.
├── train_10_inference_examples.jsonl   <- 10 randomly selected training examples used to visualize inference.
├── eval_10_inference_examples.jsonl    <- 10 randomly selected evaluation examples used to visualize inference.
├── tmp.ipynb                           <- A notebook where tested code.
├── model.py                            <- The ICAE architecture itself.
├── training_utils.py                   <- Training and data preprocessing utiities.
├── train.py                            <- The training script.
└── run.sh                              <- My experiment script.
```

I've updated the `ICAE` class in `model.py` with the `run_inference` and `encode_inference` functions. Please take a look at these functions before using them! 

## Usage

First clone this repository and install the requirements.

```
git clone https://github.com/alckasoc/icae_prontoqa
```

```
pip install -r requirements.txt
```

Make sure to also login to Hugging Face:

```
huggingface-cli login
```

I use `wandb` for logging. Make sure login when prompted during training!

For inference, you can run this following code (to run the model I've trained):

```
from training import ModelArguments, TrainingArguments
from model import ICAE
from peft import LoraConfig
import torch

model_args = ModelArguments()
training_args = TrainingArguments("./output")  # This is only for training, but the model class expects this `TrainingArguments` class for some hyperparameters

model_args.model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
model_args.lora_r = 512
model_args.lora_alpha = 128
model_args.lora_dropout = 0.05

lora_config = LoraConfig(
    r=model_args.lora_r,
    lora_alpha=model_args.lora_alpha,
    lora_dropout=model_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

model = ICAE(model_args, training_args, lora_config)

path_to_weights = ...  # Fill in.
model.load_state_dict(torch.load(path_to_weights), strict=False)
model.eval()

query = "Question: Insects are not eight-legged. Arthropods are invertebrates. Lepidopterans are insects. Each animal is multicellular. Invertebrates are animals. Butterflies are lepidopterans. Every insect is an arthropod. Arthropods are segmented. Each spider is eight-legged. Rex is a butterfly."  # This is an evaluation example.

output = model.run_inference(query)  # Returns a string output; the reconstruction.

memory_slots = model.encode_inference(query)  # Returns a vector of memory slots.
```


# backend/train.py

import os, sys, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, logging as transformers_logging
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fine-tune a LLaMA model with LoRA")
parser.add_argument("--base_model", type=str, required=True, help="Path to base model directory")
parser.add_argument("--data_file", type=str, required=True, help="Path to training dataset JSON file")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model (LoRA adapter)")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size")
parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--cutoff_len", type=int, default=512, help="Maximum sequence length (tokens)")
args = parser.parse_args()

base_model_path = args.base_model
data_file = args.data_file
output_dir = args.output_dir
num_epochs = args.epochs
batch_size = args.batch_size
gradient_accumulation = args.grad_accum
learning_rate = args.learning_rate
cutoff_len = args.cutoff_len

# Enable TF32 for speed on A100/RTX (if available)
torch.backends.cuda.matmul.allow_tf32 = True

# Set up Transformers logging to avoid too much output (we'll use explicit prints for progress)
transformers_logging.set_verbosity_warning()

# Load the dataset from the JSON file
dataset = load_dataset("json", data_files=data_file, split="train")

# Prepare the prompt format for Alpaca-style instruction tuning&#8203;:contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
def format_example(example):
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output_text = example.get("output", "").strip()
    if input_text:
        # Format with instruction and input
        prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
                  f"Write a response that appropriately completes the request.\n\n"
                  f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n")
    else:
        # Format with instruction only
        prompt = (f"Below is an instruction that describes a task. "
                  f"Write a response that appropriately completes the request.\n\n"
                  f"### Instruction:\n{instruction}\n\n### Response:\n")
    return prompt, output_text

# Load tokenizer and base model with 4-bit quantization (for memory efficiency)&#8203;:contentReference[oaicite:4]{index=4}
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", quantization_config=bnb_config)
# Enable gradient checkpointing and disable cache to reduce memory usage
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Configure LoRA fine-tuning (target key/query/value/output projections in attention)&#8203;:contentReference[oaicite:5]{index=5}
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
# Print trainable parameter info for debugging (optional)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params} / {total_params} ({100 * trainable_params/ total_params:.2f}%)")

# Preprocess the dataset to tokenize and mask prompt tokens in labels
def preprocess(example):
    prompt, answer = format_example(example)
    # Append the end-of-sequence token to the answer
    answer = answer + tokenizer.eos_token
    # Tokenize prompt and answer
    prompt_ids = tokenizer(prompt, truncation=True, max_length=cutoff_len, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, truncation=True, max_length=cutoff_len - len(prompt_ids), add_special_tokens=False)["input_ids"]
    # Combine prompt and answer, and create labels with prompt part masked out
    input_ids = prompt_ids + answer_ids
    input_ids = input_ids[:cutoff_len]
    labels = [-100] * len(prompt_ids) + answer_ids
    labels = labels[:cutoff_len]
    # Pad sequences to cutoff_len
    pad_len = cutoff_len - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
    return {"input_ids": input_ids, "labels": labels}

# Apply preprocessing to all examples
dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
# Convert dataset columns to torch tensors for Trainer
dataset.set_format("torch", columns=["input_ids", "labels"])

# Set up training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    learning_rate=learning_rate,
    bf16=torch.cuda.is_available(),  # use bfloat16 if available (e.g., A100)
    fp16=not torch.cuda.is_available(),  # otherwise use float16
    logging_steps=10,
    logging_strategy="steps",
    save_strategy="no",
    report_to="none",
    disable_tqdm=True
)
# Initialize Trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=None  # Data is already padded and collated
)

# Begin training
trainer.train()
# Save the LoRA adapter (trained parameters)
model.save_pretrained(output_dir)
# Save the tokenizer for completeness (especially if output_dir will serve as a model folder)
tokenizer.save_pretrained(output_dir)

#!/usr/bin/env python3
"""
Fine-tune Llama 3.1 8B for Clearledgr Financial AI

RunPod-optimized fine-tuning script with LoRA for efficient training
and financial domain specialization.
"""

import os
import sys
import argparse
import logging
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset, load_from_disk
import wandb

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_clearledgr_dataset(dataset_path: str) -> Dataset:
    """Load Clearledgr financial training dataset"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Check if it's a JSON file or HF dataset directory
    if dataset_path.endswith('.json'):
        # Load JSON file and convert to HF dataset
        import json
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Convert to HuggingFace dataset format
        from datasets import Dataset
        if isinstance(data, list):
            dataset = Dataset.from_list(data)
        elif isinstance(data, dict) and 'training_data' in data:
            # Handle our specific dataset structure
            dataset = Dataset.from_list(data['training_data'])
        elif isinstance(data, dict) and 'data' in data:
            dataset = Dataset.from_list(data['data'])
        else:
            # If it's a flat dict, try to extract lists
            training_samples = []
            for key, value in data.items():
                if isinstance(value, list):
                    training_samples.extend(value)
            if training_samples:
                dataset = Dataset.from_list(training_samples)
            else:
                logger.error(f"Could not find training data in JSON structure. Keys: {list(data.keys())}")
                raise ValueError(f"Invalid dataset structure in {dataset_path}")
        
        logger.info(f"Loaded {len(dataset)} samples from JSON file")
        return dataset
    else:
        # Load as HuggingFace dataset directory
        return load_from_disk(dataset_path)

def create_lora_config() -> LoraConfig:
    """Create LoRA configuration for efficient fine-tuning"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Rank - balance between efficiency and capacity
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"]  # Save these for financial vocab
    )

def load_model_and_tokenizer(model_path: str, use_4bit: bool = True):
    """Load model and tokenizer with optimization for RunPod"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model from {model_path}")
    
    # 4-bit quantization for memory efficiency
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    logger.info("✅ Model and tokenizer loaded successfully")
    return model, tokenizer

def prepare_dataset(dataset: Dataset, tokenizer, max_length: int = 2048):
    """Prepare dataset for training"""
    logger = logging.getLogger(__name__)
    
    def format_instruction(instruction, input_text, output):
        """Format the instruction-input-output into a training text"""
        if input_text and input_text.strip():
            # Format with input
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            # Format without input
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    def tokenize_function(examples):
        # Convert instruction-input-output format to text
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
            output = examples["output"][i]
            
            formatted_text = format_instruction(instruction, input_text, output)
            texts.append(formatted_text)
        
        # Tokenize the formatted text
        model_inputs = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    logger.info(f"✅ Dataset prepared: {len(tokenized_dataset)} examples")
    return tokenized_dataset

def create_training_arguments(output_dir: str, **kwargs) -> TrainingArguments:
    """Create training arguments optimized for RunPod"""
    
    # Default arguments optimized for A100 40GB
    default_args = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "save_steps": 500,
        "eval_steps": 500,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "warmup_steps": 100,
        "fp16": True,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 3,
        "report_to": ["wandb"] if "WANDB_API_KEY" in os.environ else [],
        "run_name": f"clearledgr-llama-3.1-8b-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    }
    
    # Update with any provided kwargs
    default_args.update(kwargs)
    
    return TrainingArguments(**default_args)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 8B for Clearledgr")
    parser.add_argument("--model-path", required=True, help="Path to base Llama model")
    parser.add_argument("--dataset-path", required=True, help="Path to training dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory for fine-tuned model")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--use-4bit", action="store_true", default=True, help="Use 4-bit quantization")
    parser.add_argument("--use-lora", action="store_true", default=True, help="Use LoRA fine-tuning")
    parser.add_argument("--wandb-project", default="clearledgr-llama", help="W&B project name")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("🚀 Starting Clearledgr Llama 3.1 8B Fine-tuning")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output: {args.output_dir}")
    
    # Initialize W&B if available
    if "WANDB_API_KEY" in os.environ:
        wandb.init(
            project=args.wandb_project,
            name=f"clearledgr-llama-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args)
        )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, 
        use_4bit=args.use_4bit
    )
    
    # Apply LoRA if enabled
    if args.use_lora:
        logger.info("Applying LoRA configuration...")
        lora_config = create_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load and prepare dataset
    dataset = load_clearledgr_dataset(args.dataset_path)
    
    # Split dataset if needed
    if "validation" not in dataset.column_names:
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    
    # Tokenize datasets
    train_dataset = prepare_dataset(train_dataset, tokenizer, args.max_length)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer, args.max_length)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = create_training_arguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Start training
    logger.info("🔥 Starting training...")
    
    try:
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training config
        config = {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "training_args": training_args.to_dict(),
            "lora_config": lora_config.to_dict() if args.use_lora else None,
            "training_completed": datetime.now().isoformat()
        }
        
        with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    finally:
        if "WANDB_API_KEY" in os.environ:
            wandb.finish()

if __name__ == "__main__":
    main()

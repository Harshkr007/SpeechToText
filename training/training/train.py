import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed
)
from datasets import load_metric
import os
import sys
from datetime import datetime
sys.path.append('../config')
from config.training_config import TrainingConfig
from data_preprocessing.prepare_dataset import prepare_dataset
from utils.auth import setup_huggingface_auth

wer_metric = load_metric("wer")

def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and references
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

def train():
    # Add authentication setup at the start
    try:
        setup_huggingface_auth()
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        return

    # Set random seed for reproducibility
    set_seed(42)
    
    config = TrainingConfig()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"multilingual_whisper_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting multilingual Whisper training...")
    print(f"Output directory: {output_dir}")
    
    # Prepare combined dataset
    print("Preparing multilingual dataset...")
    raw_datasets = prepare_dataset(config)
    
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(config.model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name_or_path)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        evaluation_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        gradient_checkpointing=config.gradient_checkpointing,
        group_by_length=config.group_by_length,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        hub_model_id=config.hub_model_id,
        push_to_hub=config.push_to_hub,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"],
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )
    
    # Train
    trainer.train()
    
    # Save model and processor
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")

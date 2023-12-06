import datasets
from datasets import load_dataset, Dataset, DatasetDict
from transformers import TrainingArguments
from span_marker import SpanMarkerModel, Trainer
import pandas as pd
from data_process import new_label, ori_label, process_A, process_B
import os
import argparse 
import wandb

os.environ["WANDB_API_KEY"]="XXXXX"
os.environ["WANDB_ENTITY"]="YOU_NAME"
os.environ["WANDB_PROJECT"]="YOUR_PROEJCT_NAME"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a SpanMarkerModel with PLM on NER task")
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="choose the system A or system B, please type in 'A' or 'B'",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='Babelscape/multinerd',
        help="The configuration name of the dataset to use (via the huggingface datasets library).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='roberta-base',
        help="The pre-trained English language model to generate the embedding",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The name of output director to save the checkpoint and performance metrics",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The name of director to save the final fine-tuned model",
    )
    args = parser.parse_args()
    return args

    

def main(train_data, valid_data, test_data, label, model_name, output_dir, save_dir):
    model = SpanMarkerModel.from_pretrained(
        model_name,
        labels=label,
        # SpanMarker hyperparameters:
        model_max_length=256,
        marker_max_length=128,
        entity_max_length=6,
    )
    
    train_args = TrainingArguments(
        output_dir=output_dir,
        # Training Hyperparameters:
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # gradient_accumulation_steps=2,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,  # Replace `bf16` with `fp16` if your hardware can't use bf16.
        # Other Training parameters
        logging_first_step=True,
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=2,
        report_to="wandb",
    )
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
    )
    trainer.train()
    trainer.save_model(save_dir)

    # Compute & save the metrics on the test set
    metrics = trainer.evaluate(test_data, metric_key_prefix="test")
    trainer.save_metrics("test", metrics)
    
if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    train_dataset = load_dataset(dataset, split="train")
    eval_dataset = load_dataset(dataset, split="validation")
    test_dataset = load_dataset(dataset, split="test")
    
    if args.system == "A":
        train, _, _ = process_A(train_dataset)
        eval, _, _ = process_A(eval_dataset)
        eval = eval.shuffle().select(range(3000))
        test, _, _ = process_A(test_dataset)
        label = ori_label
    elif args.system == "B":
        train, _, _ = process_B(train_dataset)
        eval, _, _ = process_B(eval_dataset)
        eval = eval.shuffle().select(range(3000))
        test, _, _ = process_B(test_dataset)
        label = new_label
    
    model_name = args.model_name
    output_dir = args.output_dir
    save_dir = args.save_dir
    main(train_data=train, valid_data=eval, test_data=test, label=label, model_name=model_name, output_dir=output_dir, save_dir=save_dir)

import os, sys
os.environ['TOKENIZERS_PARALLELISM']='false'
import pandas as pd
import numpy as np
import datasets
from datasets import load_dataset, DatasetDict, Dataset
import torch
from torch import nn
from torch.nn.functional import cross_entropy
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
import evaluate
import argparse

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

def transform(item):
    for key, value in ori_label2id.items():
      if value == item:
        return key

def process_A(dataset):
    temp = dataset.to_pandas()
    eng_data = temp[temp['lang']=='en']
    result = Dataset.from_pandas(eng_data)
    return result

def tokenize_and_align_labels(samples):
    tokenized_inputs = tokenizer(samples["tokens"], 
                                      truncation=True, 
                                      is_split_into_words=True)
    
    labels = []
    
    for idx, label in enumerate(samples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids: # set special tokens to -100
            if word_idx is None or word_idx == prev_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            prev_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

seqeval = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, 
                            axis=2)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, 
                              references=true_labels)
    
    return results

class CustomTrainer(Trainer):
    def compute_loss(self, 
                     model, 
                     inputs, 
                     return_outputs=False):
        
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss 
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
             9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
             16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
             23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]
            )
        )
        loss = loss_fct(logits.view(-1, 
                                    self.model.config.num_labels), 
                        labels.view(-1)
                        )
        return (loss, outputs) if return_outputs else loss
    
    
def main(encoded_ds, label, model_name, tokenizer, ori_id2label, ori_label2id, output_dir, save_dir):
    
    model = (AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label),
    id2label=ori_id2label,
    label2id=ori_label2id
    ))
    
    train_args = TrainingArguments(
        output_dir=output_dir,
        # Training Hyperparameters:
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # gradient_accumulation_steps=2,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_ratio=0.1,
        # fp16=True,  # Replace `bf16` with `fp16` if your hardware can't use bf16.
        # Other Training parameters
        logging_first_step=True,
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=2,
        # report_to="wandb",
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = CustomTrainer(model, 
                    args=train_args,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    tokenizer=tokenizer,
                    train_dataset=encoded_ds["train"],
                    eval_dataset=encoded_ds["eval"],
                    )
    
    trainer.train()
    trainer.save_model(save_dir)

    # Compute & save the metrics on the test set
    metrics = trainer.evaluate(encoded_ds["test"], metric_key_prefix="test")
    trainer.save_metrics("test", metrics)
    
if __name__ == "__main__":   
    args = parse_args()
    
    dataset = args.dataset
    train_dataset = load_dataset(dataset, split="train")
    eval_dataset = load_dataset(dataset, split="validation")
    test_dataset = load_dataset(dataset, split="test")
    train = process_A(train_dataset)
    evali = process_A(eval_dataset)
    test = process_A(test_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    
    ori_label2id = {
        "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
        "B-LOC": 5, "I-LOC": 6, "B-ANIM": 7, "I-ANIM": 8, "B-BIO": 9,
        "I-BIO": 10, "B-CEL": 11, "I-CEL": 12, "B-DIS": 13, "I-DIS": 14, 
        "B-EVE": 15, "I-EVE": 16, "B-FOOD": 17, "I-FOOD": 18, "B-INST": 19, 
        "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23, "I-MYTH": 24, 
        "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, 
        "I-VEHI": 30
    }
    ori_label = [key for key, value in ori_label2id.items()]
    ori_id2label = {tag: idx for idx, tag in ori_label2id.items()}
    label_list = ori_label
    
    ds = DatasetDict({
    'train': train, 
    'eval': evali, 
    'test': test})
    encoded_ds = ds.map(tokenize_and_align_labels, 
                        batched=True, 
                        remove_columns=
                            [
                                'ner_tags', 
                                'tokens',
                                '__index_level_0__',
                                'lang'
                            ]
                        )
    
    model_name = args.model_name
    output_dir = args.output_dir
    save_dir = args.save_dir
    
    main(encoded_ds=encoded_ds, label=ori_label, model_name=model_name, tokenizer=tokenizer, ori_id2label=ori_label2id, ori_label2id=ori_label2id, output_dir=output_dir, save_dir=save_dir)









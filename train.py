from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np

import evaluate

model_name = "microsoft/MiniLM-L12-H384-uncased"


dataset = load_dataset("financial_phrasebank", "sentences_allagree")

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, padding=True)


dataset = dataset.map(tokenize, batched=True)

dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": metric.compute(predictions=preds, references=labels)}

training_args = TrainingArguments(
    output_dir="./minilm-finance",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
)


trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./minilm-finance")
tokenizer.save_pretrained("./minilm-finance")

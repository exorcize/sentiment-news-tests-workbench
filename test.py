from transformers import AutoTokenizer, AutoModelForSequenceClassification
from os import getenv

import torch

TEXT = getenv('TEXT', "Apple beats earnings expectations and raises guidance.")
MODEL = getenv('MODEL', "distilbert-base-uncased-finetuned-sst-2-english")
model_name = MODEL
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = TEXT
inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=1)

labels = model.config.id2label
pred = torch.argmax(probs, dim=1).item()

print("Label:", labels[pred])
print("Confidence:", probs[0][pred].item())
print(probs)

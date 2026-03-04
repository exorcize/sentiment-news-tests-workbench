from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("./models/sentiment-onnx")
model = ORTModelForSequenceClassification.from_pretrained("./models/sentiment-onnx")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

print(pipe("Apple beats earnings expectations."))

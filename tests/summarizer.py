from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "Sachin21112004/distilbart-news-summarizer"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

with open("body.txt", "r", encoding="utf-8") as f:
    article = f.read()

inputs = tokenizer(
    article,
    return_tensors="pt",
    max_length=1024,
    truncation=True
)

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=150,
    min_length=40,
    no_repeat_ngram_size=3,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
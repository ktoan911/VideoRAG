import json

import torch
from transformers import AutoModel, AutoTokenizer

# Load model và tokenizer
model_name = "pkshatech/GLuCoSE-base-ja-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embedding(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Lấy output từ model
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.last_hidden_state: [batch, seq_len, hidden_dim]
        embeddings = outputs.last_hidden_state.mean(dim=1)  # average pooling
    return embeddings[0].tolist() 


with open(
    "/home/toannk/Desktop/Code/video-rag/all_images.json", "r", encoding="utf-8"
) as f:
    data = json.load(f)

# Thêm trường embedding
for item in data:
    item["content_embedding"] = get_embedding(item["content"])

with open(
    "/home/toannk/Desktop/Code/video-rag/all_images.json", "w", encoding="utf-8"
) as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

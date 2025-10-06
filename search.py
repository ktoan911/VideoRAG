import json

import faiss
import numpy as np
import torch
from fugashi import Tagger
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

HF_MODEL_PATH = "line-corporation/clip-japanese-base"
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(HF_MODEL_PATH, trust_remote_code=True).to(device)

text_model_name = "pkshatech/GLuCoSE-base-ja-v2"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name)

index = faiss.read_index("embeddings_index.faiss")
with open("all_images.json", "r", encoding="utf-8") as f:
    keyframes = json.load(f)

with open("all_sumaries.json", "r", encoding="utf-8") as f:
    summaries = json.load(f)


def get_embedding(text):
    # Tokenize
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Lấy output từ model
    with torch.no_grad():
        outputs = text_model(**inputs)
        # outputs.last_hidden_state: [batch, seq_len, hidden_dim]
        embeddings = outputs.last_hidden_state.mean(dim=1)  # average pooling
    return embeddings[0].tolist()


def tokenize(text: str):
    return [word.surface for word in tagger(text)]


tagger = Tagger()
corpus = [tokenize(d["summary"]) for d in summaries]
bm25 = BM25Okapi(corpus)


def find_visual_scenes(query: str, k=5):
    text_inputs = tokenizer(
        [query], return_tensors=None, padding=True, truncation=True
    ).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    query_emb = text_features.cpu().numpy().astype("float32")

    D, I = index.search(query_emb, k)
    scores = [float(score) for score in D[0] if score >= 0.25]
    results = [keyframes[idx]["video"] for idx in I[0][: len(scores)]]
    return results


def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def hybrid_search_japanese(query, bm25_top_k=25, top_k=20, alpha=0.5):
    query_embedding = get_embedding(query)
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    top_bm25_idx = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:bm25_top_k]

    combined = []
    for idx in top_bm25_idx:
        bm25_score = bm25_scores[idx]
        dense_score = cosine_similarity(
            query_embedding, summaries[idx]["summary_embedding"]
        )
        score = alpha * bm25_score + (1 - alpha) * dense_score
        combined.append((idx, score))

    final = sorted(combined, key=lambda x: x[1], reverse=True)[:top_k]

    return [summaries[i]["summary"] for i, _ in final]


def rag(query):
    clip_results = list(set(find_visual_scenes(query, k=2)))
    hybrid_results = hybrid_search_japanese(query, bm25_top_k=2, top_k=1)

    return (
        query
        + "\n"
        + "クエリに関連する画像を含む動画: "
        + ", ".join(clip_results)
        + "\n"
        + "クエリに関連する要約を含む動画:\n"
        + ", ".join(hybrid_results)
    )



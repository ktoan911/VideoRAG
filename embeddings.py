import json
import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# === LOAD MODEL ===
HF_MODEL_PATH = "line-corporation/clip-japanese-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(HF_MODEL_PATH, trust_remote_code=True).to(device)

# === LOAD IMAGE FOLDER ===
image_folder = r"Metadata KansaiTV/ドキュメンタリー劇場　３３回　「大阪の残侠　谷長五郎の生活と意見」　1964.11.22.mp4/keyframes"
image_files = [
    f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

batch_size = 32
image_dict = {}

for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i : i + batch_size]
    images = [
        Image.open(os.path.join(image_folder, f)).convert("RGB") for f in batch_files
    ]

    image_inputs = processor(images, return_tensors="pt").to(device)

    with torch.no_grad():
        batch_features = model.get_image_features(**image_inputs)
        batch_features = batch_features / batch_features.norm(p=2, dim=-1, keepdim=True)

    for fname, emb in zip(batch_files, batch_features):
        image_dict[fname] = emb.cpu().numpy().tolist()

# === MERGE VỚI METADATA ===
meta_data_file = r"ドキュメンタリー劇場　３３回　「大阪の残侠　谷長五郎の生活と意見」　1964.11.22.mp4.json"
with open(meta_data_file, "r", encoding="utf-8") as f:
    metadata = json.load(f)

for i in range(len(metadata["content"])):
    fname = metadata["content"][i]["keyframe"]  # sửa theo đúng key trong metadata
    metadata["content"][i]["keyframe_embedding"] = image_dict.get(fname)

# Lưu ra file mới
with open(
    r"ドキュメンタリー劇場　３３回　「大阪の残侠　谷長五郎の生活と意見」　1964.11.22.mp4-embed.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

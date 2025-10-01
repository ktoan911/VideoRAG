# import json
# import os

# # Folder cần lấy tên file
# folder_path = (
#     "/home/toannk/Desktop/Code/videoRAG/embed"  # đổi thành đường dẫn folder của bạn
# )
# images = []
# # Lấy tất cả file trong folder (cả file ảnh và file khác)
# all_files = os.listdir(folder_path)
# for f in all_files:
#     with open(os.path.join(folder_path, f), "r") as file:
#         data = json.load(file)
#     content = data["content"]
#     for i in range(len(content)):
#         t = {}
#         t["video"] = f
#         t["index"] = i
#         t["embeddings"] = content[i]["keyframe_embedding"]

#         images.append(t)

# # Lưu ra file mới
# with open("all_images.json", "w", encoding="utf-8") as f:
#     json.dump(images, f, indent=4, ensure_ascii=False)


import json

import faiss
import numpy as np

with open("all_images.json", "r", encoding="utf-8") as f:
    data_list = json.load(f)

dim = len(data_list[0]["embeddings"])  # kích thước vector
index = faiss.IndexFlatIP(dim)  # inner product
metadata = []  # để lưu JSON: video / file info

# Thêm embeddings vào FAISS
embeddings_np = np.array([d["embeddings"] for d in data_list], dtype="float32")
index.add(embeddings_np)

# Lưu metadata
metadata = [
    {
        "video": d["video"],
        "index": d["index"],
        "start_time": d["start_time"],
        "end_time": d["end_time"],
        "content": d["content"],
    }
    for d in data_list
]

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# Lưu FAISS index
faiss.write_index(index, "embeddings_index.faiss")

print("Đã tạo xong FAISS index và metadata JSON")

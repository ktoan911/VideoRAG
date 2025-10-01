import json
from pathlib import Path

folder_path = Path("/home/toannk/Desktop/Code/videoRAG/embed")
all_images = []
for file_path in folder_path.glob("*.json"):
    print(f"Đang xử lý file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for index, item in enumerate(data["content"]):
            caption = ""
            transcription = ""
            if "caption" in item:
                if "content" in item["caption"]:
                    caption = ". ".join(
                        [k["caption"] for k in item["caption"]["content"]]
                    )
            if "transcription" in item:
                if "content" in item["transcription"]:
                    transcription = ". ".join(
                        [k["content"] for k in item["transcription"]["content"]]
                    )
            all_images.append(
                {
                    "video": file_path.name,
                    "index": index,
                    "embeddings": item["keyframe_embedding"],
                    "start_time": item["start_time"],
                    "end_time": item["end_time"],
                    "content": f"まとめ: {item.get('summary', '')}, キャプション: {caption}, トランスクリプション: {transcription}",
                }
            )

with open("all_images.json", "w", encoding="utf-8") as f:
    json.dump(all_images, f, indent=4, ensure_ascii=False)

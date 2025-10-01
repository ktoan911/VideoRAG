def find_scenes_by_person(person_name: str, semantic_query: str, db: list, top_k: int = 1):
    """
    Tìm kiếm cảnh của một nhân vật cụ thể nói về một chủ đề cụ thể.
    """
    print(f"\n🔎 Tìm cảnh của '{person_name}' nói về '{semantic_query}'")
    
    # 1. Lọc ra các cảnh có sự xuất hiện của nhân vật
    # Việc nhận dạng 'persons_in_scene' được thực hiện ở bước tiền xử lý
    # bằng các thư viện như deepface (khuôn mặt) và pyannote.audio (giọng nói)
    filtered_scenes = [
        scene for scene in db if person_name in scene["persons_in_scene"]
    ]
    
    if not filtered_scenes:
        print(f"❌ Không tìm thấy cảnh nào có sự xuất hiện của '{person_name}'.")
        return

    print(f"ℹ️ Đã tìm thấy {len(filtered_scenes)} cảnh có sự xuất hiện của '{person_name}'. Bắt đầu tìm kiếm ngữ nghĩa...")
    
    # 2. Thực hiện tìm kiếm ngữ nghĩa trên các cảnh đã lọc
    query_embedding = model.encode(semantic_query, convert_to_tensor=True)
    
    # Lấy vector văn bản từ các cảnh đã lọc
    scene_embeddings = torch.tensor([scene["text_embedding"] for scene in filtered_scenes])
    
    # Tính toán độ tương đồng
    cosine_scores = util.cos_sim(query_embedding, scene_embeddings)[0]
    
    # Lấy kết quả tốt nhất
    top_results = torch.topk(cosine_scores, k=min(top_k, len(filtered_scenes)))
    
    print("✅ Kết quả tìm kiếm:")
    for score, idx in zip(top_results[0], top_results[1]):
        scene = filtered_scenes[idx]
        print(f"  - Video: {scene['video_id']}, Cảnh: {scene['scene_id']} (Score: {score:.4f})")
        print(f"    Timestamp: {scene['timestamp'][0]}s - {scene['timestamp'][1]}s")
        print(f"    Transcript liên quan: {scene['transcript']}")

# --- Ví dụ thực thi ---
# Giả lập các text_embedding
for scene in metadata_db:
    scene["text_embedding"] = model.encode(scene["blended_text"])

person_query = "Tanaka Ken"
topic_query = "kỷ niệm thời thơ ấu"
find_scenes_by_person(person_query, topic_query, metadata_db)
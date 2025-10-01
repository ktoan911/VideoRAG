from bertopic import BERTopic

def check_content_duplication(topic_query: str, db: list):
    """
    Kiểm tra xem một chủ đề đã được khai thác chưa bằng BERTopic.
    """
    print(f"\n🔎 Kiểm tra trùng lặp cho chủ đề: '{topic_query}'")
    
    # 1. Lấy tất cả các văn bản mô tả từ DB
    # Giả sử DB này đã được lọc trong 6 tháng qua
    documents = [scene["blended_text"] for scene in db]
    
    # 2. Xây dựng mô hình BERTopic
    # verbose=False để không in ra quá nhiều log
    topic_model = BERTopic(verbose=False)
    topics, _ = topic_model.fit_transform(documents)
    
    print("ℹ️ Các chủ đề chính đã được khai thác:")
    print(topic_model.get_topic_info())
    
    # 3. Tìm chủ đề cho câu truy vấn mới
    # find_topics sẽ trả về topic gần nhất với query
    similar_topics, similarity_scores = topic_model.find_topics(topic_query, top_n=1)
    most_similar_topic_id = similar_topics[0]
    best_score = similarity_scores[0]
    
    # 4. Đánh giá kết quả
    if best_score > 0.7: # Ngưỡng tương đồng có thể tùy chỉnh
        topic_keywords = topic_model.get_topic(most_similar_topic_id)
        print(f"\n✅ Cảnh báo: Chủ đề '{topic_query}' có vẻ đã được khai thác.")
        print(f"   Nó rất giống với Chủ đề {most_similar_topic_id}, với các từ khóa chính: {topic_keywords}")
    else:
        print(f"\n✅ Thông tin: Chủ đề '{topic_query}' có vẻ là một chủ đề mới.")

content_query = "khám phá thành phố Kyoto từ trên cao"
check_content_duplication(content_query, metadata_db)
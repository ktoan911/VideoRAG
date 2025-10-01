from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def sparse_search(query: str, db: list):
    """Mô phỏng tìm kiếm từ khóa (BM25/TF-IDF)."""
    documents = [scene["blended_text"] for scene in db]
    vectorizer = TfidfVectorizer().fit(documents)
    query_vec = vectorizer.transform([query])
    doc_vecs = vectorizer.transform(documents)
    scores = linear_kernel(query_vec, doc_vecs).flatten()
    # Trả về (chỉ số, điểm) đã sắp xếp
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

def dense_search(query: str, db: list):
    """Tìm kiếm vector ngữ nghĩa."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    scene_embeddings = torch.tensor([scene["text_embedding"] for scene in db])
    scores = util.cos_sim(query_embedding, scene_embeddings)[0].tolist()
    # Trả về (chỉ số, điểm) đã sắp xếp
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

def hybrid_search(query: str, db: list, top_k: int = 1):
    """
    Kết hợp kết quả từ sparse và dense search để kiểm chứng thông tin.
    """
    print(f"\n🔎 Tìm kiếm hỗn hợp cho: '{query}'")
    
    # 1. Thực hiện cả hai loại tìm kiếm
    sparse_results = sparse_search(query, db)
    dense_results = dense_search(query, db)
    
    # 2. Tổng hợp kết quả (Reciprocal Rank Fusion - RRF)
    # Đây là một cách đơn giản để tổng hợp: gán điểm dựa trên thứ hạng
    rrf_scores = {}
    k = 60  # Hằng số làm mịn
    
    for rank, (doc_idx, _) in enumerate(sparse_results[:20]):
        if doc_idx not in rrf_scores:
            rrf_scores[doc_idx] = 0
        rrf_scores[doc_idx] += 1 / (k + rank)
        
    for rank, (doc_idx, _) in enumerate(dense_results[:20]):
        if doc_idx not in rrf_scores:
            rrf_scores[doc_idx] = 0
        rrf_scores[doc_idx] += 1 / (k + rank)
        
    # Sắp xếp kết quả cuối cùng
    final_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    
    print("✅ Kết quả kiểm chứng thông tin:")
    for doc_idx, score in final_results[:top_k]:
        scene = db[doc_idx]
        print(f"  - Video: {scene['video_id']}, Cảnh: {scene['scene_id']} (Score RRF: {score:.4f})")
        print(f"    Timestamp: {scene['timestamp'][0]}s - {scene['timestamp'][1]}s")
        print(f"    Bản ghi thoại gốc: {scene['transcript']}")

# --- Ví dụ thực thi ---
metadata_db[2]['transcript'] = "Bộ trưởng X: Về chính sách nhập cư, chúng ta phải cân bằng giữa an ninh và phát triển kinh tế."
metadata_db[2]['blended_text'] = "Trong họp báo ngày 15/3/2024, bộ trưởng X đã phát biểu về chính sách nhập cư. Ông cho rằng cần cân bằng giữa an ninh và kinh tế."
metadata_db[2]['text_embedding'] = model.encode(metadata_db[2]['blended_text'])

fact_check_query = "bộ trưởng X đã nói gì về chính sách nhập cư ngày 15/3/2024?"
hybrid_search(fact_check_query, metadata_db)
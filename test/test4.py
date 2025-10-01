# file: requirements.txt
# --------------------------------------------------
# Các thư viện cần thiết để chạy mã nguồn MVP.
# Cài đặt bằng lệnh: pip install -r requirements.txt
# --------------------------------------------------

streamlit
pandas
numpy
torch
torchvision
Pillow
# Thư viện clip của OpenAI để tạo embedding
# Cài đặt từ github: pip install git+https://github.com/openai/CLIP.git
clip @ git+https://github.com/openai/CLIP.git

```

```python
# file: ingest_data.py
# --------------------------------------------------
# GIAI ĐOẠN 1: MVP - XỬ LÝ VÀ LẬP CHỈ MỤC DỮ LIỆU
# --------------------------------------------------
# Script này mô phỏng quá trình xử lý dữ liệu đầu vào (ingestion).
# Nó đọc file JSON, tạo embedding cho văn bản và hình ảnh bằng mô hình CLIP,
# và lưu trữ siêu dữ liệu cùng với các vector embedding để chuẩn bị cho việc truy xuất.
# Trong một hệ thống thực tế, dữ liệu này sẽ được đẩy vào Elasticsearch và Milvus.[1]
# --------------------------------------------------

import json
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
import os

# --- Cấu hình ---
# Đảm bảo bạn đã tải mô hình CLIP. Lần chạy đầu tiên sẽ tự động tải về.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
print(f"Sử dụng thiết bị: {DEVICE}")

# Đường dẫn đến dữ liệu đầu vào
INPUT_JSON_PATH = 'data.json' # File JSON bạn cung cấp
KEYFRAME_DIR = 'keyframes/'   # Thư mục chứa các keyframe (ví dụ: 'Keyframe-003-.jpg')

# Đường dẫn đến các file đầu ra (mô phỏng cơ sở dữ liệu)
METADATA_OUTPUT_PATH = 'db_metadata.csv'
EMBEDDINGS_OUTPUT_PATH = 'db_embeddings.npz'

def create_super_document(segment):
    """
    Kết hợp tất cả các trường văn bản thành một tài liệu duy nhất để tạo embedding.
    Điều này giúp nắm bắt toàn bộ ngữ nghĩa văn bản của đoạn video.
    """
    texts = [
        segment.get('title', ''),
        segment.get('summary', ''),
    ]
    # Thêm nội dung từ caption
    if 'caption' in segment and 'content' in segment['caption']:
        for cap in segment['caption']['content']:
            texts.append(cap.get('caption', ''))
    # Thêm nội dung từ transcription
    if 'transcription' in segment and 'content' in segment['transcription']:
        for trans in segment['transcription']['content']:
            texts.append(trans.get('content', ''))
    
    return ". ".join(filter(None, texts))

def generate_embeddings_for_data(data):
    """
    Tạo và lưu trữ embeddings cho toàn bộ tập dữ liệu.
    """
    metadata_list =
    text_embeddings =
    image_embeddings =

    # Giả sử dữ liệu đầu vào là một danh sách các dictionary
    if not isinstance(data, list):
        data = [data]

    for i, segment in enumerate(data):
        # --- Xử lý Văn bản ---
        super_doc = create_super_document(segment)
        if super_doc:
            text_tokens = clip.tokenize([super_doc], truncate=True).to(DEVICE)
            with torch.no_grad():
                text_features = MODEL.encode_text(text_tokens)
                text_embeddings.append(text_features.cpu().numpy())
        else:
            # Nếu không có văn bản, sử dụng vector zero
            text_embeddings.append(np.zeros((1, 512)))

        # --- Xử lý Hình ảnh ---
        keyframe_filename = segment.get('keyframe')
        if keyframe_filename:
            image_path = os.path.join(KEYFRAME_DIR, keyframe_filename)
            if os.path.exists(image_path):
                image = PREPROCESS(Image.open(image_path)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    image_features = MODEL.encode_image(image)
                    image_embeddings.append(image_features.cpu().numpy())
            else:
                print(f"Cảnh báo: Không tìm thấy file keyframe '{image_path}'")
                image_embeddings.append(np.zeros((1, 512)))
        else:
            image_embeddings.append(np.zeros((1, 512)))
            
        # --- Lưu trữ Siêu dữ liệu ---
        # Giữ lại các thông tin cần thiết để hiển thị kết quả
        metadata = {
            'id': i,
            'title': segment.get('title', ''),
            'summary': segment.get('summary', ''),
            'keyframe': segment.get('keyframe', ''),
            'start_time': segment.get('start_time', ''),
            'end_time': segment.get('end_time', ''),
        }
        metadata_list.append(metadata)

    # Chuyển đổi danh sách thành các mảng numpy
    text_embeddings_np = np.vstack(text_embeddings)
    image_embeddings_np = np.vstack(image_embeddings)

    # Chuẩn hóa embeddings (quan trọng cho việc tính toán cosine similarity)
    text_embeddings_np /= np.linalg.norm(text_embeddings_np, axis=1, keepdims=True)
    image_embeddings_np /= np.linalg.norm(image_embeddings_np, axis=1, keepdims=True)

    # Lưu siêu dữ liệu và embeddings
    df_metadata = pd.DataFrame(metadata_list)
    df_metadata.to_csv(METADATA_OUTPUT_PATH, index=False)
    np.savez(EMBEDDINGS_OUTPUT_PATH, text=text_embeddings_np, image=image_embeddings_np)

    print(f"Đã xử lý {len(data)} đoạn video.")
    print(f"Siêu dữ liệu đã được lưu tại: {METADATA_OUTPUT_PATH}")
    print(f"Embeddings đã được lưu tại: {EMBEDDINGS_OUTPUT_PATH}")

if __name__ == "__main__":
    # Tạo thư mục keyframes nếu chưa có
    if not os.path.exists(KEYFRAME_DIR):
        os.makedirs(KEYFRAME_DIR)
        print(f"Đã tạo thư mục '{KEYFRAME_DIR}'. Vui lòng đặt các file keyframe của bạn vào đây.")

    # Đọc dữ liệu từ file JSON
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        generate_embeddings_for_data(video_data)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{INPUT_JSON_PATH}'. Vui lòng tạo file này với dữ liệu của bạn.")
    except json.JSONDecodeError:
        print(f"Lỗi: File '{INPUT_JSON_PATH}' không phải là một file JSON hợp lệ.")

```

```python
# file: retrieval_backend.py
# --------------------------------------------------
# GIAI ĐOẠN 1: MVP - LOGIC TRUY XUẤT CỐT LÕI
# --------------------------------------------------
# Script này chứa logic backend để thực hiện tìm kiếm.
# Nó tải siêu dữ liệu và các vector embedding đã được xử lý trước,
# sau đó cung cấp các hàm để tìm kiếm dựa trên văn bản hoặc hình ảnh.
# --------------------------------------------------

import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image

# --- Cấu hình và Tải tài nguyên ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)

METADATA_PATH = 'db_metadata.csv'
EMBEDDINGS_PATH = 'db_embeddings.npz'

# Tải dữ liệu đã được lập chỉ mục
try:
    df_metadata = pd.read_csv(METADATA_PATH)
    embeddings_data = np.load(EMBEDDINGS_PATH)
    text_embeddings_db = embeddings_data['text']
    image_embeddings_db = embeddings_data['image']
    print("Backend: Đã tải thành công siêu dữ liệu và embeddings.")
except FileNotFoundError:
    print("Lỗi Backend: Không tìm thấy file cơ sở dữ liệu. Vui lòng chạy 'ingest_data.py' trước.")
    df_metadata = pd.DataFrame()
    text_embeddings_db, image_embeddings_db = None, None

def search_by_text(query_text, top_k=5):
    """
    Thực hiện tìm kiếm ngữ nghĩa dựa trên một truy vấn văn bản.
    Đây là kịch bản Truy xuất Văn bản-tới-Video (Text-to-Video).
    """
    if text_embeddings_db is None:
        return

    # 1. Mã hóa truy vấn văn bản thành vector
    text_tokens = clip.tokenize([query_text], truncate=True).to(DEVICE)
    with torch.no_grad():
        query_embedding = MODEL.encode_text(text_tokens).cpu().numpy()
    
    # 2. Chuẩn hóa vector truy vấn
    query_embedding /= np.linalg.norm(query_embedding)

    # 3. Tính toán độ tương đồng cosine
    # (Tích vô hướng của các vector đã được chuẩn hóa chính là cosine similarity)
    similarities = (text_embeddings_db @ query_embedding.T).squeeze()

    # 4. Lấy ra top_k kết quả có độ tương đồng cao nhất
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 5. Trả về siêu dữ liệu của các kết quả
    results = df_metadata.iloc[top_indices].to_dict('records')
    for i, res in enumerate(results):
        res['similarity'] = similarities[top_indices[i]]
        
    return results

def search_by_image(query_image, top_k=5):
    """
    Thực hiện tìm kiếm tương tự thị giác dựa trên một hình ảnh đầu vào.
    Đây là kịch bản Truy xuất Hình ảnh-tới-Video (Image-to-Video).
    """
    if image_embeddings_db is None:
        return

    # 1. Xử lý và mã hóa hình ảnh truy vấn
    image = PREPROCESS(query_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        query_embedding = MODEL.encode_image(image).cpu().numpy()

    # 2. Chuẩn hóa vector truy vấn
    query_embedding /= np.linalg.norm(query_embedding)

    # 3. Tính toán độ tương đồng cosine
    similarities = (image_embeddings_db @ query_embedding.T).squeeze()

    # 4. Lấy ra top_k kết quả
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # 5. Trả về siêu dữ liệu
    results = df_metadata.iloc[top_indices].to_dict('records')
    for i, res in enumerate(results):
        res['similarity'] = similarities[top_indices[i]]
        
    return results

```

```python
# file: app.py
# --------------------------------------------------
# GIAI ĐOẠN 1: MVP - GIAO DIỆN NGƯỜI DÙNG
# --------------------------------------------------
# Sử dụng Streamlit để tạo một giao diện web đơn giản cho hệ thống.
# Người dùng có thể nhập văn bản hoặc tải lên hình ảnh để tìm kiếm.
# Chạy ứng dụng bằng lệnh: streamlit run app.py
# Lấy cảm hứng từ các dự án mã nguồn mở tương tự.[2]
# --------------------------------------------------

import streamlit as st
from PIL import Image
import os
import retrieval_backend

# --- Cấu hình Giao diện ---
st.set_page_config(page_title="Hệ thống Truy xuất Video", layout="wide")
st.title("🎬 Hệ thống Truy xuất Video Đa phương thức")
st.markdown("Dựa trên báo cáo nghiên cứu về kiến trúc Hybrid và mô hình Embedding không gian chung.")

KEYFRAME_DIR = 'keyframes/'

def display_results(results):
    """Hàm tiện ích để hiển thị kết quả tìm kiếm."""
    if not results:
        st.warning("Không tìm thấy kết quả nào phù hợp.")
        return

    st.success(f"Tìm thấy {len(results)} kết quả:")
    
    cols = st.columns(len(results))
    for i, res in enumerate(results):
        with cols[i]:
            st.markdown(f"**{res['title']}**")
            
            keyframe_path = os.path.join(KEYFRAME_DIR, str(res['keyframe']))
            if os.path.exists(keyframe_path):
                st.image(keyframe_path, use_column_width=True)
            else:
                st.error(f"Không tìm thấy keyframe: {res['keyframe']}")
            
            st.markdown(f"**Tóm tắt:** {res['summary']}")
            st.info(f"Thời gian: {res['start_time']} - {res['end_time']}")
            st.progress(res['similarity'])
            st.caption(f"Độ tương đồng: {res['similarity']:.4f}")


# --- Tạo các Tab cho từng loại truy vấn ---
tab1, tab2 = st.tabs()

with tab1:
    st.header("Tìm kiếm Video bằng Mô tả Văn bản")
    text_query = st.text_input(
        "Nhập mô tả của bạn:", 
        placeholder="ví dụ: một chương trình nói về giải bóng chày chuyên nghiệp"
    )
    if st.button("Tìm kiếm", key="text_search_btn"):
        if text_query:
            with st.spinner("Đang tìm kiếm..."):
                results = retrieval_backend.search_by_text(text_query)
                display_results(results)
        else:
            st.error("Vui lòng nhập mô tả để tìm kiếm.")

with tab2:
    st.header("Tìm kiếm Video bằng Hình ảnh Tương tự")
    uploaded_file = st.file_uploader(
        "Tải lên một hình ảnh để tìm các video có cảnh quay tương tự:", 
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        query_image = Image.open(uploaded_file)
        st.image(query_image, caption="Hình ảnh truy vấn", width=200)
        
        if st.button("Tìm kiếm", key="image_search_btn"):
            with st.spinner("Đang tìm kiếm..."):
                results = retrieval_backend.search_by_image(query_image)
                display_results(results)

```

```python
# file: advanced_concepts.py
# --------------------------------------------------
# GIAI ĐOẠN 2 & 3: CÁC KHÁI NIỆM NÂNG CAO (MÃ KHÁI NIỆM)
# --------------------------------------------------
# Phần này chứa mã giả và các đoạn mã khái niệm để minh họa cách triển khai
# các kỹ thuật nâng cao đã được thảo luận trong báo cáo nghiên cứu.
# Đây không phải là mã có thể chạy trực tiếp mà là bản thiết kế để phát triển thêm.
# --------------------------------------------------

# --- 1. Truy xuất Kết hợp (Hybrid Search) với Elasticsearch và Milvus ---
# Giả sử bạn đã có các client cho Elasticsearch và Milvus
# from elasticsearch import Elasticsearch
# from pymilvus import MilvusClient

# es_client = Elasticsearch("http://localhost:9200")
# milvus_client = MilvusClient(uri="http://localhost:19530")

def hybrid_search(text_query, ner_filters, top_k=10):
    """
    Minh họa quy trình truy xuất kết hợp đa giai đoạn.
    """
    # Giai đoạn 1: Lọc chính xác bằng Elasticsearch [3]
    # Sử dụng các trường `ner` để thu hẹp không gian tìm kiếm.
    es_query = {
        "query": {
            "bool": {
                "must": [{"match": {"super_document": text_query}}],
                "filter": [{"term": {f"ner.{k}": v}} for k, v in ner_filters.items()]
            }
        },
        "size": 1000 # Lấy một tập ứng viên lớn
    }
    es_results = es_client.search(index="videos", body=es_query)
    candidate_ids = [doc['_id'] for doc in es_results['hits']['hits']]

    if not candidate_ids:
        return

    # Giai đoạn 2: Tìm kiếm ngữ nghĩa trên tập đã lọc với Milvus [4]
    # Mã hóa truy vấn văn bản
    # query_vector = model.encode_text(text_query)
    query_vector = [np.random.rand(512).tolist()] # Vector giả

    # Thực hiện tìm kiếm ANN chỉ trên các ID ứng viên
    milvus_results = milvus_client.search(
        collection_name="videos_collection",
        data=query_vector,
        limit=top_k,
        expr=f"video_id in {candidate_ids}" # Lọc trước khi tìm kiếm
    )
    
    # Giai đoạn 3: Hợp nhất và trả về kết quả cuối cùng
    # (Logic để kết hợp điểm số từ ES và Milvus, ví dụ: RRF [5])
    final_results = milvus_results
    return final_results


# --- 2. Mô hình hóa Thời gian (Temporal Modeling) ---
# import torch.nn as nn

# class TemporalModeler(nn.Module):
#     def __init__(self, embedding_dim, num_heads, num_layers):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, frame_embeddings):
#         # frame_embeddings có shape: (num_frames, batch_size, embedding_dim)
#         temporal_output = self.transformer_encoder(frame_embeddings)
#         return temporal_output

# # So sánh cách tiếp cận
# def get_video_embedding(frames):
#     # frames: danh sách các ảnh PIL
#     # image_encoder: bộ mã hóa hình ảnh của CLIP
#     frame_embeddings = image_encoder(frames) # Shape: [num_frames, embedding_dim]

#     # Cách tiếp cận Mean Pooling (Cơ bản) [6]
#     video_embedding_simple = frame_embeddings.mean(dim=0)

#     # Cách tiếp cận Temporal Transformer (Nâng cao) [7]
#     temporal_model = TemporalModeler(embedding_dim=512, num_heads=8, num_layers=4)
#     # Cần thay đổi shape để phù hợp với Transformer
#     temporal_input = frame_embeddings.unsqueeze(1) # -> [num_frames, 1, embedding_dim]
#     temporal_output = temporal_model(temporal_input)
#     video_embedding_advanced = temporal_output.mean(dim=0).squeeze(0)
    
#     return video_embedding_simple, video_embedding_advanced


# --- 3. Tích hợp Âm thanh với TEFAL (Text-Conditioned Feature Alignment) ---
# Đây là mã khái niệm minh họa kiến trúc TEFAL.[8, 9]

# class CrossAttentionBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

#     def forward(self, query, key_value):
#         # query: embedding văn bản (batch, 1, dim)
#         # key_value: embedding âm thanh/video (batch, seq_len, dim)
#         attn_output, _ = self.multihead_attn(query, key_value, key_value)
#         return attn_output

# class TEFAL_Model(nn.Module):
#     def __init__(self, embed_dim=512, num_heads=8):
#         super().__init__()
#         # Các bộ mã hóa cho từng phương thức
#         self.text_encoder =... # CLIP text encoder
#         self.video_encoder =... # CLIP image encoder (cho từng frame)
#         self.audio_encoder =... # AST model

#         # Các khối cross-attention độc lập
#         self.text_video_aligner = CrossAttentionBlock(embed_dim, num_heads)
#         self.text_audio_aligner = CrossAttentionBlock(embed_dim, num_heads)

#     def forward(self, text, video_frames, audio_spectrogram):
#         # 1. Trích xuất embedding thô
#         text_emb = self.text_encoder(text).unsqueeze(1) # (batch, 1, dim)
#         video_embs = self.video_encoder(video_frames)   # (batch, num_frames, dim)
#         audio_embs = self.audio_encoder(audio_spectrogram) # (batch, num_audio_patches, dim)

#         # 2. Điều chỉnh đặc trưng video và âm thanh bằng văn bản
#         conditioned_video_emb = self.text_video_aligner(query=text_emb, key_value=video_embs)
#         conditioned_audio_emb = self.text_audio_aligner(query=text_emb, key_value=audio_embs)

#         # 3. Hợp nhất các embedding đã được điều chỉnh
#         # TEFAL đề xuất hợp nhất bằng phép cộng đơn giản
#         final_embedding = conditioned_video_emb + conditioned_audio_emb
        
#         return final_embedding.squeeze(1)


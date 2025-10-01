"""
Streamlit app: tìm video theo query và hiển thị kết quả.
Yêu cầu: bạn đã có hàm `search_videos(query: str) -> list[str]` trả về list các path/URL video.
Cách chạy: `streamlit run streamlit_video_search.py`
"""
from pathlib import Path
from typing import List
import streamlit as st
import io

# ---- CONFIG ----
RESULTS_PER_PAGE = 6

# ------------ Placeholder: function của bạn ------------
# Thay bằng hàm thực tế mà bạn có sẵn. Hàm này phải nhận 1 query (str)
# và trả về list các path (có thể là path local hoặc URL) của video.
# Ví dụ:
# def search_videos(query: str) -> List[str]:
#     return ["/path/to/video1.mp4", "/path/to/video2.mp4"]

# Nếu bạn đã có hàm, import nó ở đây: from my_module import search_videos

def search_videos(query: str) -> List[str]:
    """Hàm mẫu — thay bằng hàm thực tế của bạn."""
    # demo: trả list rỗng nếu query rỗng
    if not query:
        return []
    # ví dụ giả lập 8 video local/URL (thay bằng kết quả thực)
    demo_dir = Path("./demo_videos")
    demo = []
    for i in range(1, 9):
        demo.append(str(demo_dir / f"video_{i}.mp4"))
    return demo

# ------------------------------------------------------

@st.cache_data
def run_search(query: str) -> List[str]:
    """Wrapper cached để tránh gọi lại hàm search nhiều lần khi không cần."""
    return search_videos(query)


def read_bytes(path: str) -> bytes:
    """Đọc bytes từ file local. Nếu path là URL, xử lý ở chỗ gọi (Streamlit hỗ trợ URL trực tiếp cho st.video)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File không tồn tại: {path}")
    return p.read_bytes()


def show_paginated_videos(paths: List[str], page: int, per_page: int = RESULTS_PER_PAGE):
    """Hiển thị video theo trang (pagination)."""
    start = page * per_page
    end = start + per_page
    subset = paths[start:end]

    cols = st.columns(2)
    for idx, vid in enumerate(subset):
        col = cols[idx % 2]
        with col:
            st.markdown(f"**Video #{start + idx + 1}**")
            try:
                # Nếu là URL hoặc Streamlit chấp nhận trực tiếp
                st.video(vid)
            except Exception:
                # Thử đọc bytes nếu là file local
                try:
                    data = read_bytes(vid)
                    st.video(data)
                except Exception as e:
                    st.error(f"Không thể phát video: {vid}. Lỗi: {e}")

            # Nút tải xuống (chỉ hoạt động nếu là file local)
            try:
                p = Path(vid)
                if p.exists():
                    data = p.read_bytes()
                    st.download_button(label="Tải xuống", data=data, file_name=p.name)
            except Exception:
                # với URL, user có thể mở link ở tab mới
                st.write(vid)

    # pagination control
    total_pages = (len(paths) - 1) // per_page + 1 if paths else 1
    cols_nav = st.columns([1, 6, 1])
    with cols_nav[0]:
        if st.button("<< Trước") and page > 0:
            st.session_state.page = page - 1
    with cols_nav[2]:
        if st.button("Sau >>") and page < total_pages - 1:
            st.session_state.page = page + 1

    st.caption(f"Trang {page + 1} / {total_pages}")


# --------- Streamlit UI ---------
st.set_page_config(page_title="Tìm video theo query", layout="wide")
st.title("Tìm và xem video theo query")

if "page" not in st.session_state:
    st.session_state.page = 0

# Input
with st.form(key="search_form"):
    query = st.text_input("Nhập query:")
    submitted = st.form_submit_button("Tìm")

if submitted:
    if not query:
        st.warning("Vui lòng nhập query để tìm video.")
    else:
        with st.spinner("Đang tìm..."):
            results = run_search(query)
        st.session_state.results = results
        st.session_state.page = 0

# Nếu đã có kết quả trong state, hiển thị
results = st.session_state.get("results", None)
if results is None:
    st.info("Chưa có kết quả. Nhập từ khóa và nhấn 'Tìm'.")
elif len(results) == 0:
    st.warning("Không tìm thấy video cho query này.")
else:
    # cho phép chọn số kết quả / trang
    per_page = st.selectbox("Số video mỗi trang:", options=[4, 6, 8, 12], index=1)
    # cập nhật hằng số local
    RESULTS_PER_PAGE = per_page
    # show
    show_paginated_videos(results, st.session_state.page, per_page=per_page)


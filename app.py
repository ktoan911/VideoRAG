import streamlit as st

from config import create_gemini_client
from search import rag


@st.cache_resource
def load_model():
    return create_gemini_client("gemini-2.5-flash")


model = load_model()

st.set_page_config(page_title="Chatbot Gemini", page_icon="🤖", layout="centered")

st.title("Chatbot với Gemini")

if "history" not in st.session_state:
    st.session_state.history = []  # [(role, content), ...]

if "real" not in st.session_state:
    st.session_state.real = []  # [(role, content), ...]


def call_gemini(history):
    context = "\n".join([f"{r}: {m}" for r, m in history[-20:]])
    response = model(context)
    return response


for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)


user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    reply = call_gemini(st.session_state.history + [("user", rag(user_input))])
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("assistant", reply))

    st.rerun()

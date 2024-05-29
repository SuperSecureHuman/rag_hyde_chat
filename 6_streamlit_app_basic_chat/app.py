import streamlit as st
import openai
from llama_index.llms.openai import OpenAI

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)

from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import os

st.set_page_config(
    page_title="Chat with Lex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
openai.api_key = os.getenv("OPENAI_API_KEY")
st.title("Chat with Lex 💬🦙")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about the podcast! 🎙️🦙",
        }
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(
        text="Loading and indexing the Streamlit docs hang tight! This should take 1-2 minutes."):
        embedding_model = FastEmbedEmbedding(
            model_name="mixedbread-ai/mxbai-embed-large-v1", max_length=1024)
        vector_store = MilvusVectorStore(
            uri="http://localhost:19530",
            collection_name="podcast_data_head",
            dim=1024,)
        Settings.embed_model = embedding_model
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context)
        return index


index = load_data()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True
    )

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)

import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openai_like import OpenAILike

st.set_page_config(
    page_title="HyDe with Lex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Sidebar inputs
st.sidebar.title("Configuration")
llm_base_url = st.sidebar.text_input(
    "LLM Base URL", placeholder="Enter LLM base URL (OpenAI Compitatible API)"
)
llm_api_key = st.sidebar.text_input(
    "LLM Key", type="password", placeholder="Enter LLM API Key"
)
llm_model = st.sidebar.text_input("LLM Model", placeholder="Enter LLM model name")
llm_temperature = st.sidebar.slider(
    "Temperature", 0.0, 2.0, 0.0, key="llm_temperature_slider"
)
hyde_llm_temperature = st.sidebar.slider(
    "Hyde LLM Temperature", 0.0, 2.0, 1.8, key="hyde_llm_temperature_slider"
)


Settings.llm = OpenAILike(
    model=llm_model,
    api_key=llm_api_key,
    temperature=llm_temperature,
    api_base=llm_base_url,
    is_chat_model=True,
)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about the podcast! 🎙️🦙",
        }
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading the podcast data and the embedding model"):
        embedding_model = FastEmbedEmbedding(
            model_name="mixedbread-ai/mxbai-embed-large-v1", max_length=1024
        )
        vector_store = MilvusVectorStore(
            uri="http://localhost:19530",
            collection_name="podcast_data_head",
            dim=1024,
        )
        Settings.embed_model = embedding_model
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        return index


index = load_data()


# Create the transformative query engine
def create_query_engine(index: VectorStoreIndex):
    hyde_transform = HyDEQueryTransform(include_original=True)
    base_query_engine = index.as_query_engine()
    hyde_query_engine = TransformQueryEngine(base_query_engine, hyde_transform)
    return hyde_query_engine


# Main method
def main():
    st.title("Ask me anything about the podcast! 🎙️🦙")

    query_text = st.text_input(
        "Query",
        key="query_text",
        placeholder="Enter your query here",
        help="Ask your question here.",
    )

    if query_text:
        query_engine = create_query_engine(index)
        query_bundle = HyDEQueryTransform(include_original=True)(query_text)
        hyde_doc = query_bundle.embedding_strs[0]

        st.session_state["hypo_doc"] = hyde_doc

        response = query_engine.query(query_text)
        final_res_str = response

        st.write("### Final result")
        st.write(final_res_str.response)

        # Display the reconstructed hypothetical document
        hypo_doc = st.session_state.get(
            "hypo_doc", "No hypothetical document generated"
        )
        with st.expander("### Generated document"):
            st.write(hypo_doc)

        st.write("### Source Files")

        # Extract source nodes
        source_nodes = final_res_str.source_nodes

        # Sort source nodes by score
        sorted_nodes = sorted(
            source_nodes, key=lambda node: node.dict().get("score", 0), reverse=True
        )

        # Display each source in a dropdown
        for node in sorted_nodes:
            node_dict = node.dict()
            file_name = node_dict["node"]["metadata"].get("file_name", "Unknown")
            text_content = node_dict["node"].get("text", "No content available")
            score = node_dict.get("score", 0)

            with st.expander(f"**{file_name} (score = {score})**"):
                st.write(f"**File Name:** {file_name}")
                st.write(f"**Content:** {text_content}")


if __name__ == "__main__":
    main()

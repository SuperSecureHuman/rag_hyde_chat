from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
)

from llama_index.core.query_engine import CitationQueryEngine
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding


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


query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    citation_chunk_size=512,
)
response = query_engine.query("Whats the best way to learn programming?")
print(response)
print(len(response.source_nodes))
print(response.get_formatted_sources())

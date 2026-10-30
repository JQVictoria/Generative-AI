# 10.5 Retriever - Used Independently (Halaman 119-120)
import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "huggingface"),
)

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = OpenAILike(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    context_window=1_000_000,
    is_chat_model=True,
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.environ["HF_HOME"],
)

documents = [
    Document(text="ChromaDB is an open-source embedding database with a simple Python API.", metadata={"tag": "vector-db"}),
    Document(text="Pinecone is a managed vector database designed for production workloads.", metadata={"tag": "vector-db"}),
    Document(text="FAISS is a library by Meta for efficient similarity search on dense vectors.", metadata={"tag": "vector-search"}),
    Document(text="pgvector adds vector similarity search to PostgreSQL.", metadata={"tag": "vector-db"}),
    Document(text="Weaviate is an open-source vector database with a GraphQL API.", metadata={"tag": "vector-db"}),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

if __name__ == "__main__":
    # Retrieve without synthesising an answer
    nodes = retriever.retrieve("open source vector databases")
    print("Retrieved nodes:")
    for node in nodes:
        print(f" [{node.score:.4f}] {node.text}")
        print(f"  tag: {node.metadata.get('tag')}")

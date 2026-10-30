# 10.7 Persistent Storage (Halaman 122-123)
import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "huggingface"),
)

from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
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

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "storage", "llama_index_store")

documents = [
    Document(text="LlamaIndex persists indexes to disk as JSON files."),
    Document(text="Loading a persisted index avoids re-embedding on every run."),
]

if __name__ == "__main__":
    # Build and save index
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index saved to {PERSIST_DIR}")

    # Load index back - no re-embedding needed
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    loaded_index = load_index_from_storage(storage_context)
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query("How does LlamaIndex handle persistence?")
    print("After reload:", response)

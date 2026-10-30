# 10.4 VectorStoreIndex (Halaman 118-119)
import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "huggingface"),
)

from llama_index.core import VectorStoreIndex, Document, Settings
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
    Document(text="RAG combines retrieval with generation to answer questions from external knowledge."),
    Document(text="Fine-tuning trains a model on new data to improve performance on a specific task."),
    Document(text="Prompt engineering optimises the input to an LLM to get better outputs without any training."),
    Document(text="Vector databases store high-dimensional embeddings and support fast approximate nearest-neighbour search."),
    Document(text="LangChain provides abstractions for building chains, agents, and memory-enabled LLM applications."),
]

# Build index - embeds all documents automatically
index = VectorStoreIndex.from_documents(documents)

# Query engine - retrieves + synthesises answer
query_engine = index.as_query_engine(similarity_top_k=3)

if __name__ == "__main__":
    response = query_engine.query("What is the difference between RAG and fine-tuning?")
    print("Answer:", response)
    print("\nSource nodes:")
    for node in response.source_nodes:
        print(f" Score {node.score:.4f}: {node.text[:60]}...")

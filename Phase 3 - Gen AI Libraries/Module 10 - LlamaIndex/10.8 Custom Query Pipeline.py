# 10.8 Custom Query Pipeline (Halaman 123-124)
import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "huggingface"),
)

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
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
    Document(text="HNSW (Hierarchical Navigable Small World) is the graph algorithm powering most modern vector search."),
    Document(text="IVF (Inverted File Index) partitions vectors into clusters for fast approximate search."),
    Document(text="PQ (Product Quantisation) compresses vectors to reduce memory usage at the cost of recall."),
    Document(text="Exact nearest-neighbour search guarantees correctness but is O(n) - unusable at scale."),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

if __name__ == "__main__":
    # Different response modes
    for mode in [ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE]:
        synthesiser = get_response_synthesizer(response_mode=mode)
        engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesiser)
        resp = engine.query("What algorithms are used in vector search?")
        print(f"[{mode}] {str(resp)[:120]}\n")

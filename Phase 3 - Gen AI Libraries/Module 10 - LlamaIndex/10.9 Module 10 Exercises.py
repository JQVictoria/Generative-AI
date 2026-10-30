import os, re, tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "huggingface"),
)

from llama_index.core import VectorStoreIndex, SummaryIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
Settings.llm = HuggingFaceLLM(
    model_name=model_name,
    tokenizer_name=model_name,
    context_window=2_048,
    max_new_tokens=128,
    generate_kwargs={"do_sample": False},
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.environ["HF_HOME"],
)

print("=" * 60)
print("RUNNING EXERCISES MODULE 10 - LLAMAINDEX")
print("=" * 60)

# --- Exercise 1: Folder .txt Q&A System ---
print("\n--- Exercise 1: Folder .txt Q&A ---")
with tempfile.TemporaryDirectory() as tmp_dir:
    # Buat file dummy
    Path(tmp_dir, "doc1.txt").write_text("LlamaIndex is optimized for document indexing and structure.", encoding="utf-8")
    Path(tmp_dir, "doc2.txt").write_text("ChromaDB is a vector database used for local embeddings storage.", encoding="utf-8")
    
    docs = [Document(text=p.read_text("utf-8"), metadata={"filename": p.name}) for p in Path(tmp_dir).glob("*.txt")]
    engine = VectorStoreIndex.from_documents(docs).as_query_engine(similarity_top_k=2)
    
    def ask(question: str) -> dict:
        resp = engine.query(question)
        sources = list({n.metadata["filename"] for n in resp.source_nodes})
        return {"answer": str(resp).strip(), "sources": sources}
    
    res1 = ask("What is ChromaDB used for?")
    print(f"Q: What is ChromaDB used for?")
    print(f"Answer : {res1['answer']}")
    print(f"Sources: {res1['sources']}")

# --- Exercise 2: VectorStoreIndex vs SummaryIndex ---
print("\n--- Exercise 2: VectorStoreIndex vs SummaryIndex Comparison ---")
sample_paragraphs = [
    Document(text="Section 1: The model achieved 85% accuracy on dataset A."),
    Document(text="Section 2: The model achieved 92% accuracy on dataset B."),
    Document(text="Section 3: Overall conclusion shows higher stability on dataset B.")
]
v_eng = VectorStoreIndex.from_documents(sample_paragraphs).as_query_engine(similarity_top_k=1)
s_eng = SummaryIndex.from_documents(sample_paragraphs).as_query_engine()

q_fact = "What was the accuracy on dataset A?"
q_summary = "Summarize the findings across all sections."

print(f"[Fact Retrieval] Q: {q_fact}")
print(f"  VectorStoreIndex: {str(v_eng.query(q_fact)).strip()}")
print(f"  SummaryIndex    : {str(s_eng.query(q_fact)).strip()}")
print(f"[Synthesis] Q: {q_summary}")
print(f"  VectorStoreIndex: {str(v_eng.query(q_summary)).strip()}")
print(f"  SummaryIndex    : {str(s_eng.query(q_summary)).strip()}")

# --- Exercise 3: Metadata-Filtered Retriever ---
print("\n--- Exercise 3: Metadata-Filtered Retriever ---")
corpus = [
    Document(text="New embedding techniques released.", metadata={"year": 2024, "topic": "embeddings"}),
    Document(text="Old embedding benchmarks overview.", metadata={"year": 2022, "topic": "embeddings"}),
    Document(text="Agent workflow systems.", metadata={"year": 2024, "topic": "agents"}),
]
idx = VectorStoreIndex.from_documents(corpus)
filters = MetadataFilters(filters=[
    ExactMatchFilter(key="topic", value="embeddings"),
    ExactMatchFilter(key="year", value=2024)
])
retriever = idx.as_retriever(similarity_top_k=2, filters=filters)
filtered_nodes = retriever.retrieve("modern embeddings")
print("Retrieved nodes matching topic='embeddings' & year=2024:")
for n in filtered_nodes:
    print(f"  - Text: {n.text} | Metadata: {n.metadata}")

# --- Exercise 4: Two-Stage Pipeline (Top-10 -> LLM Rerank Top-3) ---
print("\n--- Exercise 4: Top-10 Retrieval + LLM Reranking ---")
docs_pool = [Document(text=f"Document {i}: Guide on optimizing transformer speed and memory.") for i in range(1, 12)]
pool_idx = VectorStoreIndex.from_documents(docs_pool)
cand_nodes = VectorIndexRetriever(index=pool_idx, similarity_top_k=5).retrieve("optimizing memory")

for n in cand_nodes:
    score_response = str(Settings.llm.complete(
        f"Score the document's relevance to the query from 0 to 10. Return only one integer.\n"
        f"Query: optimizing memory\nDocument: {n.text}\nScore:"
    ))
    score_match = re.search(r"\b(?:10|[0-9])\b", score_response)
    n.score = float(score_match.group(0)) if score_match else 0.0

reranked = sorted(cand_nodes, key=lambda x: x.score, reverse=True)[:3]
print("Top-3 Reranked Nodes:")
for i, n in enumerate(reranked, 1):
    print(f"  {i}. Score: {n.score} | {n.text}")
print("=" * 60)
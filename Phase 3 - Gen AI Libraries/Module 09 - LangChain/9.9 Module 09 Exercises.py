import os, hashlib
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Pastikan API key OpenRouter terbaca
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY belum terpasang di file .env")

llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
parser = StrOutputParser()

print("=" * 60)
print("RUNNING EXERCISES MODULE 09")
print("=" * 60)

# ============================================================
# Exercise 1: LCEL Fact Extraction & Quiz Generator
# ============================================================
print("\n--- Exercise 1: Facts Extraction & Quiz Generator ---")
facts_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract exactly 3 key facts from the text. Return one fact per line without numbering."),
    ("human", "{text}")
])
quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write 1 quiz question for each fact provided. Return ONLY the questions, one per line."),
    ("human", "{facts}")
])
quiz_chain = (
    facts_prompt 
    | llm 
    | parser 
    | (lambda f: {"facts": f}) 
    | quiz_prompt 
    | llm 
    | parser 
    | RunnableLambda(lambda s: [q.strip() for q in s.strip().split("\n") if q.strip()])
)

sample_doc = (
    "LangChain is a framework for developing applications powered by large language models. "
    "LCEL uses the pipe operator to compose runnables cleanly. "
    "ChromaDB is a lightweight vector database suitable for local development."
)
ex1_result = quiz_chain.invoke({"text": sample_doc})
print(f"Hasil Ekstraksi & Pertanyaan Kuis (List):\n{ex1_result}")

# ============================================================
# Exercise 2: Manual MultiQueryRetriever
# ============================================================
print("\n--- Exercise 2: MultiQueryRetriever ---")
para_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 paraphrased versions of the query. Return one per line, no numbering."),
    ("human", "{query}")
])
para_chain = para_prompt | llm | parser

class MockRetriever:
    """Mock retriever sederhana untuk menguji fungsionalitas tanpa vector DB terpisah"""
    def invoke(self, q: str):
        return [
            Document(page_content=f"Document snippet about: {q[:15]}..."),
            Document(page_content="Static doc: LangChain LCEL makes pipelines composable.")
        ]

def multi_query_retrieve(query: str, retriever) -> list:
    paraphrases_raw = para_chain.invoke({"query": query})
    queries = [query] + [q.strip() for q in paraphrases_raw.strip().split("\n") if q.strip()][:3]
    print(f"Kueri yang dievaluasi: {queries}")
    
    seen, docs = set(), []
    for q in queries:
        for d in retriever.invoke(q):
            h = hashlib.md5(d.page_content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                docs.append(d)
    return docs

ex2_docs = multi_query_retrieve("how does LCEL pipe work?", MockRetriever())
print(f"Total Dokumen Unik Ditemukan: {len(ex2_docs)}")
for i, d in enumerate(ex2_docs, 1):
    print(f"  {i}. {d.page_content}")

# ============================================================
# Exercise 3: Map-Reduce Summarisation
# ============================================================
print("\n--- Exercise 3: Map-Reduce Summarisation ---")
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
map_chain = ChatPromptTemplate.from_messages([
    ("system", "Summarise this text chunk in 1 short sentence:"), 
    ("human", "{c}")
]) | llm | parser

reduce_chain = ChatPromptTemplate.from_messages([
    ("system", "Combine these chunk summaries into a single final summary:"), 
    ("human", "{s}")
]) | llm | parser

def map_reduce_summary(text: str) -> str:
    chunks = splitter.split_text(text)
    print(f"Memecah teks panjang menjadi {len(chunks)} chunks...")
    mapped = map_chain.batch([{"c": c} for c in chunks])
    return reduce_chain.invoke({"s": "\n".join(mapped)})

long_text = (
    "Retrieval-Augmented Generation connects LLMs to external data sources. "
    "It mitigates hallucinations by fetching factual content at inference time. "
    "Vector databases like ChromaDB store high-dimensional embeddings for quick semantic retrieval. "
    "LangChain simplifies this entire workflow with LCEL pipelines and ready-to-use document loaders."
)
ex3_summary = map_reduce_summary(long_text)
print(f"Final Summary:\n{ex3_summary}")

# ============================================================
# Exercise 4: RAG with Groundedness Score
# ============================================================
print("\n--- Exercise 4: RAG Answer with Groundedness Filter ---")
class Groundedness(BaseModel):
    score: int = Field(description="Groundedness score from 0 to 10")
    reason: str = Field(description="Short reason for the score")

eval_chain = ChatPromptTemplate.from_messages([
    ("system", "Score groundedness (0-10) of the answer strictly against the context."),
    ("human", "Context: {context}\n\nAnswer: {answer}")
]) | llm.with_structured_output(Groundedness)

def rag_with_score(question: str, context: str) -> dict:
    ans_chain = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using ONLY the context provided:\n{context}"), 
        ("human", "{question}")
    ]) | llm | parser
    
    answer = ans_chain.invoke({"context": context, "question": question})
    evaluation = eval_chain.invoke({"context": context, "answer": answer})
    return {"answer": answer, "score": evaluation.score, "reason": evaluation.reason}

context_text = "LangChain expression language (LCEL) uses the pipe operator '|' to bind components."
ex4_res = rag_with_score("What operator does LCEL use?", context_text)
print(f"Question : What operator does LCEL use?")
print(f"Answer   : {ex4_res['answer']}")
print(f"Score    : {ex4_res['score']}/10 ({ex4_res['reason']})")
print("\n" + "=" * 60)
import time, json, tempfile
import numpy as np
import chromadb

print("=" * 60)
print("RUNNING EXERCISES MODULE 11 - VECTOR DATABASES")
print("=" * 60)

# --- Exercise 1: VectorDBBenchmark ---
print("\n--- Exercise 1: VectorDB Benchmark ---")
client = chromadb.Client()
col_bench = client.get_or_create_collection("bench")
rng = np.random.default_rng(42)

n_docs, n_queries, dim = 500, 50, 64  # Skala disesuaikan agar berjalan cepat
col_bench.add(
    ids=[f"id_{i}" for i in range(n_docs)],
    documents=[f"Doc {i}" for i in range(n_docs)],
    embeddings=rng.standard_normal((n_docs, dim)).tolist()
)
query_vecs = rng.standard_normal((n_queries, dim)).tolist()
latencies = []
t0 = time.perf_counter()
for q in query_vecs:
    t_start = time.perf_counter()
    col_bench.query(query_embeddings=[q], n_results=5)
    latencies.append((time.perf_counter() - t_start) * 1000)
total_time = time.perf_counter() - t0

print(f"Inserted: {n_docs} docs | Queries: {n_queries}")
print(f"Avg Latency : {np.mean(latencies):.2f} ms")
print(f"P95 Latency : {np.percentile(latencies, 95):.2f} ms")
print(f"Throughput  : {n_queries / total_time:.2f} queries/sec")

# --- Exercise 2: MultiCollectionSearch ---
print("\n--- Exercise 2: Multi-Collection Search ---")
col_a = client.get_or_create_collection("kb_tech")
col_b = client.get_or_create_collection("kb_faq")

col_a.add(ids=["t1"], documents=["Tech: RAG architecture overview"], embeddings=rng.standard_normal((1, dim)).tolist())
col_b.add(ids=["f1"], documents=["FAQ: How to reset passwords"], embeddings=rng.standard_normal((1, dim)).tolist())

def multi_search(cols, q_emb, k=2):
    merged = []
    for c in cols:
        res = c.query(query_embeddings=[q_emb], n_results=min(k, c.count()))
        for doc, dist in zip(res["documents"][0], res["distances"][0]):
            merged.append({"col": c.name, "doc": doc, "score": round(1 - dist / 2, 4)})
    return sorted(merged, key=lambda x: x["score"], reverse=True)[:k]

q_search = rng.standard_normal(dim).tolist()
res2 = multi_search([col_a, col_b], q_search, k=2)
for item in res2:
    print(f"  [{item['col']}] Score: {item['score']} | {item['doc']}")

# --- Exercise 3: TenantVectorStore ---
print("\n--- Exercise 3: Tenant Isolation ---")
tenant_1 = client.get_or_create_collection("tenant_user123")
tenant_2 = client.get_or_create_collection("tenant_user456")

tenant_1.add(ids=["d1"], documents=["Private note User 123"], embeddings=rng.standard_normal((1, dim)).tolist())
tenant_2.add(ids=["d2"], documents=["Private note User 456"], embeddings=rng.standard_normal((1, dim)).tolist())

res_t1 = tenant_1.query(query_embeddings=[q_search], n_results=1)
print(f"Search in tenant_user123 collection: {res_t1['documents'][0]}")

# --- Exercise 4: Collection JSON Backup & Restore ---
print("\n--- Exercise 4: JSON Backup & Restore ---")
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_json:
    tmp_path = tmp_json.name

# Export
data = col_a.get(include=["documents", "embeddings", "metadatas"])
records = [{"id": i, "text": d, "embedding": list(e), "metadata": m or {}} for i, d, e, m in zip(data["ids"], data["documents"], data["embeddings"], data["metadatas"])]
with open(tmp_path, "w") as f:
    json.dump(records, f)
print(f"Exported {len(records)} records to {tmp_path}")

# Import back to new collection
with open(tmp_path) as f:
    loaded_records = json.load(f)
col_restored = client.get_or_create_collection("kb_tech_restored")
col_restored.add(
    ids=[r["id"] for r in loaded_records],
    documents=[r["text"] for r in loaded_records],
    embeddings=[r["embedding"] for r in loaded_records]
)
print(f"Restored count in new collection: {col_restored.count()}")
print("=" * 60)
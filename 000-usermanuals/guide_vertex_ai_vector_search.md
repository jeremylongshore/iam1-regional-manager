# Vertex AI Vector Search: Comprehensive User Manual for AI Agents

**Version:** 1.0
**Last Updated:** 2025-11-12
**Target Audience:** AI Agents learning RAG and search technologies on Google Cloud

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [What is Vertex AI Vector Search](#what-is-vertex-ai-vector-search)
3. [When to Use Vector Search](#when-to-use-vector-search)
4. [Core Concepts](#core-concepts)
5. [Architecture](#architecture)
6. [Quickstart (10 Minutes)](#quickstart-10-minutes)
7. [Complete Implementation Guide](#complete-implementation-guide)
8. [Hybrid Search (Semantic + Keyword)](#hybrid-search-semantic--keyword)
9. [Performance Tuning](#performance-tuning)
10. [Cost Optimization](#cost-optimization)
11. [Integration Patterns](#integration-patterns)
12. [Troubleshooting](#troubleshooting)
13. [Best Practices](#best-practices)
14. [Anti-Patterns](#anti-patterns)
15. [Real-World Examples](#real-world-examples)

---

## Executive Summary

**Vertex AI Vector Search** (formerly Matching Engine) is Google Cloud's fully-managed, highly scalable vector database powered by the **ScaNN algorithm** - one of the most efficient Approximate Nearest Neighbor (ANN) algorithms in the industry.

### Key Capabilities

- **Millisecond Latency**: Find similar items from billions of embeddings in 10-50ms
- **Massive Scale**: Handle billions of vectors with horizontal scaling
- **High Availability**: 99.5% SLA with auto-scaling and auto-healing
- **Hybrid Search**: Combine semantic search (dense embeddings) with keyword search (sparse embeddings)
- **Streaming Updates**: Real-time index updates for dynamic data
- **Multi-Modal**: Support text, image, audio, video embeddings

### What You'll Learn

- How to build and deploy vector search indexes
- Semantic search vs keyword search vs hybrid search
- Performance optimization for production workloads
- Integration with LLMs for RAG applications
- Cost-effective architecture patterns

---

## What is Vertex AI Vector Search

### The Problem It Solves

Traditional databases use exact matching (SQL WHERE clauses, keyword search). But modern AI applications need **semantic similarity search**:

- Find products similar to "red running shoes" → return "crimson athletic sneakers"
- Find documents about "climate change" → return docs about "global warming"
- Recommend movies similar to user's watch history
- Detect anomalies in sensor data

### How It Works

```
┌─────────────────────────────────────────────────────┐
│  1. EMBEDDING GENERATION                            │
│  Text/Image → ML Model → Vector (e.g., 768 dims)   │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  2. INDEX BUILDING                                  │
│  Vectors → ScaNN Algorithm → Tree Structure        │
│  (Quantization + Tree Partitioning)                │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  3. QUERY                                           │
│  Query Vector → ANN Search → Top-K Similar Items   │
│  (10-50ms for billions of vectors)                 │
└─────────────────────────────────────────────────────┘
```

### The ScaNN Advantage

Google's ScaNN (Scalable Nearest Neighbors) algorithm provides:

1. **Speed**: 2-5x faster than alternatives (FAISS, Annoy, HNSW)
2. **Accuracy**: Better recall at same latency
3. **Scalability**: Proven at Google Search and YouTube scale

---

## When to Use Vector Search

### ✅ Ideal Use Cases

| Use Case | Description | Example |
|----------|-------------|---------|
| **Semantic Search** | Find conceptually similar items | "Find docs about AI" → returns docs about "machine learning", "neural networks" |
| **RAG (Retrieval-Augmented Generation)** | Ground LLM responses in real data | Chatbot retrieves relevant docs before generating answer |
| **Recommendation Systems** | Find similar products/content | "Customers who liked X also liked Y" |
| **Duplicate Detection** | Find near-duplicates | Identify similar support tickets |
| **Anomaly Detection** | Find outliers | Detect unusual log patterns |
| **Multi-Modal Search** | Search across modalities | Find images similar to text description |

### ❌ NOT Ideal For

- **Exact keyword matching** → Use Vertex AI Search or Elasticsearch
- **Structured queries** → Use BigQuery or Cloud SQL
- **Graph relationships** → Use Neo4j or graph databases
- **Very small datasets** (<1000 items) → Simple in-memory search is faster

---

## Core Concepts

### 1. Embeddings

**Embeddings** are numerical representations of data (text, images, etc.) in a high-dimensional space.

```python
# Example: Text embedding
text = "Google Cloud Vector Search"
embedding = [0.123, -0.456, 0.789, ..., 0.234]  # 768 dimensions

# Similar texts have similar embeddings
text1 = "machine learning"     # [0.5, 0.3, -0.2, ...]
text2 = "artificial intelligence"  # [0.48, 0.32, -0.18, ...]  ← Close in space!
text3 = "pizza recipe"         # [-0.9, 0.1, 0.8, ...]   ← Far in space
```

**Types of Embeddings:**

- **Dense Embeddings**: All dimensions have values (semantic meaning)
  - Example: `[0.123, -0.456, 0.789, ...]` (768 values)
  - Generated by: Vertex AI Embeddings API, Sentence Transformers, etc.

- **Sparse Embeddings**: Most dimensions are zero (keyword frequency)
  - Example: `{"values": [0.8, 0.3], "dimensions": [42, 156]}` (only 2 non-zero out of 10,000)
  - Generated by: TF-IDF, BM25, SPLADE algorithms

### 2. Distance Metrics

How similarity is measured:

| Metric | Formula | When to Use | Range |
|--------|---------|-------------|-------|
| **DOT_PRODUCT** | `sum(a[i] * b[i])` | Normalized embeddings (most common) | Higher = more similar |
| **COSINE** | `dot(a,b) / (norm(a) * norm(b))` | Non-normalized embeddings | -1 to 1 (1 = identical) |
| **EUCLIDEAN** | `sqrt(sum((a[i] - b[i])^2))` | Absolute distance matters | Lower = more similar |

**Recommendation**: Use `DOT_PRODUCT_DISTANCE` with Vertex AI Embeddings API (embeddings are pre-normalized).

### 3. Index Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Tree-AH** | Tree-based with Asymmetric Hashing | General purpose (recommended for most cases) |
| **Brute Force** | Exact search | Small datasets (<10K items), 100% recall required |

### 4. Index Update Methods

| Method | Latency | Use Case |
|--------|---------|----------|
| **BATCH_UPDATE** | Minutes to hours | Static catalogs, periodic updates |
| **STREAM_UPDATE** | Seconds | Dynamic inventory, real-time recommendations |

### 5. Endpoints

| Type | Description | Access |
|------|-------------|--------|
| **Public Endpoint** | Accessible via public internet (IAM-secured) | Recommended for most use cases |
| **VPC Endpoint** | Private network access only | Strict security requirements |

---

## Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     YOUR APPLICATION                         │
│  (Python, Java, Node.js, etc.)                              │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ API Request (gRPC/REST)
                         ↓
┌──────────────────────────────────────────────────────────────┐
│              VERTEX AI INDEX ENDPOINT                        │
│  - Load Balancer                                            │
│  - Auto-scaling (1-100+ nodes)                              │
│  - Health Monitoring                                        │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                 VECTOR SEARCH INDEX                          │
│  - ScaNN Algorithm                                          │
│  - Quantized Vectors                                        │
│  - Tree Structure                                           │
│  - Metadata Storage                                         │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│              CLOUD STORAGE (GCS)                            │
│  - Embedding Data (JSONL/Avro)                             │
│  - Index Snapshots                                         │
└──────────────────────────────────────────────────────────────┘
```

### Component Details

#### Index
- Stores vectors and metadata
- Built from embeddings in Cloud Storage
- Immutable (batch) or mutable (streaming)
- Supports billions of vectors

#### Index Endpoint
- Server instance that serves queries
- Auto-scales based on QPS
- High availability (99.5% SLA)
- Supports multiple deployed indexes

#### Deployed Index
- Association between Index and Endpoint
- Can configure replicas, machine type
- Supports A/B testing (deploy multiple versions)

---

## Quickstart (10 Minutes)

### Prerequisites

```bash
# Install SDK
pip install google-cloud-aiplatform

# Set environment
export PROJECT_ID="your-project-id"
export LOCATION="us-central1"

# Enable APIs
gcloud services enable aiplatform.googleapis.com storage.googleapis.com

# Authenticate
gcloud auth application-default login
```

### Step 1: Prepare Embeddings (2 minutes)

```python
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
import json

# Initialize
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Get embeddings
model = TextEmbeddingModel.from_pretrained("text-embedding-005")
texts = [
    "Google Cloud Platform",
    "Amazon Web Services",
    "Microsoft Azure",
    "Pizza recipe",
]

embeddings = []
for i, text in enumerate(texts):
    emb = model.get_embeddings([text])[0].values
    embeddings.append({"id": str(i), "embedding": emb})

# Save to JSONL
with open("embeddings.json", "w") as f:
    for item in embeddings:
        f.write(json.dumps(item) + "\n")

# Upload to GCS
!gsutil mb gs://{PROJECT_ID}-vectors
!gsutil cp embeddings.json gs://{PROJECT_ID}-vectors/
```

### Step 2: Create Index (2 minutes build time)

```python
# Create index
my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="quickstart-index",
    contents_delta_uri=f"gs://{PROJECT_ID}-vectors",
    dimensions=768,
    approximate_neighbors_count=10,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
)

print(f"Index created: {my_index.resource_name}")
```

### Step 3: Deploy Index (30 minutes first time, seconds after)

```python
# Create endpoint
my_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="quickstart-endpoint",
    public_endpoint_enabled=True
)

# Deploy index to endpoint
my_endpoint.deploy_index(
    index=my_index,
    deployed_index_id="quickstart_deployed"
)

print(f"Endpoint ready: {my_endpoint.resource_name}")
```

### Step 4: Query (< 50ms)

```python
# Generate query embedding
query_text = "Google Cloud"
query_emb = model.get_embeddings([query_text])[0].values

# Search
response = my_endpoint.find_neighbors(
    deployed_index_id="quickstart_deployed",
    queries=[query_emb],
    num_neighbors=3
)

# Results
for neighbor in response[0]:
    print(f"ID: {neighbor.id}, Distance: {neighbor.distance}")

# Output:
# ID: 0, Distance: 1.0000  (Google Cloud Platform - exact match)
# ID: 1, Distance: 0.7234  (Amazon Web Services - similar)
# ID: 2, Distance: 0.6891  (Microsoft Azure - similar)
```

🎉 **Done!** You've built your first semantic search engine.

---

## Complete Implementation Guide

### Data Format Requirements

#### JSONL Format (Recommended)

```json
{"id": "1", "embedding": [0.1, 0.2, ...], "restricts": [{"namespace": "category", "allow": ["tech"]}]}
{"id": "2", "embedding": [0.3, 0.4, ...], "restricts": [{"namespace": "category", "allow": ["food"]}]}
```

**Required Fields:**
- `id` (string): Unique identifier
- `embedding` (array of floats): Dense vector

**Optional Fields:**
- `restricts`: Filtering/namespacing
- `crowding_tag`: Group similar items
- `numeric_restricts`: Numeric filtering
- `sparse_embedding`: For hybrid search (see below)

#### Avro Format (For Large Scale)

```python
# Schema
{
    "type": "record",
    "name": "embedding",
    "fields": [
        {"name": "id", "type": "string"},
        {"name": "embedding", "type": {"type": "array", "items": "double"}}
    ]
}
```

### Creating Production Indexes

#### Batch Index (Static Data)

```python
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Create batch index
batch_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="production-batch-index",
    contents_delta_uri="gs://my-bucket/embeddings/",
    dimensions=768,
    approximate_neighbors_count=150,  # Higher for better recall
    distance_measure_type="DOT_PRODUCT_DISTANCE",

    # Advanced configs
    shard_size="SHARD_SIZE_SMALL",  # or MEDIUM, LARGE
    index_update_method="BATCH_UPDATE",

    # Filtering support
    enable_crowding=True,
    enable_restricts=True,
)
```

**Build Time Estimates:**
- 1M vectors: ~30 minutes
- 10M vectors: ~2 hours
- 100M vectors: ~8 hours

#### Streaming Index (Dynamic Data)

```python
# Create streaming index
stream_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="production-stream-index",
    contents_delta_uri="gs://my-bucket/embeddings/",  # Initial load
    dimensions=768,
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",

    # Enable streaming
    index_update_method="STREAM_UPDATE",
)

# Add vectors in real-time
new_vectors = [
    {"datapoint_id": "item_1001", "feature_vector": [0.1, 0.2, ...]},
    {"datapoint_id": "item_1002", "feature_vector": [0.3, 0.4, ...]},
]

stream_index.upsert_datapoints(datapoints=new_vectors)  # Updates in seconds

# Remove vectors
stream_index.remove_datapoints(datapoint_ids=["item_999"])
```

### Deploying Endpoints

#### Basic Deployment

```python
# Create endpoint
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="prod-endpoint",
    public_endpoint_enabled=True,
)

# Deploy with auto-scaling
endpoint.deploy_index(
    index=my_index,
    deployed_index_id="v1_deployed",

    # Auto-scaling config
    min_replica_count=2,  # High availability
    max_replica_count=10,  # Handle traffic spikes

    # Machine type
    machine_type="n1-standard-16",  # More CPUs = lower latency
)
```

#### Advanced Deployment (A/B Testing)

```python
# Deploy index V1
endpoint.deploy_index(
    index=index_v1,
    deployed_index_id="v1",
    min_replica_count=2,
    max_replica_count=5,
)

# Deploy index V2 for A/B testing
endpoint.deploy_index(
    index=index_v2,
    deployed_index_id="v2",
    min_replica_count=1,
    max_replica_count=3,
)

# Route 90% traffic to V1, 10% to V2
# (Traffic routing configured in application layer)
```

### Advanced Query Patterns

#### Basic Query

```python
response = endpoint.find_neighbors(
    deployed_index_id="v1_deployed",
    queries=[query_embedding],
    num_neighbors=10
)
```

#### Query with Filtering

```python
# Restrict search to specific namespace
response = endpoint.find_neighbors(
    deployed_index_id="v1_deployed",
    queries=[query_embedding],
    num_neighbors=10,

    # Only search items with category="electronics"
    restricts=[{
        "namespace": "category",
        "allow_list": ["electronics"]
    }]
)
```

#### Batch Queries (Better Throughput)

```python
# Query multiple embeddings at once
query_embeddings = [emb1, emb2, emb3]  # 3 different queries

responses = endpoint.find_neighbors(
    deployed_index_id="v1_deployed",
    queries=query_embeddings,
    num_neighbors=10
)

# responses[0] = results for emb1
# responses[1] = results for emb2
# responses[2] = results for emb3
```

---

## Hybrid Search (Semantic + Keyword)

Hybrid search combines **dense embeddings** (semantic) with **sparse embeddings** (keywords) for superior search quality.

### Why Hybrid Search?

**Semantic Search Limitations:**
- Can't find exact product codes: "SKU-12345"
- Struggles with new terms: "iPhone 15 Pro" (if trained before release)
- Misses exact brand names: "Patagonia" vs "outdoor clothing"

**Keyword Search Limitations:**
- No semantic understanding: "car" ≠ "vehicle"
- Can't handle synonyms: "buy" ≠ "purchase"

**Hybrid Search = Best of Both Worlds** ✨

### Implementation

#### Step 1: Generate Sparse Embeddings

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Train TF-IDF vectorizer on your corpus
corpus = ["Google Cloud Vector Search", "Amazon S3 Storage", ...]
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

# Generate sparse embedding
def get_sparse_embedding(text):
    tfidf_vector = vectorizer.transform([text])
    values = []
    dims = []
    for i, val in enumerate(tfidf_vector.data):
        values.append(float(val))
        dims.append(int(tfidf_vector.indices[i]))
    return {"values": values, "dimensions": dims}

sparse_emb = get_sparse_embedding("Google Cloud")
# {"values": [0.8, 0.6], "dimensions": [42, 156]}
```

#### Step 2: Create Hybrid Index

```json
{
  "id": "1",
  "embedding": [0.1, 0.2, 0.3, ...],  // Dense embedding (semantic)
  "sparse_embedding": {                // Sparse embedding (keywords)
    "values": [0.8, 0.6, 0.3],
    "dimensions": [42, 156, 891]
  }
}
```

```python
# Create hybrid index (supports both dense and sparse)
hybrid_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="hybrid-index",
    contents_delta_uri="gs://my-bucket/hybrid-embeddings/",
    dimensions=768,
    approximate_neighbors_count=10,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
)
```

#### Step 3: Hybrid Query

```python
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import HybridQuery

# Generate both dense and sparse embeddings for query
query_text = "cloud storage"
dense_emb = dense_model.get_embeddings([query_text])[0].values
sparse_emb = get_sparse_embedding(query_text)

# Create hybrid query
hybrid_query = HybridQuery(
    dense_embedding=dense_emb,
    sparse_embedding_values=sparse_emb["values"],
    sparse_embedding_dimensions=sparse_emb["dimensions"],

    # Weight between semantic (1.0) and keyword (0.0)
    rrf_ranking_alpha=0.5  # 0.5 = equal weight
)

# Search
response = endpoint.find_neighbors(
    deployed_index_id="hybrid_deployed",
    queries=[hybrid_query],
    num_neighbors=10
)

# Each result has both distances
for neighbor in response[0]:
    print(f"ID: {neighbor.id}")
    print(f"  Dense distance: {neighbor.distance}")
    print(f"  Sparse distance: {neighbor.sparse_distance}")
```

### RRF (Reciprocal Rank Fusion)

The `rrf_ranking_alpha` parameter controls how results are merged:

| Alpha | Behavior |
|-------|----------|
| `1.0` | Pure semantic search (ignore keywords) |
| `0.5` | Equal weight to semantic and keywords |
| `0.0` | Pure keyword search (ignore semantics) |

**Recommendation**: Start with `0.5`, tune based on evaluation metrics.

---

## Performance Tuning

### Latency Optimization

#### 1. Index Configuration

```python
# Lower latency config
low_latency_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="low-latency-index",
    contents_delta_uri="gs://my-bucket/embeddings/",
    dimensions=768,

    # Trade recall for speed
    approximate_neighbors_count=50,  # Lower = faster
    leaf_node_embedding_count=500,   # Lower = faster
    leaf_nodes_to_search_percent=5,  # Lower = faster
)
```

**Latency vs Recall Trade-off:**
- Lower `leaf_nodes_to_search_percent` → Faster, lower recall
- Higher `leaf_nodes_to_search_percent` → Slower, higher recall

#### 2. Endpoint Configuration

```python
# Use more powerful machines
endpoint.deploy_index(
    index=my_index,
    deployed_index_id="fast_deployed",

    machine_type="n1-highmem-16",  # More memory = faster
    min_replica_count=3,            # More replicas = better latency
)
```

#### 3. Query Optimization

```python
# Batch queries for better throughput
queries = [emb1, emb2, emb3]  # Process 3 queries at once
responses = endpoint.find_neighbors(queries=queries, num_neighbors=10)

# Reduce neighbors
response = endpoint.find_neighbors(
    queries=[query_emb],
    num_neighbors=10  # Fewer neighbors = faster
)
```

### Throughput Optimization

```python
# High throughput config
endpoint.deploy_index(
    index=my_index,
    deployed_index_id="high_throughput",

    # Auto-scale aggressively
    min_replica_count=5,
    max_replica_count=50,

    # Enable auto-scaling based on CPU
    enable_access_logging=True,
)
```

### Recall Optimization

```python
# High recall config
high_recall_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="high-recall-index",
    contents_delta_uri="gs://my-bucket/embeddings/",
    dimensions=768,

    # Trade speed for accuracy
    approximate_neighbors_count=200,  # Higher = better recall
    leaf_node_embedding_count=1000,   # Higher = better recall
    leaf_nodes_to_search_percent=20,  # Higher = better recall
)
```

**Benchmark Results** (10M vectors, 768 dims):

| Config | QPS | P50 Latency | P99 Latency | Recall@10 |
|--------|-----|-------------|-------------|-----------|
| Low Latency | 1000 | 15ms | 35ms | 0.85 |
| Balanced | 500 | 25ms | 60ms | 0.92 |
| High Recall | 200 | 50ms | 120ms | 0.97 |

---

## Cost Optimization

### Pricing Model

**Index Storage:**
- $0.40 per GB per month

**Index Endpoints:**
- Machine type pricing (e.g., n1-standard-16: ~$0.76/hour)
- Auto-scaling: Pay only for active replicas

**Queries:**
- Included in endpoint cost (no per-query fee)

### Cost Reduction Strategies

#### 1. Right-Size Machine Types

```python
# Development/Low-traffic
machine_type="n1-standard-4"  # $0.19/hour

# Production/Medium-traffic
machine_type="n1-standard-16"  # $0.76/hour

# High-traffic/Low-latency
machine_type="n1-highmem-32"  # $1.90/hour
```

#### 2. Auto-Scaling

```python
# Scale to zero during off-hours
endpoint.deploy_index(
    index=my_index,
    deployed_index_id="cost_optimized",

    min_replica_count=1,   # Minimum availability
    max_replica_count=10,  # Handle peak traffic
)
```

#### 3. Reduce Index Size

```python
# Use smaller dimensions (if acceptable)
# 768 dims → 384 dims = 50% storage savings

# Use quantization (automatic in ScaNN)
# float32 → int8 = 75% storage savings (built-in)

# Remove unused metadata
# Fewer restricts/crowding tags = less storage
```

#### 4. Batch Updates

```python
# Batch index updates to reduce rebuild costs
# Update once per day instead of continuous streaming
# Batch cost: ~$5 per rebuild
# Streaming cost: $0.40/GB/month + compute
```

### Cost Example

**Scenario**: E-commerce with 1M product embeddings

- **Index Size**: 1M vectors × 768 dims × 4 bytes = 3 GB
- **Storage**: 3 GB × $0.40 = **$1.20/month**
- **Endpoint**: 2 replicas × n1-standard-16 × 730 hours = **$1,109/month**
- **Total**: **~$1,110/month**

**Optimization**: Use n1-standard-4 with auto-scaling (1-5 replicas):
- **Minimum**: 1 replica × $0.19/hour × 730 = **$139/month**
- **Average** (2.5 replicas): **$347/month**
- **Savings**: 69% reduction

---

## Integration Patterns

### Pattern 1: RAG with LLMs

```python
from vertexai.language_models import TextGenerationModel

# 1. Convert user query to embedding
query = "How do I use Cloud Storage?"
query_emb = embedding_model.get_embeddings([query])[0].values

# 2. Retrieve relevant documents
response = endpoint.find_neighbors(
    deployed_index_id="docs_deployed",
    queries=[query_emb],
    num_neighbors=5
)

# 3. Fetch document content
docs = []
for neighbor in response[0]:
    doc_content = fetch_document(neighbor.id)  # Your function
    docs.append(doc_content)

# 4. Generate answer with LLM
context = "\n\n".join(docs)
prompt = f"""Answer this question using only the provided context.

Question: {query}

Context:
{context}

Answer:"""

llm = TextGenerationModel.from_pretrained("text-bison@002")
answer = llm.predict(prompt).text

print(answer)
```

### Pattern 2: Two-Tower Recommendation

```python
# Separate embeddings for users and items
user_embedding = user_model.encode(user_features)
item_embeddings = item_model.encode(item_catalog)

# Store item embeddings in Vector Search
# Query with user embedding to get recommendations
recommendations = endpoint.find_neighbors(
    deployed_index_id="items_deployed",
    queries=[user_embedding],
    num_neighbors=20
)
```

### Pattern 3: Multi-Stage Ranking

```python
# Stage 1: Fast retrieval with Vector Search (1000 candidates)
candidates = endpoint.find_neighbors(
    deployed_index_id="products_deployed",
    queries=[query_emb],
    num_neighbors=1000  # Broad recall
)

# Stage 2: Re-rank with heavy model (top 20)
from vertexai.aiplatform import Ranking

reranked = Ranking.rerank(
    query=query_text,
    candidates=[c.id for c in candidates[0][:100]],
    top_n=20
)

# Stage 3: Business logic (filters, boosts, personalization)
final_results = apply_business_rules(reranked)
```

### Pattern 4: Hybrid Search Application

```python
class HybridSearchEngine:
    def __init__(self, endpoint, deployed_index_id):
        self.endpoint = endpoint
        self.deployed_index_id = deployed_index_id
        self.dense_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        self.sparse_vectorizer = load_tfidf_vectorizer()

    def search(self, query, semantic_weight=0.5, top_k=10):
        # Generate embeddings
        dense_emb = self.dense_model.get_embeddings([query])[0].values
        sparse_emb = self._get_sparse_embedding(query)

        # Create hybrid query
        hybrid_query = HybridQuery(
            dense_embedding=dense_emb,
            sparse_embedding_values=sparse_emb["values"],
            sparse_embedding_dimensions=sparse_emb["dimensions"],
            rrf_ranking_alpha=semantic_weight
        )

        # Search
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[hybrid_query],
            num_neighbors=top_k
        )

        return response[0]

# Usage
engine = HybridSearchEngine(endpoint, "hybrid_deployed")
results = engine.search("cloud storage", semantic_weight=0.7, top_k=20)
```

---

## Troubleshooting

### Common Issues

#### Issue: Index creation fails with "Invalid dimensions"

**Cause**: Dimension mismatch between declared and actual embeddings.

**Solution**:
```python
# Check your embedding dimensions
embedding = model.get_embeddings(["test"])[0].values
print(f"Actual dimensions: {len(embedding)}")  # Should match index dimensions

# Common models:
# text-embedding-005: 768 dims
# textembedding-gecko@003: 768 dims
# multimodalembedding@001: 1408 dims
```

#### Issue: Query returns no results

**Cause**: Distance threshold too strict, or embeddings not indexed.

**Solution**:
```python
# Check if index has data
index_stats = my_index.to_dict()
print(index_stats)  # Look for deployed_indexes count

# Try query without filters
response = endpoint.find_neighbors(
    deployed_index_id="my_deployed",
    queries=[query_emb],
    num_neighbors=10
    # Remove any restricts/filters for debugging
)

# Check distance values
for n in response[0]:
    print(f"ID: {n.id}, Distance: {n.distance}")
# Low distances (< 0.5) suggest poor matches
```

#### Issue: High latency (> 100ms)

**Causes**:
1. Too many neighbors requested
2. Index not optimized
3. Endpoint under-provisioned

**Solutions**:
```python
# 1. Reduce neighbors
num_neighbors=10  # Instead of 100

# 2. Optimize index
leaf_nodes_to_search_percent=5  # Lower value

# 3. Scale up endpoint
endpoint.mutate_deployed_index(
    deployed_index_id="my_deployed",
    min_replica_count=3  # Add more replicas
)
```

#### Issue: Out of memory errors

**Cause**: Machine type too small for index size.

**Solution**:
```python
# Use high-memory machine types
endpoint.deploy_index(
    index=large_index,
    deployed_index_id="large_deployed",
    machine_type="n1-highmem-32"  # 208 GB RAM
)

# Or reduce index size
# - Lower dimensions
# - Shard large indexes
# - Remove unnecessary metadata
```

#### Issue: Streaming updates not appearing

**Cause**: Updates propagate in 10-60 seconds.

**Solution**:
```python
# Wait after upsert
index.upsert_datapoints(datapoints=[...])
time.sleep(30)  # Wait for propagation

# Verify with query
response = endpoint.find_neighbors(
    queries=[query_emb],
    num_neighbors=10
)
```

---

## Best Practices

### 1. Index Design

✅ **DO:**
- Use `DOT_PRODUCT_DISTANCE` with normalized embeddings (Vertex AI Embeddings)
- Set `approximate_neighbors_count` to expected `num_neighbors` × 10-20
- Enable `restricts` if you need filtering
- Use streaming indexes for dynamic data
- Version your indexes (v1, v2, ...) for safe updates

❌ **DON'T:**
- Mix different embedding models in same index
- Store embeddings with different dimensions
- Use batch indexes for real-time applications
- Forget to normalize embeddings for dot product

### 2. Endpoint Management

✅ **DO:**
- Use auto-scaling (min 2 replicas for HA)
- Deploy multiple index versions for A/B testing
- Monitor query latency and QPS
- Set up alerts for endpoint health
- Use public endpoints unless VPC required

❌ **DON'T:**
- Run production on single replica (no HA)
- Deploy untested indexes directly to production
- Use oversized machine types (waste money)
- Ignore latency SLO violations

### 3. Query Optimization

✅ **DO:**
- Batch multiple queries when possible
- Request only needed neighbors (don't over-fetch)
- Use filtering to reduce search space
- Cache frequent queries
- Monitor query patterns

❌ **DON'T:**
- Query one-by-one in loops (use batch queries)
- Request 100+ neighbors if you only need 10
- Ignore slow query patterns
- Fetch embeddings on every query (cache them)

### 4. Data Quality

✅ **DO:**
- Use high-quality embedding models
- Normalize embeddings consistently
- Clean text before embedding (remove HTML, etc.)
- Use task-specific embeddings (retrieval vs classification)
- Validate embeddings before indexing

❌ **DON'T:**
- Mix embeddings from different models
- Use generic embeddings for specialized domains
- Skip data quality checks
- Forget to update embeddings when data changes

### 5. Monitoring

✅ **DO:**
- Monitor query latency (p50, p95, p99)
- Track recall metrics
- Set up alerts for errors
- Log failed queries
- Monitor index update lag (streaming)

❌ **DON'T:**
- Deploy without monitoring
- Ignore latency spikes
- Skip recall evaluation
- Assume updates are instant

---

## Anti-Patterns

### ❌ Anti-Pattern 1: Using Vector Search for Exact Matching

**Bad**:
```python
# Searching for exact product SKU
query = "SKU-12345"
query_emb = model.get_embeddings([query])[0].values
results = endpoint.find_neighbors(queries=[query_emb], num_neighbors=10)
```

**Why**: Vector search finds *similar* items, not exact matches. SKU won't match semantically.

**Good**:
```python
# Use database for exact matching
exact_match = db.query("SELECT * FROM products WHERE sku = 'SKU-12345'")

# Or use hybrid search (keyword component will match exact SKU)
```

### ❌ Anti-Pattern 2: One Query Per Item in Loop

**Bad**:
```python
for item in items:
    query_emb = model.get_embeddings([item])[0].values
    results = endpoint.find_neighbors(queries=[query_emb], num_neighbors=5)
    process(results)
```

**Why**: 1000 items = 1000 API calls = high latency + cost.

**Good**:
```python
# Batch process
embeddings = [model.get_embeddings([item])[0].values for item in items]
results = endpoint.find_neighbors(queries=embeddings, num_neighbors=5)
```

### ❌ Anti-Pattern 3: Ignoring Embedding Quality

**Bad**:
```python
# Using same embedding for search and classification
query_emb = model.get_embeddings([query])[0].values  # Generic
```

**Good**:
```python
# Use task-specific embeddings
from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# For retrieval
query_emb = model.get_embeddings(
    [query],
    task_type="RETRIEVAL_QUERY"  # Optimized for search
)[0].values

# Index embeddings with RETRIEVAL_DOCUMENT
doc_embs = model.get_embeddings(
    documents,
    task_type="RETRIEVAL_DOCUMENT"
)
```

### ❌ Anti-Pattern 4: Storing Everything in Vector Search

**Bad**:
```python
# Storing full product metadata in Vector Search
{
    "id": "123",
    "embedding": [...],
    "name": "Product Name",
    "description": "Long description...",  # ❌ Inefficient
    "price": 99.99,
    "inventory": 50,
    "images": [...]  # ❌ Don't store images
}
```

**Good**:
```python
# Store only IDs and embeddings
{
    "id": "123",
    "embedding": [...],
    "restricts": [{"namespace": "category", "allow": ["electronics"]}]
}

# Fetch metadata from database after search
results = endpoint.find_neighbors(queries=[query_emb], num_neighbors=10)
product_ids = [n.id for n in results[0]]
products = database.get_products(product_ids)  # Fetch full details
```

---

## Real-World Examples

### Example 1: E-Commerce Product Search

```python
class ProductSearchEngine:
    """Hybrid search for e-commerce products."""

    def __init__(self, project_id, location):
        aiplatform.init(project=project_id, location=location)
        self.endpoint = self._get_endpoint()
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        self.sparse_vectorizer = self._load_vectorizer()

    def index_products(self, products_df):
        """Index product catalog."""
        # Generate embeddings
        texts = (
            products_df['name'] + " " +
            products_df['description'] + " " +
            products_df['category']
        )

        dense_embs = []
        for batch in self._batch(texts, 250):
            embs = self.embedding_model.get_embeddings(
                list(batch),
                task_type="RETRIEVAL_DOCUMENT"
            )
            dense_embs.extend([e.values for e in embs])

        # Generate sparse embeddings
        sparse_embs = [
            self._get_sparse_embedding(text)
            for text in texts
        ]

        # Create hybrid embeddings
        embeddings = []
        for i, row in products_df.iterrows():
            embeddings.append({
                "id": str(row['product_id']),
                "embedding": dense_embs[i],
                "sparse_embedding": sparse_embs[i],
                "restricts": [{
                    "namespace": "category",
                    "allow": [row['category']]
                }]
            })

        # Upload to GCS and create index
        self._upload_and_index(embeddings)

    def search(self, query, category=None, top_k=20):
        """Search products with optional category filter."""
        # Generate query embeddings
        dense_emb = self.embedding_model.get_embeddings(
            [query],
            task_type="RETRIEVAL_QUERY"
        )[0].values

        sparse_emb = self._get_sparse_embedding(query)

        # Create hybrid query
        hybrid_query = HybridQuery(
            dense_embedding=dense_emb,
            sparse_embedding_values=sparse_emb["values"],
            sparse_embedding_dimensions=sparse_emb["dimensions"],
            rrf_ranking_alpha=0.6  # Slightly favor semantics
        )

        # Build restricts
        restricts = []
        if category:
            restricts = [{
                "namespace": "category",
                "allow_list": [category]
            }]

        # Search
        response = self.endpoint.find_neighbors(
            deployed_index_id="products_v1",
            queries=[hybrid_query],
            num_neighbors=top_k,
            restricts=restricts if restricts else None
        )

        # Fetch product details
        product_ids = [n.id for n in response[0]]
        products = self._fetch_products(product_ids)

        return products

# Usage
engine = ProductSearchEngine(PROJECT_ID, LOCATION)
results = engine.search("wireless headphones", category="electronics", top_k=10)
```

### Example 2: Semantic Document Search (RAG)

```python
class DocumentRAG:
    """RAG system for internal documentation."""

    def __init__(self, project_id, location):
        self.endpoint = self._get_endpoint()
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        self.llm = TextGenerationModel.from_pretrained("text-bison@002")

    def ingest_documents(self, docs_path):
        """Chunk and index documents."""
        chunks = []

        # Load and chunk documents
        for doc_file in glob.glob(f"{docs_path}/*.txt"):
            with open(doc_file) as f:
                text = f.read()

            # Chunk into paragraphs (512 tokens each)
            doc_chunks = self._chunk_text(text, chunk_size=512)

            for i, chunk in enumerate(doc_chunks):
                chunks.append({
                    "id": f"{doc_file}_{i}",
                    "text": chunk,
                    "source": doc_file
                })

        # Generate embeddings
        embeddings = []
        for batch in self._batch(chunks, 250):
            texts = [c["text"] for c in batch]
            embs = self.embedding_model.get_embeddings(
                texts,
                task_type="RETRIEVAL_DOCUMENT"
            )

            for chunk, emb in zip(batch, embs):
                embeddings.append({
                    "id": chunk["id"],
                    "embedding": emb.values,
                })

        self._upload_and_index(embeddings)

    def query(self, question):
        """Answer question using RAG."""
        # 1. Retrieve relevant chunks
        query_emb = self.embedding_model.get_embeddings(
            [question],
            task_type="RETRIEVAL_QUERY"
        )[0].values

        response = self.endpoint.find_neighbors(
            deployed_index_id="docs_v1",
            queries=[query_emb],
            num_neighbors=5
        )

        # 2. Fetch chunk texts
        chunk_ids = [n.id for n in response[0]]
        chunks = self._fetch_chunks(chunk_ids)
        context = "\n\n".join([c["text"] for c in chunks])

        # 3. Generate answer
        prompt = f"""Answer the question using only the provided context. If the answer is not in the context, say "I don't know."

Question: {question}

Context:
{context}

Answer:"""

        answer = self.llm.predict(
            prompt,
            temperature=0.2,
            max_output_tokens=256
        ).text

        return {
            "answer": answer,
            "sources": [c["source"] for c in chunks]
        }

# Usage
rag = DocumentRAG(PROJECT_ID, LOCATION)
result = rag.query("How do I deploy to Cloud Run?")
print(result["answer"])
print("Sources:", result["sources"])
```

### Example 3: Anomaly Detection

```python
class AnomalyDetector:
    """Detect anomalous patterns in logs using embeddings."""

    def __init__(self, endpoint, deployed_index_id):
        self.endpoint = endpoint
        self.deployed_index_id = deployed_index_id
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    def index_normal_logs(self, normal_logs):
        """Index known normal log patterns."""
        embeddings = []

        for i, log in enumerate(normal_logs):
            emb = self.embedding_model.get_embeddings([log])[0].values
            embeddings.append({
                "id": f"normal_{i}",
                "embedding": emb
            })

        self._upload_and_index(embeddings)

    def detect_anomaly(self, log_entry, threshold=0.7):
        """
        Detect if log entry is anomalous.
        Returns True if no similar normal logs found.
        """
        # Get embedding
        log_emb = self.embedding_model.get_embeddings([log_entry])[0].values

        # Find most similar normal logs
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[log_emb],
            num_neighbors=5
        )

        # Check similarity to normal patterns
        if not response[0]:
            return True, 0.0  # No matches = anomaly

        max_similarity = response[0][0].distance

        is_anomaly = max_similarity < threshold
        return is_anomaly, max_similarity

# Usage
detector = AnomalyDetector(endpoint, "normal_logs_v1")

# Index normal logs
normal_logs = [
    "INFO: User logged in successfully",
    "INFO: Request processed in 45ms",
    "DEBUG: Cache hit for key user_123",
]
detector.index_normal_logs(normal_logs)

# Detect anomaly
new_log = "ERROR: Database connection timeout after 30s"
is_anomaly, similarity = detector.detect_anomaly(new_log, threshold=0.7)

if is_anomaly:
    print(f"🚨 ANOMALY DETECTED! Similarity: {similarity}")
else:
    print(f"✅ Normal log. Similarity: {similarity}")
```

---

## Summary

Vertex AI Vector Search is Google Cloud's production-grade vector database for building semantic search, RAG, recommendations, and more. Key takeaways:

### When to Use
- ✅ Semantic search at scale (millions to billions of vectors)
- ✅ RAG applications requiring low-latency retrieval
- ✅ Hybrid search (semantic + keyword)
- ✅ Recommendation systems
- ❌ Exact keyword matching (use Vertex AI Search)
- ❌ Small datasets (< 1000 items)

### Key Features
- **ScaNN Algorithm**: Industry-leading ANN performance
- **Hybrid Search**: Combine dense and sparse embeddings
- **Streaming Updates**: Real-time index updates
- **Auto-Scaling**: Handle variable traffic
- **High Availability**: 99.5% SLA

### Costs
- **Storage**: $0.40/GB/month
- **Compute**: Machine type pricing (e.g., $0.76/hour for n1-standard-16)
- **Optimization**: Use auto-scaling, right-sized machines

### Performance
- **Latency**: 10-50ms for billions of vectors
- **Throughput**: 1000+ QPS per replica
- **Recall**: 85-97% depending on configuration

### Next Steps
1. Try the [Quickstart](#quickstart-10-minutes)
2. Experiment with [Hybrid Search](#hybrid-search-semantic--keyword)
3. Read the [Comparison Guide](./guide_rag_search_comparison.md)
4. Explore [RAG Engine](./guide_rag_engine.md) for higher-level abstraction

---

## Additional Resources

### Documentation
- [Vertex AI Vector Search Overview](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
- [Create and Manage Indexes](https://cloud.google.com/vertex-ai/docs/vector-search/create-manage-index)
- [Query Indexes](https://cloud.google.com/vertex-ai/docs/vector-search/query-index-public-endpoint)
- [Input Data Format](https://cloud.google.com/vertex-ai/docs/vector-search/setup/format-structure)

### Tutorials
- [Vector Search Quickstart Notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/vector-search-quickstart.ipynb)
- [Hybrid Search Tutorial](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/hybrid-search.ipynb)
- [Text Embeddings + Vector Search](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/intro-textemb-vectorsearch.ipynb)

### Research Papers
- [ScaNN: Efficient Vector Similarity Search](https://arxiv.org/abs/1908.10396)
- [Announcing ScaNN (Google AI Blog)](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html)

### Community
- [Stack Overflow: google-cloud-vertex-ai tag](https://stackoverflow.com/questions/tagged/google-cloud-vertex-ai)
- [Google Cloud Community](https://www.googlecloudcommunity.com/)

---

**End of Manual** | [View Comparison Guide →](./guide_rag_search_comparison.md)

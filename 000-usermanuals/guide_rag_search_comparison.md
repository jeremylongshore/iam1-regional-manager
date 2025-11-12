# RAG and Search Technologies Comparison Guide for Google Cloud

**Version:** 1.0
**Last Updated:** 2025-11-12
**Target Audience:** AI Agents learning RAG and search technologies on Google Cloud

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Technology Overview](#technology-overview)
3. [Decision Matrix](#decision-matrix)
4. [Feature Comparison](#feature-comparison)
5. [Cost Comparison](#cost-comparison)
6. [Performance Comparison](#performance-comparison)
7. [Use Case Recommendations](#use-case-recommendations)
8. [Migration Paths](#migration-paths)
9. [Architecture Decision Flowchart](#architecture-decision-flowchart)
10. [Combination Patterns](#combination-patterns)
11. [FAQ](#faq)
12. [Decision Checklist](#decision-checklist)

---

## Executive Summary

Google Cloud offers **four main technologies** for RAG and search applications. This guide helps you choose the right one.

### The Four Technologies

| Technology | Type | Best For | Complexity |
|-----------|------|----------|-----------|
| **Vertex AI Vector Search** | Vector database | Pure semantic search, custom RAG, low latency | Medium |
| **Vertex AI Search** | Enterprise search | Hybrid search, document search, out-of-box quality | Low |
| **RAG Engine** | RAG framework | Quick RAG prototyping, Gemini apps | Low |
| **Custom RAG** | DIY framework | Maximum control, specific requirements | High |

### Quick Decision Guide

```
Need search quickly? → Vertex AI Search
Need RAG quickly? → RAG Engine
Need custom control? → Custom RAG with Vector Search
Need lowest latency? → Vector Search
Need keyword + semantic? → Vertex AI Search OR Hybrid Vector Search
```

---

## Technology Overview

### Vertex AI Vector Search

**What:** Scalable vector database for similarity search

```python
# Pure vector similarity
query_embedding = model.get_embeddings(["query"])[0].values
results = endpoint.find_neighbors(
    queries=[query_embedding],
    num_neighbors=10
)
```

**Key Characteristics:**
- Pure vector similarity search
- ScaNN algorithm (Google's ANN)
- Millisecond latency
- Billions of vectors
- Manual integration with LLMs

**Pricing:** Endpoint-based ($0.76/hour for n1-standard-16)

---

### Vertex AI Search

**What:** Complete enterprise search engine

```python
# Automatic hybrid search + LLM summaries
response = search_client.search(
    serving_config=serving_config,
    query="Who is the CEO?",  # Natural language
    content_search_spec=summary_spec  # Auto-generates summary
)
print(response.summary.summary_text)  # LLM-generated answer
```

**Key Characteristics:**
- Keyword + Semantic hybrid search
- Document parsing (PDF, HTML, etc.)
- LLM summaries and answers
- Filters, facets, ranking
- Out-of-box search quality

**Pricing:** Query-based (~$45/1000 queries for Enterprise tier)

---

### RAG Engine

**What:** Managed RAG framework

```python
# Automatic chunking, embedding, indexing, retrieval
corpus = rag.create_corpus(display_name="docs")
rag.upload_file(corpus_name=corpus.name, path="doc.pdf")

# LLM generation with RAG (all automatic)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="question",
    config=GenerateContentConfig(tools=[rag_tool])
)
```

**Key Characteristics:**
- High-level RAG abstraction
- Automatic document processing
- Multiple vector DB backends
- Native Gemini integration
- Managed infrastructure

**Pricing:** Vector DB costs + embedding costs + LLM costs

---

### Custom RAG

**What:** DIY RAG pipeline

```python
# Build everything yourself
chunks = chunk_document(doc)
embeddings = generate_embeddings(chunks)
index_vectors(embeddings, vector_db)

contexts = retrieve_contexts(query, vector_db)
prompt = build_prompt(query, contexts)
answer = llm.generate(prompt)
```

**Key Characteristics:**
- Full control over every step
- Use any LLM, any vector DB
- Custom chunking, retrieval logic
- Maximum flexibility
- High maintenance

**Pricing:** Component costs (you optimize)

---

## Decision Matrix

### Scenario-Based Decisions

| Scenario | Best Choice | Why |
|----------|------------|-----|
| **E-commerce product search** | Vertex AI Search | Hybrid search, filters, facets, out-of-box quality |
| **Customer support chatbot** | Vertex AI Search + RAG Engine | Search for docs, RAG for answers |
| **Internal knowledge base** | Vertex AI Search OR RAG Engine | Both good; Search for UX, RAG Engine for rapid dev |
| **Recommendation system** | Vector Search | Pure semantic similarity, lowest latency |
| **Semantic code search** | Vector Search + Custom RAG | Custom code chunking, precise retrieval |
| **Legal document search** | Vertex AI Search | Keyword precision, document parsing |
| **Multi-language QA** | RAG Engine | Quick setup, multi-language embeddings |
| **Real-time similar item** | Vector Search | Sub-10ms latency critical |
| **Research paper search** | Vertex AI Search | Hybrid search, citations, summaries |

### Feature Requirements

| Need | Vector Search | Vertex AI Search | RAG Engine | Custom RAG |
|------|--------------|------------------|-----------|------------|
| **Semantic search only** | ✅ Best | ⚠️ Overkill | ✅ Good | ✅ Good |
| **Keyword search** | ❌ No (unless hybrid) | ✅ Built-in | ❌ No | ✅ Can add |
| **Hybrid (semantic + keyword)** | ✅ Manual | ✅ Automatic | ❌ No | ✅ Can add |
| **Document parsing** | ❌ No | ✅ Auto | ✅ Auto | ❌ Build yourself |
| **LLM summaries** | ❌ No | ✅ Built-in | ✅ With Gemini | ✅ Build yourself |
| **Citations** | ❌ No | ✅ Built-in | ✅ With Gemini | ✅ Build yourself |
| **Filters/facets** | ✅ Restricts | ✅ Rich filters | ⚠️ Limited | ✅ Build yourself |
| **Sub-10ms latency** | ✅ Yes | ❌ No (50-200ms) | ❌ No | ✅ Possible |
| **Billions of vectors** | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Depends on DB |
| **Custom embeddings** | ✅ Full control | ⚠️ Limited | ⚠️ Limited | ✅ Full control |

### Operational Requirements

| Need | Vector Search | Vertex AI Search | RAG Engine | Custom RAG |
|------|--------------|------------------|-----------|------------|
| **Quick setup (< 1 hour)** | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Minimal code** | ⚠️ Medium | ✅ Minimal | ✅ Minimal | ❌ Lots |
| **No ops overhead** | ⚠️ Some ops | ✅ Fully managed | ✅ Fully managed | ❌ High ops |
| **Auto-scaling** | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Build yourself |
| **Monitoring built-in** | ✅ Basic | ✅ Rich | ⚠️ Limited | ❌ Build yourself |
| **Version control** | ✅ Indexes | ✅ Apps | ✅ Corpora | ✅ Git |

---

## Feature Comparison

### Complete Feature Matrix

| Feature | Vector Search | Vertex AI Search | RAG Engine | Custom RAG |
|---------|--------------|------------------|-----------|------------|
| **RETRIEVAL** ||||
| Semantic search | ✅ Core | ✅ Built-in | ✅ Via vector DB | ✅ Build |
| Keyword search | ❌ No* | ✅ Built-in | ❌ No | ✅ Build |
| Hybrid search | ✅ Manual | ✅ Auto | ❌ No | ✅ Build |
| Query expansion | ❌ No | ✅ Auto | ❌ No | ✅ Build |
| Spell correction | ❌ No | ✅ Auto | ❌ No | ✅ Build |
| **INDEXING** ||||
| Batch indexing | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Streaming updates | ✅ Yes | ❌ No | ✅ Yes** | ✅ Depends |
| Auto document parsing | ❌ No | ✅ Yes | ✅ Yes | ❌ Build |
| Custom chunking | ✅ Full control | ⚠️ Limited | ⚠️ Limited | ✅ Full control |
| Custom embeddings | ✅ Any model | ⚠️ Google only | ⚠️ Google primary | ✅ Any model |
| **SEARCH FEATURES** ||||
| Filtering | ✅ Restricts | ✅ Rich | ⚠️ Basic | ✅ Build |
| Faceting | ❌ No | ✅ Yes | ❌ No | ✅ Build |
| Ranking/boosting | ⚠️ Manual | ✅ ML ranking | ❌ No | ✅ Build |
| Re-ranking | ❌ No | ✅ Yes | ❌ No | ✅ Build |
| **LLM INTEGRATION** ||||
| Answer extraction | ❌ No | ✅ Built-in | ✅ Gemini | ✅ Build |
| Summarization | ❌ No | ✅ Built-in | ✅ Gemini | ✅ Build |
| Citations | ❌ No | ✅ Built-in | ✅ Gemini | ✅ Build |
| Multi-turn chat | ❌ No | ❌ No | ✅ Gemini | ✅ Build |
| **DATA SOURCES** ||||
| Local files | ✅ Via GCS | ✅ Via GCS | ✅ Direct | ✅ Any |
| Cloud Storage | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| BigQuery | ❌ No | ✅ Yes | ❌ No | ✅ Build |
| Google Drive | ❌ No | ❌ No | ✅ Yes | ✅ Build |
| Websites (crawler) | ❌ No | ✅ Yes | ❌ No | ✅ Build |
| **PERFORMANCE** ||||
| Query latency | 10-50ms | 50-200ms | 50-200ms | Varies |
| Max scale | Billions | Billions | Billions | Varies |
| Throughput | 1000+ QPS | 1000+ QPS | Varies | Varies |
| **COST** ||||
| Pricing model | Endpoint-based | Query-based | Component-based | Component-based |
| Relative cost | $$$ | $$$$ | $$$ | $$ |

*Hybrid search available with sparse embeddings
**Depends on backend vector DB

---

## Cost Comparison

### Pricing Models

#### Vector Search
```
Storage: $0.40/GB/month
Endpoint: $0.76/hour (n1-standard-16) × replicas × 730 hours
Queries: Included

Example (1M vectors, 2 replicas):
- Storage: 3GB × $0.40 = $1.20/month
- Endpoints: 2 × $0.76 × 730 = $1,109/month
Total: ~$1,110/month
```

#### Vertex AI Search
```
Storage: $0.024/GB/month
Queries: $45/1000 queries (Enterprise tier)

Example (100K queries/month, 10GB docs):
- Storage: 10GB × $0.024 = $0.24/month
- Queries: 100 × $45 = $4,500/month
Total: ~$4,500/month
```

#### RAG Engine
```
Vector DB cost (e.g., Vector Search endpoint)
+ Embedding generation ($0.025/1000 calls)
+ LLM generation ($0.50/1M tokens)

Example (1K docs, 10K queries/month):
- Vector Search: $550/month (1 replica)
- Embeddings: 10K × $0.025/1000 = $0.25/month
- LLM: 10K queries × 500 tokens × $0.50/1M = $2.50/month
Total: ~$553/month
```

#### Custom RAG
```
Your chosen vector DB
+ Your chosen LLM
+ Infrastructure (if self-hosted)

Example (self-hosted Weaviate):
- VM (n1-standard-8): $240/month
- Storage: $20/month
- LLM API (GPT-4): $300/month
Total: ~$560/month
```

### Cost Comparison Table

| Scenario | Vector Search | Vertex AI Search | RAG Engine | Custom RAG |
|----------|--------------|------------------|-----------|------------|
| **Small (1K docs, 1K queries/month)** | $550 | $50 | $70 | $100 |
| **Medium (10K docs, 10K queries/month)** | $1,110 | $500 | $553 | $250 |
| **Large (100K docs, 100K queries/month)** | $2,220 | $4,500 | $1,100 | $500 |
| **Very Large (1M docs, 1M queries/month)** | $5,550 | $45,000 | $3,000 | $1,500 |

### Cost Optimization Strategies

#### For Vector Search
```python
# Use auto-scaling
endpoint.deploy_index(
    index=my_index,
    min_replica_count=1,  # Scale to 1 during off-hours
    max_replica_count=10
)

# Use smaller machine types for dev
machine_type="n1-standard-4"  # $0.19/hour vs $0.76/hour
```

#### For Vertex AI Search
```python
# Use Basic tier for simple search ($3/1000 queries)
search_tier=discoveryengine.SearchTier.SEARCH_TIER_BASIC

# Cache frequent queries
query_cache = {}
if query in query_cache:
    return query_cache[query]

# Disable features you don't need
# No summary_spec, no extractive_answers
```

#### For RAG Engine
```python
# Use efficient vector DB backend
# Pinecone serverless cheaper than Vector Search for low volume

# Batch embedding generation
embeddings = model.get_embeddings(batch_of_100_texts)

# Use smaller chunk sizes (fewer embeddings)
chunk_size=256  # vs 512
```

#### For Custom RAG
```python
# Use open-source everything
# Weaviate (self-hosted) + open LLM (Llama 2)

# Optimize infrastructure
# Spot instances, auto-shutdown

# Implement aggressive caching
```

---

## Performance Comparison

### Latency Comparison

| Operation | Vector Search | Vertex AI Search | RAG Engine | Custom RAG |
|-----------|--------------|------------------|-----------|------------|
| **Query (p50)** | 15ms | 80ms | 100ms | Varies |
| **Query (p99)** | 35ms | 200ms | 300ms | Varies |
| **Index build (1M docs)** | 30min | 1-2 hours | 30min* | Varies |
| **Streaming update** | 10-30s | ❌ N/A | 10-30s* | Varies |

*Depends on backend vector DB

### Throughput Comparison

| Metric | Vector Search | Vertex AI Search | RAG Engine | Custom RAG |
|--------|--------------|------------------|-----------|------------|
| **QPS per replica** | 1000+ | 100-500 | Varies* | Varies |
| **Max concurrent** | 10,000+ | 5,000+ | Varies* | Varies |
| **Batch size** | 100 queries | 1 query | 1 query | Varies |

*Depends on backend vector DB and LLM

### Recall/Precision

| Metric | Vector Search | Vertex AI Search | RAG Engine | Custom RAG |
|--------|--------------|------------------|-----------|------------|
| **Recall@10** | 85-97%* | 90-95%** | 85-97%*** | Varies |
| **Precision@10** | N/A (semantic) | 85-95% | N/A (semantic) | Varies |
| **Hybrid recall** | 90-98% | 95-98% | N/A | Varies |

*Depends on index configuration
**Automatic hybrid search
***Depends on vector DB backend

---

## Use Case Recommendations

### E-Commerce

**Recommendation:** Vertex AI Search

**Why:**
- Hybrid search (product codes + semantic)
- Filters (price, category, availability)
- Facets (category, brand, price ranges)
- Out-of-box search quality

```python
# Perfect for e-commerce
response = search_client.search(
    query="wireless headphones under $100",
    filter='price < 100 AND in_stock = true',
    facet_specs=[category_facet, price_facet]
)
```

**Alternative:** Custom RAG with Vector Search (if need custom ranking)

---

### Customer Support Chatbot

**Recommendation:** Vertex AI Search + RAG Engine

**Why:**
- Vertex AI Search: Find relevant articles (hybrid search)
- RAG Engine: Generate conversational answers

```python
# 1. Search for articles
articles = vertex_ai_search.search("how to reset password")

# 2. Generate answer with RAG
response = rag_engine.generate_answer(
    question="how to reset password",
    contexts=articles
)
```

**Alternative:** RAG Engine only (simpler but less search quality)

---

### Internal Knowledge Base

**Recommendation:** Vertex AI Search OR RAG Engine

**Vertex AI Search if:**
- Need UX features (snippets, facets, filters)
- Users search with keywords
- Large document corpus (1M+ docs)

**RAG Engine if:**
- Primarily chatbot interface
- Using Gemini models
- Rapid development priority

```python
# Vertex AI Search: Better for search UI
search_results = search.query("company policy on remote work")

# RAG Engine: Better for chat interface
answer = rag.chat("what's our remote work policy?")
```

---

### Recommendation System

**Recommendation:** Vector Search

**Why:**
- Pure semantic similarity
- Lowest latency (10-50ms)
- Scales to millions of users/items
- Custom Two-Tower model embeddings

```python
# User embedding × Item embeddings
user_embedding = user_model.encode(user_features)
recommendations = vector_search.find_neighbors(
    query=user_embedding,
    num_neighbors=20
)
```

**Not recommended:** Vertex AI Search (overkill), RAG Engine (not designed for recs)

---

### Legal Document Search

**Recommendation:** Vertex AI Search

**Why:**
- Keyword precision critical (case numbers, statutes)
- Document parsing (PDF contracts)
- Citations and sources
- Enterprise compliance features

```python
# Exact case law + semantic precedents
response = search(
    query="Smith v. Jones 2020",
    filter='document_type: "case_law"',
    boost_spec=recent_cases_boost
)
```

**Not recommended:** Pure Vector Search (misses exact terms)

---

### Academic Research Papers

**Recommendation:** Vertex AI Search

**Why:**
- Hybrid search (titles, authors, DOIs + semantic)
- Summarization for abstracts
- Citations built-in
- Filter by year, journal, etc.

```python
response = search(
    query="transformer neural networks attention",
    filter='year >= 2017',
    summary_spec=enable_summary
)
```

---

### Real-Time Anomaly Detection

**Recommendation:** Vector Search

**Why:**
- Sub-10ms latency critical
- Pure similarity search
- Streaming updates for new normal patterns

```python
# Detect if log is anomalous (low similarity to normal logs)
similarity = vector_search.find_neighbors(
    query=log_embedding,
    num_neighbors=1
)[0].distance

is_anomaly = similarity < threshold
```

**Not recommended:** Vertex AI Search (too slow), RAG Engine (not designed for this)

---

## Migration Paths

### From Custom RAG to RAG Engine

**When:**
- Reduce maintenance overhead
- Standardize on Gemini
- Want managed infrastructure

**Migration Steps:**

```python
# 1. Export embeddings from current vector DB
embeddings = current_vector_db.export_all()

# 2. Create RAG corpus
corpus = rag.create_corpus(display_name="migrated-corpus")

# 3. Upload documents (RAG Engine re-embeds)
for doc in documents:
    rag.upload_file(corpus_name=corpus.name, path=doc)

# 4. Update application code
# Before:
contexts = custom_retrieve(query)
answer = custom_llm.generate(query, contexts)

# After:
answer = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=query,
    config=GenerateContentConfig(tools=[rag_tool])
)
```

**Considerations:**
- Re-embedding costs
- Slight quality differences (embeddings may differ)
- Test thoroughly before full migration

---

### From Vector Search to Vertex AI Search

**When:**
- Need keyword search
- Want out-of-box search quality
- Need document parsing

**Migration Steps:**

```python
# 1. Create Vertex AI Search datastore
datastore = create_datastore(display_name="migrated-search")

# 2. Upload documents (not embeddings)
import_documents(
    datastore_id=datastore_id,
    paths=["gs://bucket/docs/*"]  # Original documents
)

# 3. Update application code
# Before:
query_emb = embed(query)
results = vector_search.find_neighbors(query_emb)

# After:
results = vertex_ai_search.search(query)  # Automatic hybrid
```

**Considerations:**
- Cost model change (endpoint → query-based)
- Latency increase (15ms → 80ms)
- Gain keyword search + LLM features

---

### From Vertex AI Search to Custom RAG

**When:**
- Need custom LLMs (non-Gemini)
- Need advanced retrieval logic
- Need cost optimization

**Migration Steps:**

```python
# 1. Export documents from Vertex AI Search
# (Download from GCS source)

# 2. Build custom chunking
chunks = custom_chunker(documents)

# 3. Generate embeddings
embeddings = custom_embedder(chunks)

# 4. Index in vector DB
vector_db.index(embeddings)

# 5. Build custom RAG
def custom_rag(query):
    contexts = vector_db.retrieve(query)
    answer = custom_llm.generate(query, contexts)
    return answer
```

**Considerations:**
- Lose out-of-box features
- Increase maintenance
- Gain full control

---

## Architecture Decision Flowchart

```
START: Need search or RAG?
│
├─ Need SEARCH (findability)? ──→ YES ──┐
│                                        │
├─ Need RAG (chatbot answers)? ──→ YES ──┤
│                                        │
└─ Need BOTH? ──────────────────→ YES ──┘
                                         │
                                         ↓
                       ┌─────────────────────────────────┐
                       │  What's your primary need?      │
                       └─────────────────────────────────┘
                                         │
                  ┌──────────────────────┼──────────────────────┐
                  │                      │                      │
                  ↓                      ↓                      ↓
           ┌────────────┐       ┌──────────────┐      ┌──────────────┐
           │   SEARCH   │       │     RAG      │      │     BOTH     │
           │  QUALITY   │       │    SPEED     │      │   FEATURES   │
           └────────────┘       └──────────────┘      └──────────────┘
                  │                      │                      │
                  ↓                      ↓                      ↓
        ┌──────────────────┐   ┌──────────────────┐  ┌──────────────────┐
        │ Vertex AI Search │   │   RAG Engine     │  │ Vertex AI Search │
        │                  │   │        OR        │  │       +          │
        │ - Hybrid search  │   │ Custom RAG with  │  │   RAG Engine     │
        │ - Facets/filters │   │  Vector Search   │  │                  │
        │ - Doc parsing    │   │                  │  │ - Search UX      │
        │ - Out-of-box     │   │ - Rapid dev      │  │ - Chat answers   │
        └──────────────────┘   │ - Full control   │  └──────────────────┘
                               └──────────────────┘

Need PURE SEMANTIC SIMILARITY? (not search or RAG)
│
├─ Need sub-10ms latency? ──→ YES ──→ Vector Search
│
├─ Building recommendation? ──→ YES ──→ Vector Search
│
└─ Need billions of vectors? ──→ YES ──→ Vector Search

Need CUSTOM REQUIREMENTS?
│
├─ Custom chunking logic? ──→ YES ──→ Custom RAG
│
├─ Non-Gemini LLM? ──→ YES ──→ Custom RAG
│
├─ Existing vector DB? ──→ YES ──→ Custom RAG
│
└─ Maximum control? ──→ YES ──→ Custom RAG
```

---

## Combination Patterns

### Pattern 1: Vertex AI Search + RAG Engine

**Use Case:** Best of both worlds - search UX + chat interface

```python
class HybridSearchChatbot:
    def __init__(self):
        self.search = VertexAISearch(...)
        self.rag = RAGEngine(...)

    def search_mode(self, query):
        """Traditional search results"""
        return self.search.search(query)

    def chat_mode(self, question):
        """Conversational answer"""
        return self.rag.generate_answer(question)

    def hybrid(self, query):
        """Search results + chat answer"""
        search_results = self.search.search(query)
        answer = self.rag.generate_answer(query)
        return {
            "answer": answer,
            "sources": search_results
        }
```

**Benefits:**
- Users can search OR chat
- Search provides sources for RAG
- Best UX for both modes

---

### Pattern 2: Vector Search + Custom RAG

**Use Case:** Maximum control with best vector DB

```python
class CustomRAGWithVectorSearch:
    def __init__(self):
        self.vector_search = VertexAIVectorSearch(...)
        self.chunker = CustomChunker()
        self.llm = AnthropicClaude()  # Non-Gemini LLM

    def ingest(self, documents):
        # Custom chunking
        chunks = self.chunker.chunk(documents, strategy="semantic")

        # Custom embeddings
        embeddings = self.custom_embedder(chunks)

        # Store in Vector Search
        self.vector_search.index(embeddings)

    def query(self, question):
        # Retrieve with Vector Search
        contexts = self.vector_search.retrieve(question, top_k=10)

        # Custom re-ranking
        contexts = self.reranker.rerank(question, contexts)

        # Generate with custom LLM
        answer = self.llm.generate(question, contexts)

        return answer
```

**Benefits:**
- Best vector DB (Vector Search)
- Full control over pipeline
- Use any LLM

---

### Pattern 3: Multi-Stage Retrieval

**Use Case:** Combine multiple search technologies

```python
class MultiStageRetrieval:
    def __init__(self):
        self.vertex_search = VertexAISearch(...)  # Stage 1: Broad retrieval
        self.vector_search = VectorSearch(...)     # Stage 2: Semantic re-rank

    def retrieve(self, query):
        # Stage 1: Get 100 candidates from Vertex AI Search
        candidates = self.vertex_search.search(query, top_k=100)

        # Stage 2: Re-rank with Vector Search
        embeddings = [self.embed(c.text) for c in candidates]
        query_emb = self.embed(query)

        similarities = self.vector_search.compute_similarity(
            query_emb,
            embeddings
        )

        # Return top 10 after re-ranking
        return self.rank_by_similarity(candidates, similarities)[:10]
```

**Benefits:**
- Vertex AI Search: Broad recall (keyword + semantic)
- Vector Search: Precise re-ranking
- Best of both

---

## FAQ

### Q1: Which is fastest?

**A:** Vector Search (10-50ms) > Vertex AI Search (50-200ms) > RAG Engine (depends on backend)

For pure latency-critical applications (recommendations, real-time search), use Vector Search.

---

### Q2: Which is cheapest?

**A:** Depends on scale:
- **Small (<10K queries/month):** Vertex AI Search (pay-per-query)
- **Medium:** RAG Engine or Custom RAG
- **Large (>1M queries/month):** Custom RAG (optimized)

---

### Q3: Which is easiest to get started?

**A:** Vertex AI Search = RAG Engine (both < 1 hour setup)

Vector Search and Custom RAG require more setup time.

---

### Q4: Can I use multiple together?

**A:** Yes! See [Combination Patterns](#combination-patterns)

Common combinations:
- Vertex AI Search + RAG Engine (search + chat)
- Vector Search + Custom RAG (best DB + control)
- Multi-stage (multiple technologies)

---

### Q5: Which for pure semantic search?

**A:** Vector Search

It's purpose-built for vector similarity, with lowest latency and highest scalability.

---

### Q6: Which for keyword search?

**A:** Vertex AI Search

Vector Search doesn't do keyword search (unless you implement hybrid with sparse embeddings).

---

### Q7: Which for hybrid search?

**A:** Vertex AI Search (automatic) OR Vector Search with sparse embeddings (manual)

Vertex AI Search is easier; Vector Search gives more control.

---

### Q8: Which supports non-Gemini LLMs?

**A:** Vector Search, Custom RAG

RAG Engine and Vertex AI Search are optimized for Gemini.

---

### Q9: Can I migrate between them?

**A:** Yes, but with effort. See [Migration Paths](#migration-paths).

Easiest: RAG Engine → Custom RAG (export embeddings)
Hardest: Vertex AI Search → Vector Search (re-architect)

---

### Q10: Which for production at scale?

**A:** All four can scale to billions of docs.

- **Managed:** Vertex AI Search, RAG Engine
- **Maximum control:** Vector Search, Custom RAG

---

## Decision Checklist

Use this checklist to make your decision:

### Requirements

- [ ] What's your primary need?
  - [ ] Search (findability)
  - [ ] RAG (chatbot answers)
  - [ ] Recommendations
  - [ ] Other: ___________

- [ ] What type of search?
  - [ ] Semantic only
  - [ ] Keyword only
  - [ ] Hybrid (both)

- [ ] What's your scale?
  - [ ] < 10K documents
  - [ ] 10K - 100K documents
  - [ ] 100K - 1M documents
  - [ ] > 1M documents

- [ ] What's your query volume?
  - [ ] < 1K queries/month
  - [ ] 1K - 100K queries/month
  - [ ] 100K - 1M queries/month
  - [ ] > 1M queries/month

- [ ] What's your latency requirement?
  - [ ] < 10ms (real-time)
  - [ ] < 50ms (interactive)
  - [ ] < 200ms (acceptable)
  - [ ] No strict requirement

### Operational

- [ ] How quickly do you need it?
  - [ ] This week (use Vertex AI Search or RAG Engine)
  - [ ] This month (any option)
  - [ ] No rush (can build custom)

- [ ] What's your team's expertise?
  - [ ] ML/AI experts (can build custom)
  - [ ] Software engineers (RAG Engine or Vector Search)
  - [ ] No ML expertise (Vertex AI Search)

- [ ] Operational preferences?
  - [ ] Fully managed (Vertex AI Search, RAG Engine)
  - [ ] Some ops acceptable (Vector Search)
  - [ ] Self-managed preferred (Custom RAG)

### Technical

- [ ] Which LLM?
  - [ ] Gemini (RAG Engine, Vertex AI Search)
  - [ ] GPT/Claude/other (Custom RAG)
  - [ ] No LLM needed (Vector Search)

- [ ] Custom requirements?
  - [ ] Custom chunking (Custom RAG)
  - [ ] Custom embeddings (Vector Search, Custom RAG)
  - [ ] Standard is fine (Vertex AI Search, RAG Engine)

- [ ] Data sources?
  - [ ] GCS only (all options)
  - [ ] BigQuery (Vertex AI Search, Custom RAG)
  - [ ] Google Drive (RAG Engine)
  - [ ] Websites (Vertex AI Search)

### Budget

- [ ] What's your budget?
  - [ ] Optimize for low cost (Custom RAG)
  - [ ] Willing to pay for managed (Vertex AI Search, RAG Engine)
  - [ ] No budget constraints (any option)

### Decision

Based on your answers:

**If mostly "managed", "Gemini", "quick" → Vertex AI Search or RAG Engine**

**If "latency critical", "semantic only" → Vector Search**

**If "custom requirements", "any LLM", "cost optimize" → Custom RAG**

**If "hybrid search", "out-of-box quality" → Vertex AI Search**

**If "RAG with Gemini", "rapid dev" → RAG Engine**

---

## Summary

### Quick Reference

| Choose... | When... |
|-----------|---------|
| **Vertex AI Search** | Need hybrid search, out-of-box quality, document parsing |
| **Vector Search** | Need pure semantic, low latency, billions of vectors |
| **RAG Engine** | Need RAG quickly, using Gemini, want managed |
| **Custom RAG** | Need max control, custom requirements, any LLM |

### Key Takeaways

1. **No single best choice** - depends on your requirements
2. **Can combine technologies** - use multiple for different needs
3. **Start simple, evolve** - begin with managed, customize later
4. **Cost vs Control trade-off** - managed costs more, gives less control
5. **Latency hierarchy** - Vector Search < Vertex AI Search < RAG Engine

### Next Steps

1. Review your [Decision Checklist](#decision-checklist)
2. Read the detailed guide for your chosen technology:
   - [Vertex AI Vector Search Guide](./guide_vertex_ai_vector_search.md)
   - [Vertex AI Search Guide](./guide_vertex_ai_search.md)
   - [RAG Engine Guide](./guide_rag_engine.md)
3. Start with a prototype
4. Evaluate quality and cost
5. Iterate and optimize

---

**End of Comparison Guide**

**Related Manuals:**
- [Vertex AI Vector Search Guide →](./guide_vertex_ai_vector_search.md)
- [Vertex AI Search Guide →](./guide_vertex_ai_search.md)
- [RAG Engine Guide →](./guide_rag_engine.md)

# RAG Engine on Vertex AI: Comprehensive User Manual for AI Agents

**Version:** 1.0
**Last Updated:** 2025-11-12
**Target Audience:** AI Agents learning RAG and search technologies on Google Cloud

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [What is RAG Engine](#what-is-rag-engine)
3. [When to Use RAG Engine](#when-to-use-rag-engine)
4. [Core Concepts](#core-concepts)
5. [Architecture](#architecture)
6. [Quickstart (5 Minutes)](#quickstart-5-minutes)
7. [Vector Database Options](#vector-database-options)
8. [Complete Implementation Guide](#complete-implementation-guide)
9. [Query Patterns](#query-patterns)
10. [Integration with Gemini](#integration-with-gemini)
11. [Advanced Techniques](#advanced-techniques)
12. [Evaluation and Monitoring](#evaluation-and-monitoring)
13. [Troubleshooting](#troubleshooting)
14. [Best Practices](#best-practices)
15. [Comparison with Custom RAG](#comparison-with-custom-rag)

---

## Executive Summary

**RAG Engine** is Vertex AI's unified framework for building Retrieval-Augmented Generation (RAG) applications. It provides a **high-level, managed abstraction** over vector databases and LLMs, making RAG development simple and production-ready.

### Key Capabilities

- **Unified API**: Single interface for multiple vector databases
- **Managed Infrastructure**: No vector database ops required
- **Automatic Chunking**: Built-in document processing
- **Embedding Generation**: Integrated with Vertex AI Embeddings
- **Multi-Source**: Ingest from GCS, Drive, local files
- **Production-Ready**: Auto-scaling, monitoring, versioning

### Supported Vector Databases

| Vector Database | Type | Best For |
|----------------|------|----------|
| **Vertex AI Vector Search** | Google-managed | Production, scale, performance |
| **Vertex AI Feature Store** | Google-managed | ML features + vectors |
| **Vertex AI Search** | Google-managed | Enterprise search |
| **Pinecone** | Third-party SaaS | Multi-cloud, managed |
| **Weaviate** | Third-party OSS | Flexible, self-hosted |

### What You'll Learn

- RAG Engine concepts and workflow
- How to build RAG apps with different vector databases
- Query patterns and retrieval strategies
- Integration with Gemini for generation
- Evaluation and tuning
- When to use RAG Engine vs custom RAG

---

## What is RAG Engine

### The RAG Challenge

Building production RAG systems involves many complex pieces:

```
Traditional RAG Stack (Complex):

1. Document Chunking ← Manual implementation
2. Embedding Generation ← API calls, batching, rate limits
3. Vector Database ← Setup, scaling, operations
4. Retrieval Logic ← Similarity search, filtering
5. Prompt Engineering ← Context assembly
6. LLM Integration ← API calls, error handling
7. Response Parsing ← Extract, format, cite
8. Monitoring & Evaluation ← Custom metrics
```

### RAG Engine Solution (Simplified)

```python
# RAG Engine handles ALL of the above:

from vertexai import rag

# 1. Create corpus (handles embedding, chunking, indexing)
corpus = rag.create_corpus(display_name="my-docs")

# 2. Import files (auto-chunks, auto-embeds, auto-indexes)
rag.import_files(
    corpus_name=corpus.name,
    paths=["gs://my-bucket/docs/*"]
)

# 3. Query with LLM (auto-retrieves, auto-prompts, auto-generates)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What is RAG?",
    config=GenerateContentConfig(
        tools=[rag_retrieval_tool]  # Automatic RAG!
    )
)

print(response.text)  # Grounded answer with citations
```

### Key Abstractions

#### 1. **RAG Corpus**

A **corpus** is a collection of documents.

```python
corpus = rag.create_corpus(
    display_name="product-docs",
    backend_config=rag.RagVectorDbConfig(
        # Choose vector database
        vector_db=rag.VertexVectorSearch(...)  # or Pinecone, Weaviate, etc.
    )
)
```

**Corpus handles:**
- Document storage
- Chunking configuration
- Embedding model selection
- Vector database connection

#### 2. **RAG Files**

Individual documents within a corpus.

```python
# Upload file
rag_file = rag.upload_file(
    corpus_name=corpus.name,
    path="manual.pdf",
    display_name="Product Manual"
)

# Or import many files
rag.import_files(
    corpus_name=corpus.name,
    paths=["gs://bucket/docs/*.pdf"]
)
```

**RAG Engine handles:**
- File parsing (PDF, HTML, TXT, DOCX, etc.)
- Text extraction
- Chunking (configurable)
- Embedding generation (automatic)
- Vector indexing (automatic)

#### 3. **Retrieval Tool**

Connect corpus to LLM for generation.

```python
from google.genai.types import Tool, Retrieval, VertexRagStore

tool = Tool(
    retrieval=Retrieval(
        vertex_rag_store=VertexRagStore(
            rag_corpora=[corpus.name],
            similarity_top_k=10
        )
    )
)

# Use with any Gemini model
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="question",
    config=GenerateContentConfig(tools=[tool])
)
```

---

## When to Use RAG Engine

### ✅ Use RAG Engine When:

| Scenario | Reason |
|----------|--------|
| **Building RAG from scratch** | Avoid reinventing the wheel |
| **Want managed infrastructure** | No vector DB ops required |
| **Need quick prototyping** | Get RAG working in minutes |
| **Using Gemini models** | Native integration |
| **Want automatic chunking/embedding** | Built-in document processing |
| **Need multi-source ingestion** | GCS, Drive, local files supported |

### ❌ Use Custom RAG When:

| Scenario | Reason |
|----------|--------|
| **Need custom chunking logic** | RAG Engine chunking is configurable but not fully custom |
| **Using non-Gemini LLMs** | RAG Engine optimized for Gemini |
| **Need advanced retrieval** | E.g., hybrid search, re-ranking, filters |
| **Existing vector DB** | Already have vector DB infrastructure |
| **Need full control** | Custom pipelines, custom embeddings |

### ✅ vs ❌ Quick Decision Matrix

```
Start Here:
│
├─ Do you need RAG quickly? ──→ YES ──→ Use RAG Engine
│                             ↓ NO
│                             ↓
├─ Do you need advanced retrieval? ──→ YES ──→ Use Custom RAG
│                                    ↓ NO
│                                    ↓
├─ Are you using Gemini? ──→ YES ──→ Use RAG Engine
│                         ↓ NO
│                         ↓
└─ Are you using other LLMs? ──→ Use Custom RAG with LangChain
```

---

## Core Concepts

### 1. Data Ingestion

**Supported Sources:**

| Source | Path Format | Example |
|--------|-------------|---------|
| **Local File** | File path | `"manual.pdf"` |
| **Cloud Storage** | GCS URI | `"gs://bucket/docs/*.pdf"` |
| **Google Drive** | Drive URL | `"https://drive.google.com/file/d/FILE_ID"` |
| **Google Drive Folder** | Drive folder URL | `"https://drive.google.com/drive/folders/FOLDER_ID"` |

**File Formats:**

- PDF, HTML, TXT, MD
- DOCX, PPTX
- JSON (with schema)

### 2. Data Transformation

**Chunking:**

RAG Engine automatically chunks documents for better retrieval:

```python
transformation_config = rag.TransformationConfig(
    chunking_config=rag.ChunkingConfig(
        chunk_size=512,      # Tokens per chunk
        chunk_overlap=50     # Overlap between chunks
    )
)

rag.import_files(
    corpus_name=corpus.name,
    paths=["gs://bucket/docs/*"],
    transformation_config=transformation_config
)
```

**Chunk Size Recommendations:**

| Use Case | Chunk Size | Overlap |
|----------|-----------|---------|
| **General Q&A** | 512 tokens | 50 tokens |
| **Long-form documents** | 1024 tokens | 100 tokens |
| **Short snippets** | 256 tokens | 25 tokens |
| **Code documentation** | 768 tokens | 75 tokens |

### 3. Embedding Generation

**Automatic Embedding:**

RAG Engine uses Vertex AI Embeddings API by default:

```python
# Default: text-embedding-005 (768 dims)
corpus = rag.create_corpus(
    display_name="my-corpus",
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model="publishers/google/models/text-embedding-005"
            )
        )
    )
)
```

**Supported Models:**

| Model | Dimensions | Best For |
|-------|-----------|----------|
| `text-embedding-005` | 768 | General text (recommended) |
| `textembedding-gecko@003` | 768 | Legacy support |
| `multimodalembedding@001` | 1408 | Multimodal (text + image) |

### 4. Retrieval

**Retrieval Configuration:**

```python
retrieval_config = rag.RagRetrievalConfig(
    top_k=10,  # Return top 10 chunks
    filter=rag.Filter(
        vector_distance_threshold=0.5  # Similarity threshold
    )
)

response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
    text="What is RAG?",
    rag_retrieval_config=retrieval_config
)

# Access retrieved chunks
for context in response.contexts.contexts:
    print(context.text)
    print(context.distance)  # Similarity score
```

---

## Architecture

### RAG Engine Architecture

```
┌──────────────────────────────────────────────────────┐
│                  APPLICATION                         │
│  (Your Python Code)                                  │
└────────────────────┬─────────────────────────────────┘
                     │
                     │ RAG Engine API
                     ↓
┌──────────────────────────────────────────────────────┐
│              RAG ENGINE (Managed)                    │
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │  Document Processing Pipeline               │   │
│  │  - File Upload                              │   │
│  │  - Parsing (PDF, HTML, etc.)               │   │
│  │  - Chunking                                │   │
│  │  - Embedding Generation                    │   │
│  │  - Indexing                                │   │
│  └─────────────────────────────────────────────┘   │
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │  Retrieval Engine                           │   │
│  │  - Query Understanding                      │   │
│  │  - Vector Search                            │   │
│  │  - Filtering                                │   │
│  │  - Ranking                                  │   │
│  └─────────────────────────────────────────────┘   │
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │  LLM Integration                            │   │
│  │  - Context Assembly                         │   │
│  │  - Prompt Engineering                       │   │
│  │  - Generation                               │   │
│  │  - Citation Extraction                      │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────┬─────────────────────────────────┘
                     │
                     ↓ Stores vectors
┌──────────────────────────────────────────────────────┐
│           VECTOR DATABASE (Your Choice)              │
│  - Vertex AI Vector Search                          │
│  - Pinecone                                          │
│  - Weaviate                                          │
│  - Feature Store                                     │
│  - Vertex AI Search                                  │
└──────────────────────────────────────────────────────┘
```

### Workflow

```
1. INGEST
   Documents → RAG Engine → [Parse, Chunk, Embed, Index]
   ↓
2. STORE
   Vectors → Vector Database (your choice)
   ↓
3. QUERY
   User Question → RAG Engine → Vector Search → Top-K Chunks
   ↓
4. GENERATE
   Question + Chunks → Gemini → Grounded Answer
```

---

## Quickstart (5 Minutes)

### Prerequisites

```bash
# Install SDKs
pip install google-cloud-aiplatform google-genai

# Set up
export PROJECT_ID="your-project-id"
export LOCATION="us-central1"

# Authenticate
gcloud auth application-default login

# Enable APIs
gcloud services enable aiplatform.googleapis.com
```

### Step 1: Initialize (30 seconds)

```python
import vertexai
from vertexai import rag
from google import genai

PROJECT_ID = "your-project-id"
LOCATION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Initialize Gemini client
client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
```

### Step 2: Create Corpus (1 minute)

```python
# Create corpus with default settings (uses Vertex AI Vector Search)
corpus = rag.create_corpus(display_name="my-first-rag-corpus")

print(f"Corpus created: {corpus.name}")
```

### Step 3: Upload File (1 minute)

```python
# Create a sample file
with open("test.txt", "w") as f:
    f.write("""
    Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs by
    allowing them to access external data sources. This reduces hallucinations
    and provides up-to-date information.
    """)

# Upload to corpus
rag_file = rag.upload_file(
    corpus_name=corpus.name,
    path="test.txt",
    display_name="RAG Introduction"
)

print(f"File uploaded: {rag_file.name}")
```

### Step 4: Query with Gemini (1 minute)

```python
from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore

# Create retrieval tool
rag_tool = Tool(
    retrieval=Retrieval(
        vertex_rag_store=VertexRagStore(
            rag_corpora=[corpus.name],
            similarity_top_k=5
        )
    )
)

# Generate answer
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What is RAG and why is it helpful?",
    config=GenerateContentConfig(tools=[rag_tool])
)

print(response.text)
```

**Output:**
```
RAG (Retrieval-Augmented Generation) is a technique that enhances Large Language
Models by allowing them to access external data sources during generation. This
is helpful because it reduces hallucinations (generating false information) and
provides up-to-date information that may not be in the LLM's training data.
```

🎉 **Done!** You've built your first RAG application.

---

## Vector Database Options

RAG Engine supports multiple vector databases. Choose based on your needs:

### Option 1: Vertex AI Vector Search (Recommended)

**Best for:** Production, scale, Google Cloud native

```python
# Create corpus with Vector Search
vector_db = rag.VertexVectorSearch(
    index=index.resource_name,  # Pre-created Vector Search index
    index_endpoint=endpoint.resource_name
)

corpus = rag.create_corpus(
    display_name="vvs-corpus",
    backend_config=rag.RagVectorDbConfig(vector_db=vector_db)
)
```

**Prerequisites:**
```python
from google.cloud import aiplatform

# 1. Create Vector Search index
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-index",
    dimensions=768,
    approximate_neighbors_count=10,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    index_update_method="STREAM_UPDATE",  # Required for RAG Engine
)

# 2. Create endpoint
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="rag-endpoint",
    public_endpoint_enabled=True
)

# 3. Deploy index
endpoint.deploy_index(index=index, deployed_index_id="rag_deployed")
```

**Pros:**
- Fully managed by Google
- Best performance (ScaNN algorithm)
- Scales to billions of vectors
- Auto-scaling, high availability

**Cons:**
- Requires Vector Search setup
- Higher cost (endpoint always running)

### Option 2: Vertex AI Feature Store

**Best for:** ML features + vectors together

```python
# Create corpus with Feature Store
vector_db = rag.VertexFeatureStore(
    resource_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/featureOnlineStores/my_store"
)

corpus = rag.create_corpus(
    display_name="feature-store-corpus",
    backend_config=rag.RagVectorDbConfig(vector_db=vector_db)
)
```

**Pros:**
- Combines vectors with ML features
- Managed by Google
- Good for ML pipelines

**Cons:**
- More complex setup
- Primarily for ML use cases

### Option 3: Pinecone

**Best for:** Multi-cloud, fully managed SaaS

```python
from google.cloud import secretmanager

# 1. Store API key in Secret Manager
secret_client = secretmanager.SecretManagerServiceClient()
secret = secret_client.create_secret(
    parent=f"projects/{PROJECT_ID}",
    secret_id="pinecone-api-key",
    secret={"replication": {"automatic": {}}}
)

secret_version = secret_client.add_secret_version(
    parent=secret.name,
    payload={"data": "your-pinecone-api-key".encode()}
)

# 2. Grant RAG Engine access to secret
SERVICE_ACCOUNT = f"service-{PROJECT_NUMBER}@gcp-sa-vertex-rag.iam.gserviceaccount.com"

!gcloud secrets add-iam-policy-binding {secret.name} \
    --member="serviceAccount:{SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor"

# 3. Create corpus with Pinecone
vector_db = rag.Pinecone(
    index_name="my-pinecone-index",
    api_key=secret_version.name
)

corpus = rag.create_corpus(
    display_name="pinecone-corpus",
    backend_config=rag.RagVectorDbConfig(vector_db=vector_db)
)
```

**Pros:**
- Fully managed SaaS
- Multi-cloud (AWS, GCP, Azure)
- Simple setup
- Good documentation

**Cons:**
- Third-party dependency
- Additional cost (Pinecone pricing)
- Data egress charges

### Option 4: Weaviate

**Best for:** Open source, self-hosted, flexibility

```python
# Create corpus with Weaviate
vector_db = rag.Weaviate(
    weaviate_http_endpoint="https://my-weaviate-instance.com",
    collection_name="rag_collection",
    api_key=secret_version.name  # Optional, if using auth
)

corpus = rag.create_corpus(
    display_name="weaviate-corpus",
    backend_config=rag.RagVectorDbConfig(vector_db=vector_db)
)
```

**Pros:**
- Open source (Apache 2.0)
- Highly flexible
- Good community
- Can self-host

**Cons:**
- Need to manage infrastructure (if self-hosted)
- More operational overhead

### Option 5: Vertex AI Search

**Best for:** Enterprise search with RAG

```python
# Create corpus with Vertex AI Search
vector_db = rag.VertexAISearch(
    datastore=f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATASTORE_ID}"
)

corpus = rag.create_corpus(
    display_name="vais-corpus",
    backend_config=rag.RagVectorDbConfig(vector_db=vector_db)
)
```

**Pros:**
- Full enterprise search features
- Hybrid search (keyword + semantic)
- Document parsing included
- LLM grounding built-in

**Cons:**
- Higher cost
- More features than needed for simple RAG

### Comparison Table

| Vector DB | Managed | Cost | Setup | Performance | Best For |
|-----------|---------|------|-------|-------------|----------|
| **Vertex AI Vector Search** | ✅ Google | $$$ | Medium | ⭐⭐⭐⭐⭐ | Production scale |
| **Feature Store** | ✅ Google | $$$ | Hard | ⭐⭐⭐⭐ | ML pipelines |
| **Pinecone** | ✅ SaaS | $$$ | Easy | ⭐⭐⭐⭐ | Multi-cloud |
| **Weaviate** | ❌ Self | $ | Hard | ⭐⭐⭐ | Flexibility |
| **Vertex AI Search** | ✅ Google | $$$$ | Easy | ⭐⭐⭐ | Enterprise search |

---

## Complete Implementation Guide

### Import from Cloud Storage

```python
# Import all PDFs from GCS bucket
rag.import_files(
    corpus_name=corpus.name,
    paths=["gs://my-bucket/docs/*.pdf"],

    # Chunking configuration
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=1024,
            chunk_overlap=100
        )
    ),

    # Rate limiting
    max_embedding_requests_per_min=900
)
```

### Import from Google Drive

```python
# Import from Drive folder
DRIVE_FOLDER_ID = "your-folder-id"

# Grant access to RAG Engine service account
SERVICE_ACCOUNT = f"service-{PROJECT_NUMBER}@gcp-sa-vertex-rag.iam.gserviceaccount.com"
# (Share Drive folder with this service account as "Viewer")

# Import
rag.import_files(
    corpus_name=corpus.name,
    paths=[f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"],
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=512,
            chunk_overlap=50
        )
    )
)
```

### List and Manage Files

```python
# List files in corpus
files = rag.list_files(corpus_name=corpus.name)

for file in files:
    print(f"Name: {file.display_name}")
    print(f"  Resource: {file.name}")
    print(f"  Size: {file.size_bytes} bytes")

# Get specific file
file = rag.get_file(name="projects/.../corpora/.../ragFiles/...")

# Delete file
rag.delete_file(name=file.name)
```

### Update Corpus Configuration

```python
# Update corpus display name
updated_corpus = rag.update_corpus(
    corpus_name=corpus.name,
    display_name="Updated Corpus Name"
)

# Note: Cannot change backend_config after creation
# Must create new corpus for different vector DB
```

---

## Query Patterns

### Pattern 1: Direct Context Retrieval

Get contexts without LLM generation:

```python
# Retrieve relevant chunks
response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=corpus.name,
            # Optional: specify files
            # rag_file_ids=["file1", "file2"]
        )
    ],
    text="What is RAG?",
    rag_retrieval_config=rag.RagRetrievalConfig(
        top_k=10,
        filter=rag.Filter(
            vector_distance_threshold=0.5
        )
    )
)

# Access contexts
for ctx in response.contexts.contexts:
    print(f"Text: {ctx.text}")
    print(f"Distance: {ctx.distance}")
    print(f"Source: {ctx.source_uri}")
```

**Use Case:** When you want to build custom prompts or use non-Gemini LLMs.

### Pattern 2: Generation with Retrieval Tool

Let RAG Engine handle retrieval + generation:

```python
from google.genai.types import (
    GenerateContentConfig,
    Retrieval,
    Tool,
    VertexRagStore,
    VertexRagStoreRagResource
)

# Create retrieval tool
rag_tool = Tool(
    retrieval=Retrieval(
        vertex_rag_store=VertexRagStore(
            rag_resources=[
                VertexRagStoreRagResource(rag_corpus=corpus.name)
            ],
            similarity_top_k=10,
            vector_distance_threshold=0.4
        )
    )
)

# Generate with tool
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What is RAG?",
    config=GenerateContentConfig(tools=[rag_tool])
)

print(response.text)
```

**Use Case:** Production RAG applications with Gemini.

### Pattern 3: Multi-Corpus Query

Query multiple corpora:

```python
# Create multiple corpora
corpus_1 = rag.create_corpus(display_name="product-docs")
corpus_2 = rag.create_corpus(display_name="support-articles")

# Query both
rag_tool = Tool(
    retrieval=Retrieval(
        vertex_rag_store=VertexRagStore(
            rag_resources=[
                VertexRagStoreRagResource(rag_corpus=corpus_1.name),
                VertexRagStoreRagResource(rag_corpus=corpus_2.name),
            ],
            similarity_top_k=10
        )
    )
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="How do I install the product?",
    config=GenerateContentConfig(tools=[rag_tool])
)
```

**Use Case:** When you have different content types or sources.

### Pattern 4: File-Specific Query

Query specific files within corpus:

```python
# Get file IDs
files = rag.list_files(corpus_name=corpus.name)
file_ids = [f.name.split("/")[-1] for f in files]

# Query specific files
response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=corpus.name,
            rag_file_ids=file_ids[:3]  # Only first 3 files
        )
    ],
    text="query"
)
```

**Use Case:** When you want to search within specific documents only.

---

## Integration with Gemini

### Basic Integration

```python
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What is RAG?",
    config=GenerateContentConfig(tools=[rag_tool])
)

print(response.text)
```

### With System Instructions

```python
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What is RAG?",
    config=GenerateContentConfig(
        tools=[rag_tool],
        system_instruction="""
        You are a helpful technical assistant. Always use the retrieval tool
        to find information before answering. Provide concise answers with
        citations. If the information is not in the retrieved context, say
        "I don't have that information."
        """
    )
)
```

### Multi-Turn Conversations

```python
# Create chat session
chat = client.chats.create(
    model="gemini-2.0-flash",
    config=GenerateContentConfig(tools=[rag_tool])
)

# Turn 1
response1 = chat.send_message("What is RAG?")
print(response1.text)

# Turn 2 (maintains context)
response2 = chat.send_message("How does it reduce hallucinations?")
print(response2.text)

# Turn 3
response3 = chat.send_message("Give me an example.")
print(response3.text)
```

### With Generation Config

```python
from google.genai.types import GenerationConfig

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What is RAG?",
    config=GenerateContentConfig(
        tools=[rag_tool],
        temperature=0.2,  # Lower = more factual
        top_p=0.8,
        top_k=40,
        max_output_tokens=512
    )
)
```

---

## Advanced Techniques

### 1. Custom Embedding Models

```python
# Use custom embedding endpoint
corpus = rag.create_corpus(
    display_name="custom-emb-corpus",
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                # Custom model endpoint
                endpoint=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
            )
        )
    )
)
```

### 2. Hybrid Context Assembly

Combine RAG retrieval with other sources:

```python
# 1. Get RAG contexts
rag_response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
    text="query",
    rag_retrieval_config=rag.RagRetrievalConfig(top_k=5)
)

rag_contexts = [ctx.text for ctx in rag_response.contexts.contexts]

# 2. Get other contexts (e.g., from database)
db_contexts = fetch_from_database("query")

# 3. Combine
all_contexts = rag_contexts + db_contexts

# 4. Custom prompt
prompt = f"""Answer using the provided contexts.

Question: {query}

Contexts:
{chr(10).join(all_contexts)}

Answer:"""

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)
```

### 3. Re-Ranking

```python
# 1. Broad retrieval (top 50)
response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
    text="query",
    rag_retrieval_config=rag.RagRetrievalConfig(top_k=50)
)

# 2. Re-rank with cross-encoder
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

contexts = [(ctx.text, ctx.distance) for ctx in response.contexts.contexts]
scores = reranker.predict([("query", ctx[0]) for ctx in contexts])

# 3. Sort by reranker scores
reranked = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)

# 4. Use top 10 reranked contexts
top_contexts = [ctx[0][0] for ctx in reranked[:10]]
```

### 4. Streaming Responses

```python
# Stream generation
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain RAG in detail",
    config=GenerateContentConfig(tools=[rag_tool]),
    stream=True
)

for chunk in response:
    print(chunk.text, end="", flush=True)
```

---

## Evaluation and Monitoring

### Evaluation Metrics

```python
from vertexai import rag

# Evaluate RAG quality
eval_results = rag.evaluate(
    corpus_name=corpus.name,
    test_queries=[
        {"query": "What is RAG?", "expected_answer": "..."},
        {"query": "How does RAG work?", "expected_answer": "..."},
    ],
    metrics=[
        "retrieval_recall",
        "retrieval_precision",
        "answer_relevance",
        "faithfulness"
    ]
)

print(eval_results)
```

### Monitor RAG Pipeline

```python
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rag_query_with_logging(question):
    logger.info(f"Query: {question}")

    # Retrieve
    start = time.time()
    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
        text=question
    )
    retrieval_time = time.time() - start

    logger.info(f"Retrieved {len(response.contexts.contexts)} contexts in {retrieval_time:.2f}s")

    # Generate
    start = time.time()
    gen_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=question,
        config=GenerateContentConfig(tools=[rag_tool])
    )
    generation_time = time.time() - start

    logger.info(f"Generated answer in {generation_time:.2f}s")

    return gen_response.text
```

---

## Troubleshooting

### Common Issues

#### Issue: Corpus creation fails

**Error:** `FAILED_PRECONDITION: Service account not found`

**Solution:**
```bash
# RAG Engine service account is auto-created on first corpus creation
# Wait 60 seconds after first API call, then retry

# Or manually create:
gcloud beta services identity create --service=aiplatform.googleapis.com
```

#### Issue: File upload fails with permission error

**Error:** `PERMISSION_DENIED: Service account lacks access to GCS`

**Solution:**
```bash
# Grant service account access to GCS bucket
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
SERVICE_ACCOUNT="service-${PROJECT_NUMBER}@gcp-sa-vertex-rag.iam.gserviceaccount.com"

gsutil iam ch serviceAccount:${SERVICE_ACCOUNT}:roles/storage.objectViewer gs://your-bucket
```

#### Issue: Poor retrieval quality

**Causes:**
1. Chunk size too large/small
2. Low similarity threshold
3. Not enough documents

**Solutions:**
```python
# Tune chunking
transformation_config = rag.TransformationConfig(
    chunking_config=rag.ChunkingConfig(
        chunk_size=512,  # Try different sizes
        chunk_overlap=50
    )
)

# Lower similarity threshold
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=10,
    filter=rag.Filter(
        vector_distance_threshold=0.3  # Lower = more permissive
    )
)

# Add more documents
rag.import_files(corpus_name=corpus.name, paths=[...])
```

#### Issue: LLM not using retrieved context

**Causes:**
1. Retrieved context not relevant
2. LLM ignoring tool
3. System instruction too restrictive

**Solutions:**
```python
# 1. Verify retrieval
response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
    text="query"
)
print(response.contexts.contexts)  # Check if relevant

# 2. Explicit system instruction
system_instruction = """
IMPORTANT: Always use the retrieval tool to find relevant information
before answering. Base your answer ONLY on the retrieved context.
"""

# 3. Lower temperature
config = GenerateContentConfig(
    tools=[rag_tool],
    temperature=0.1,  # More deterministic
    system_instruction=system_instruction
)
```

---

## Best Practices

### 1. Corpus Organization

✅ **DO:**
- Create separate corpora for different content types
- Use descriptive display names
- Group related documents together
- Version corpora (v1, v2, ...)

❌ **DON'T:**
- Mix unrelated content in one corpus
- Use generic names ("corpus-1")
- Create too many small corpora

### 2. Chunking Strategy

✅ **DO:**
- Start with 512 tokens, tune based on results
- Use overlap (50-100 tokens) to avoid context loss
- Test different chunk sizes for your use case

❌ **DON'T:**
- Use very large chunks (> 2048 tokens)
- Use zero overlap
- Forget to validate chunking quality

### 3. Retrieval Configuration

✅ **DO:**
- Set `top_k` to 2-3x expected relevant chunks
- Use reasonable similarity threshold (0.3-0.6)
- Monitor retrieval quality

❌ **DON'T:**
- Set `top_k` too high (wastes cost)
- Use very strict threshold (0.8+) unless needed
- Ignore retrieval metrics

### 4. Production Deployment

✅ **DO:**
- Monitor latency and costs
- Implement caching for frequent queries
- Use streaming for long responses
- Set up error handling and retries

❌ **DON'T:**
- Deploy without monitoring
- Ignore cost spikes
- Skip error handling

---

## Comparison with Custom RAG

| Aspect | RAG Engine | Custom RAG |
|--------|-----------|------------|
| **Setup Time** | 5 minutes | Hours to days |
| **Code Complexity** | < 20 lines | 100s of lines |
| **Maintenance** | Minimal (managed) | High (you maintain) |
| **Customization** | Limited | Full control |
| **Cost** | $$$ (managed overhead) | $$ (DIY cheaper) |
| **Best For** | Rapid development, Gemini apps | Custom requirements, other LLMs |

### When to Use Each

**Use RAG Engine:**
- Building RAG from scratch
- Using Gemini models
- Want managed infrastructure
- Need quick prototyping
- Standard RAG use cases

**Use Custom RAG:**
- Need custom chunking logic
- Using non-Gemini LLMs (GPT, Claude, etc.)
- Need advanced retrieval (hybrid search, re-ranking)
- Have existing vector DB
- Need full control over pipeline

---

## Summary

RAG Engine is Vertex AI's **high-level managed framework** for building RAG applications. Key takeaways:

### Key Features
- **Unified API**: Single interface for multiple vector databases
- **Automatic Processing**: Chunking, embedding, indexing
- **Native Gemini Integration**: Built for Gemini models
- **Multi-Source Ingestion**: GCS, Drive, local files

### Supported Vector Databases
- ✅ Vertex AI Vector Search (recommended)
- ✅ Pinecone
- ✅ Weaviate
- ✅ Feature Store
- ✅ Vertex AI Search

### When to Use
- ✅ Building RAG quickly
- ✅ Using Gemini
- ✅ Want managed infrastructure
- ❌ Need custom chunking (limited options)
- ❌ Using non-Gemini LLMs

### Next Steps
1. Try the [Quickstart](#quickstart-5-minutes)
2. Explore [Vector Database Options](#vector-database-options)
3. Read [Comparison Guide](./guide_rag_search_comparison.md)

---

## Additional Resources

### Documentation
- [RAG Engine Overview](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview)
- [RAG API Reference](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/rag-api)

### Tutorials
- [Intro RAG Engine](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/intro_rag_engine.ipynb)
- [RAG Engine with Vector Search](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_vector_search.ipynb)
- [RAG Engine with Pinecone](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_pinecone.ipynb)
- [RAG Engine Evaluation](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_evaluation.ipynb)

---

**End of Manual** | [View Comparison Guide →](./guide_rag_search_comparison.md)

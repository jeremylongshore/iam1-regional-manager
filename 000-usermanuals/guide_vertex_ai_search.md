# Vertex AI Search (Enterprise Search): Comprehensive User Manual for AI Agents

**Version:** 1.0
**Last Updated:** 2025-11-12
**Target Audience:** AI Agents learning RAG and search technologies on Google Cloud

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [What is Vertex AI Search](#what-is-vertex-ai-search)
3. [When to Use Vertex AI Search](#when-to-use-vertex-ai-search)
4. [Core Concepts](#core-concepts)
5. [Architecture](#architecture)
6. [Quickstart (15 Minutes)](#quickstart-15-minutes)
7. [Complete Implementation Guide](#complete-implementation-guide)
8. [Blended Search (Multi-Datastore)](#blended-search-multi-datastore)
9. [Grounding with Gemini](#grounding-with-gemini)
10. [Search Tuning and Optimization](#search-tuning-and-optimization)
11. [Cost and Performance](#cost-and-performance)
12. [Integration Patterns](#integration-patterns)
13. [Troubleshooting](#troubleshooting)
14. [Best Practices](#best-practices)
15. [Anti-Patterns](#anti-patterns)

---

## Executive Summary

**Vertex AI Search** (formerly Gen App Builder Search) is Google Cloud's fully-managed enterprise search platform that brings **Google-quality search** to your private data. It combines decades of Google Search expertise with modern LLM capabilities.

### Key Capabilities

- **Out-of-the-Box Search**: Deploy production search in minutes, no ML expertise needed
- **Hybrid Search**: Automatic blending of semantic search + keyword search + LLM understanding
- **Multi-Source**: Search across GCS, BigQuery, websites, databases simultaneously
- **LLM Grounding**: Ground Gemini responses in your data with citations
- **Extractive Answers**: LLM-powered answer extraction from documents
- **Ranking API**: Google-quality ranking and re-ranking
- **Enterprise Features**: Access controls, compliance, audit logs

### What You'll Learn

- How to create datastores and search apps
- Semantic + keyword search with automatic blending
- LLM grounding and answer generation
- Search tuning and quality optimization
- Integration with Gemini and LangChain
- Production deployment patterns

---

## What is Vertex AI Search

### The Complete Enterprise Search Solution

Unlike Vector Search (which provides only vector similarity), Vertex AI Search is a **complete search engine** with:

```
┌──────────────────────────────────────────────────────┐
│  VERTEX AI SEARCH (Enterprise Search)                │
│                                                       │
│  ✓ Keyword Search (BM25, TF-IDF)                    │
│  ✓ Semantic Search (Dense Embeddings)                │
│  ✓ LLM Understanding (Query Intent)                  │
│  ✓ Document Processing (PDF, HTML, TXT, ...)        │
│  ✓ Ranking & Re-ranking                              │
│  ✓ Filtering & Faceting                              │
│  ✓ Spell Correction                                  │
│  ✓ Query Expansion                                   │
│  ✓ Answer Extraction                                 │
│  ✓ Summarization                                     │
│  ✓ Citation & Grounding                              │
└──────────────────────────────────────────────────────┘
```

### Google Search for Your Data

Vertex AI Search brings the same technology powering Google Search to your enterprise:

1. **Deep Information Retrieval**: 20+ years of Google Search innovations
2. **Natural Language Understanding**: LLM-powered query understanding
3. **Scalability**: Handle billions of documents
4. **Relevance**: Machine-learned ranking algorithms

---

## When to Use Vertex AI Search

### ✅ Ideal Use Cases

| Use Case | Description | Why Vertex AI Search |
|----------|-------------|---------------------|
| **Enterprise Knowledge Base** | Search internal docs, wikis, support articles | Out-of-box search quality + LLM answers |
| **Customer Support** | Find relevant help articles | Answer extraction + citations |
| **E-commerce Product Search** | Product catalog search | Hybrid search + filters + facets |
| **Legal/Compliance** | Search contracts, policies | Document parsing + exact keyword match |
| **RAG Applications** | Ground LLM responses | Native Gemini integration |
| **Website Search** | Search your public website | Crawl and index automatically |
| **Multi-Source Search** | Search across GCS, BigQuery, websites | Data blending built-in |

### ✅ vs ❌ Comparison

| Scenario | Vertex AI Search | Vector Search |
|----------|------------------|---------------|
| Need keyword + semantic search | ✅ Built-in | ❌ Only semantic (unless hybrid) |
| Need answer extraction | ✅ Built-in | ❌ Must build custom |
| Need document parsing | ✅ Auto-parse PDF/HTML | ❌ Must pre-process |
| Need filters/facets | ✅ Built-in | ✅ With restricts |
| Maximum control over vectors | ❌ Abstracted | ✅ Full control |
| Lowest latency (< 10ms) | ❌ 50-200ms | ✅ 10-50ms |
| Cheapest for simple search | ✅ Pay-per-query | ❌ Pay-per-hour endpoints |

---

## Core Concepts

### 1. Data Stores

**Data Stores** are collections of documents to search. Types:

| Type | Source | Use Case |
|------|--------|----------|
| **Unstructured** | GCS (PDF, HTML, TXT, DOCX, etc.) | Documents, manuals, wikis |
| **Structured** | BigQuery tables | Products, users, transactions |
| **Website** | Public websites (crawler) | Public documentation, blogs |

### 2. Search Apps (Engines)

**Search Apps** (Engines) are search endpoints connected to one or more data stores.

```
Search App "Customer Support"
├── Data Store 1: Help Articles (GCS)
├── Data Store 2: Product Docs (Website)
└── Data Store 3: Support Tickets (BigQuery)
```

**Search Tiers:**

| Tier | Features | Price |
|------|----------|-------|
| **Basic** | Keyword search, basic ranking | Lower cost |
| **Enterprise** | Semantic search, LLM features, extractive answers | Higher cost |

**Recommendation**: Use **Enterprise tier** for RAG and LLM grounding.

### 3. Search Features

#### Content Search Spec

Configuration for search behavior:

- **Snippet Spec**: Return snippets (excerpts) from documents
- **Summary Spec**: LLM-generated summaries with citations
- **Extractive Answer Spec**: Extract precise answers to questions
- **Extractive Segment Spec**: Extract relevant segments

#### Query Expansion

Automatically expand queries with synonyms:

- `AUTO`: Let system decide
- `DISABLED`: Exact query only

#### Spell Correction

Automatically fix typos:

- `AUTO`: Suggest corrections
- `DISABLED`: Use exact query

### 4. Grounding

**Grounding** connects LLM outputs to factual sources:

```
User Query: "What was revenue in Q4 2020?"
      ↓
[Vertex AI Search]
Retrieve: "Q4 2020 revenue was $56.9B" (from earnings.pdf)
      ↓
[Gemini with Grounding]
Generate: "According to the Q4 2020 earnings report,
           revenue was $56.9 billion. [1]"
      ↓
[1] earnings.pdf, page 3
```

**Benefits:**
- Reduce hallucinations
- Provide citations/sources
- Keep answers up-to-date

### 5. Ranking

**Ranking API** provides Google-quality re-ranking:

1. **First-stage retrieval**: Get top 100-1000 candidates (fast)
2. **Second-stage ranking**: Re-rank with heavy models (accurate)

---

## Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────┐
│                  YOUR APPLICATION                    │
│  (Web App, Chatbot, API Service)                    │
└────────────────────┬─────────────────────────────────┘
                     │
                     │ Search API Request
                     ↓
┌──────────────────────────────────────────────────────┐
│              VERTEX AI SEARCH APP                    │
│  (Search Engine Configuration)                       │
│  - Query Understanding (LLM)                         │
│  - Query Expansion                                   │
│  - Spell Correction                                  │
└────────────────────┬─────────────────────────────────┘
                     │
                     ↓ Query to Data Stores
┌──────────────────────────────────────────────────────┐
│                   DATA STORES                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ GCS Docs   │  │ BigQuery   │  │ Website    │    │
│  └────────────┘  └────────────┘  └────────────┘    │
│                                                      │
│  - Document Parsing                                  │
│  - Indexing (Keyword + Semantic)                    │
│  - Ranking Models                                    │
└────────────────────┬─────────────────────────────────┘
                     │
                     ↓ Return Results
┌──────────────────────────────────────────────────────┐
│              SEARCH RESULTS                          │
│  - Snippets                                          │
│  - Extractive Answers                                │
│  - LLM Summary (with citations)                      │
│  - Facets / Filters                                  │
└──────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingestion**: Upload documents → Auto-parsing → Indexing
2. **Query**: User query → Query understanding → Multi-modal search
3. **Retrieval**: Keyword search + Semantic search → Merge results
4. **Ranking**: Machine-learned ranking → Top-K results
5. **Enhancement**: Answer extraction → Summarization → Citations

---

## Quickstart (15 Minutes)

### Prerequisites

```bash
# Install SDK
pip install google-cloud-discoveryengine google-cloud-aiplatform

# Set environment
export PROJECT_ID="your-project-id"
export LOCATION="global"  # or us, eu

# Enable APIs
gcloud services enable discoveryengine.googleapis.com

# Authenticate
gcloud auth application-default login
```

### Step 1: Create Data Store (5 minutes)

**Option A: Via Console** (Recommended for first time)

1. Go to [Vertex AI Agent Builder](https://console.cloud.google.com/gen-app-builder)
2. Click **Create Data Store**
3. Choose **Cloud Storage** as source
4. Enter bucket: `gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs`
5. Click **Create**

**Option B: Via Python SDK**

```python
from google.cloud import discoveryengine
from google.api_core.client_options import ClientOptions

PROJECT_ID = "your-project-id"
LOCATION = "global"
DATASTORE_ID = "my-first-datastore"

# Create client
client_options = ClientOptions(
    api_endpoint=f"{LOCATION}-discoveryengine.googleapis.com"
) if LOCATION != "global" else None
client = discoveryengine.DataStoreServiceClient(client_options=client_options)

# Create data store
data_store = discoveryengine.DataStore(
    display_name="My First Data Store",
    industry_vertical=discoveryengine.IndustryVertical.GENERIC,
    content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
)

operation = client.create_data_store(
    parent=client.collection_path(PROJECT_ID, LOCATION, "default_collection"),
    data_store=data_store,
    data_store_id=DATASTORE_ID,
)

try:
    response = operation.result(timeout=90)
    print(f"Data store created: {response.name}")
except:
    print("Data store creation in progress...")
```

### Step 2: Import Documents (5 minutes)

```python
from google.cloud import discoveryengine

# Import documents from GCS
source_uri = "gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs/*"

request = discoveryengine.ImportDocumentsRequest(
    parent=client.branch_path(
        project=PROJECT_ID,
        location=LOCATION,
        data_store=DATASTORE_ID,
        branch="default_branch"
    ),
    gcs_source=discoveryengine.GcsSource(
        input_uris=[source_uri],
        data_schema="content"  # Auto-parse documents
    ),
    reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
)

operation = client.import_documents(request=request)
print(f"Import started: {operation.operation.name}")
# Wait 5-10 minutes for documents to be indexed
```

### Step 3: Create Search App (2 minutes)

```python
from google.cloud import discoveryengine

# Create search engine
engine = discoveryengine.Engine(
    display_name="My Search App",
    solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
    industry_vertical=discoveryengine.IndustryVertical.GENERIC,
    data_store_ids=[DATASTORE_ID],

    # Enable Enterprise features (LLM, extractive answers)
    search_engine_config=discoveryengine.Engine.SearchEngineConfig(
        search_tier=discoveryengine.SearchTier.SEARCH_TIER_ENTERPRISE,
        search_add_ons=[discoveryengine.SearchAddOn.SEARCH_ADD_ON_LLM],
    ),
)

engine_client = discoveryengine.EngineServiceClient(client_options=client_options)
operation = engine_client.create_engine(
    parent=engine_client.collection_path(PROJECT_ID, LOCATION, "default_collection"),
    engine=engine,
    engine_id="my-search-app",
)

response = operation.result(timeout=90)
print(f"Search app created: {response.name}")
```

### Step 4: Search! (< 1 minute)

```python
from google.cloud import discoveryengine

# Create search client
search_client = discoveryengine.SearchServiceClient(client_options=client_options)

# Build request
serving_config = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/"
    f"collections/default_collection/engines/my-search-app/"
    f"servingConfigs/default_search"
)

# Search with LLM summary
request = discoveryengine.SearchRequest(
    serving_config=serving_config,
    query="Who is the CEO of Google?",
    page_size=10,

    # Enable features
    content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True
        ),
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=5,
            include_citations=True,
        ),
    ),
    query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
        condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO
    ),
    spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
        mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
    ),
)

# Execute search
response = search_client.search(request)

# Print summary
print("Summary:", response.summary.summary_text)

# Print results
for result in response.results:
    doc = result.document
    print(f"\nTitle: {doc.derived_struct_data.get('title', 'N/A')}")
    print(f"Link: {doc.derived_struct_data.get('link', 'N/A')}")
```

🎉 **Done!** You've built an enterprise search engine with LLM summaries.

---

## Complete Implementation Guide

### Data Store Types in Detail

#### 1. Unstructured Data Store (Documents)

**Supported Formats:**
- PDF, HTML, TXT, DOCX, PPTX
- JSON, CSV (with schema)
- Markdown

**Example: GCS Documents**

```python
# Upload documents to GCS
!gsutil -m cp *.pdf gs://my-bucket/docs/

# Import to data store
request = discoveryengine.ImportDocumentsRequest(
    parent=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/dataStores/{DATASTORE_ID}/branches/default_branch",
    gcs_source=discoveryengine.GcsSource(
        input_uris=["gs://my-bucket/docs/*.pdf"],
        data_schema="content"  # Auto-parse
    ),
    reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
)

operation = document_client.import_documents(request=request)
```

**Document Metadata (Optional):**

```json
{
  "id": "doc-123",
  "jsonData": "{\"title\": \"Product Manual\", \"category\": \"electronics\"}",
  "content": {
    "mimeType": "application/pdf",
    "uri": "gs://my-bucket/docs/manual.pdf"
  }
}
```

#### 2. Structured Data Store (BigQuery)

**Example: Product Catalog**

```python
# BigQuery table schema
# products table:
# - product_id (STRING)
# - name (STRING)
# - description (STRING)
# - price (FLOAT64)
# - category (STRING)

# Create structured data store
data_store = discoveryengine.DataStore(
    display_name="Product Catalog",
    industry_vertical=discoveryengine.IndustryVertical.RETAIL,
    content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
)

# Import from BigQuery
request = discoveryengine.ImportDocumentsRequest(
    parent=branch_path,
    bigquery_source=discoveryengine.BigQuerySource(
        project_id=PROJECT_ID,
        dataset_id="my_dataset",
        table_id="products",
        data_schema="custom",  # Use BigQuery schema
        # Map fields
        # id_field: "product_id"
        # content_field: "name,description"  # Searchable fields
    ),
)

operation = document_client.import_documents(request=request)
```

#### 3. Website Data Store (Crawler)

```python
# Create website data store
data_store = discoveryengine.DataStore(
    display_name="Company Website",
    industry_vertical=discoveryengine.IndustryVertical.GENERIC,
    content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
)

# Configure crawler (done via Console)
# 1. Specify URL patterns to include/exclude
# 2. Set crawl frequency
# 3. Advanced: Custom headers, JavaScript rendering

# URL patterns:
# Include: https://example.com/docs/*
# Exclude: https://example.com/admin/*
```

### Search Configuration Options

#### Complete Search Request

```python
request = discoveryengine.SearchRequest(
    serving_config=serving_config,
    query="user query",
    page_size=10,
    offset=0,  # For pagination

    # ===== CONTENT SEARCH =====
    content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(

        # Snippets (excerpts)
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True,
            max_snippet_count=5,
        ),

        # LLM Summary
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=5,  # Use top 5 results for summary
            include_citations=True,
            ignore_adversarial_query=True,  # Ignore harmful queries
            ignore_non_summary_seeking_query=True,  # Skip "navigational" queries

            # Custom prompt for summarization
            model_prompt_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
                preamble="You are a helpful assistant. Provide concise answers with citations."
            ),

            # Model version
            model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                version="stable"  # or "preview" for latest
            ),
        ),

        # Extractive Answers (precise answers)
        extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
            max_extractive_answer_count=3,
            max_extractive_segment_count=1,
        ),
    ),

    # ===== QUERY EXPANSION =====
    query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
        condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO
        # AUTO: System decides
        # DISABLED: No expansion
    ),

    # ===== SPELL CORRECTION =====
    spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
        mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        # AUTO: Suggest corrections
        # SUGGESTION_ONLY: Show suggestion but don't auto-correct
    ),

    # ===== FILTERING =====
    filter='category: "electronics" AND price < 100',

    # ===== FACETING =====
    facet_specs=[
        discoveryengine.SearchRequest.FacetSpec(
            facet_key=discoveryengine.SearchRequest.FacetSpec.FacetKey(
                key="category"
            ),
            limit=10  # Top 10 categories
        ),
    ],

    # ===== BOOST/BURY =====
    boost_spec=discoveryengine.SearchRequest.BoostSpec(
        condition_boost_specs=[
            discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                condition='category: "featured"',
                boost=2.0  # 2x boost for featured items
            )
        ]
    ),

    # ===== RANKING =====
    # (Configured in Console or via Ranking API)
)

response = search_client.search(request)
```

### Response Handling

```python
# Print summary
if response.summary:
    print("=== SUMMARY ===")
    print(response.summary.summary_text)

    if response.summary.summary_with_metadata:
        print("\n=== CITATIONS ===")
        for citation in response.summary.summary_with_metadata.citations:
            for source in citation.sources:
                print(f"[{source.reference_index}] {source.uri}")

# Print extractive answers
if response.results:
    for i, result in enumerate(response.results[:3]):
        doc_data = result.document.derived_struct_data

        print(f"\n=== RESULT {i+1} ===")
        print(f"Title: {doc_data.get('title', 'N/A')}")
        print(f"Link: {doc_data.get('link', 'N/A')}")

        # Snippets
        if 'snippets' in doc_data:
            print("\nSnippets:")
            for snippet in doc_data['snippets']:
                print(f"  - {snippet.get('snippet', '')}")

        # Extractive answers
        if 'extractive_answers' in doc_data:
            print("\nAnswers:")
            for answer in doc_data['extractive_answers']:
                print(f"  - {answer.get('content', '')}")

# Print facets
if response.facets:
    print("\n=== FACETS ===")
    for facet in response.facets:
        print(f"{facet.key}:")
        for value in facet.values:
            print(f"  {value.value} ({value.count})")
```

---

## Blended Search (Multi-Datastore)

Search across multiple data sources simultaneously.

### Setup Multi-Datastore Search App

```python
# Create multiple data stores
gcs_datastore_id = "docs-gcs"
bq_datastore_id = "products-bq"
website_datastore_id = "website-crawler"

# Create search app with multiple data stores
engine = discoveryengine.Engine(
    display_name="Blended Search App",
    solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
    industry_vertical=discoveryengine.IndustryVertical.GENERIC,

    # Link all data stores
    data_store_ids=[gcs_datastore_id, bq_datastore_id, website_datastore_id],

    search_engine_config=discoveryengine.Engine.SearchEngineConfig(
        search_tier=discoveryengine.SearchTier.SEARCH_TIER_ENTERPRISE,
        search_add_ons=[discoveryengine.SearchAddOn.SEARCH_ADD_ON_LLM],
    ),
)

# Create engine
operation = engine_client.create_engine(
    parent=engine_client.collection_path(PROJECT_ID, LOCATION, "default_collection"),
    engine=engine,
    engine_id="blended-search",
)
```

### Query Blended Search

```python
# Query searches ALL data stores automatically
response = search_client.search(
    discoveryengine.SearchRequest(
        serving_config=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/engines/blended-search/servingConfigs/default_search",
        query="cloud storage pricing",
        page_size=10,

        # Results will come from all data stores
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=5,
                include_citations=True,
            ),
        ),
    )
)

# Response includes results from all sources
for result in response.results:
    doc = result.document
    datastore = doc.parent_document_id  # Which datastore it came from
    print(f"From {datastore}: {doc.derived_struct_data.get('title')}")
```

---

## Grounding with Gemini

Integrate Vertex AI Search with Gemini for grounded generation.

### Method 1: Direct Grounding with GenerativeModel

```python
from vertexai.generative_models import GenerativeModel, Tool
import vertexai.preview.generative_models as generative_models

# Initialize Vertex AI
import vertexai
vertexai.init(project=PROJECT_ID, location="us-central1")

# Configure grounding with Vertex AI Search
grounding_tool = Tool.from_retrieval(
    retrieval=generative_models.grounding.Retrieval(
        source=generative_models.grounding.VertexAISearch(
            datastore=DATASTORE_ID,
            project=PROJECT_ID,
            location=LOCATION,
        ),
        disable_attribution=False,  # Include citations
    )
)

# Create model with grounding
model = GenerativeModel(
    "gemini-2.0-flash",
    tools=[grounding_tool],
    system_instruction="""
    You are a helpful assistant. Always use the grounding tool to find
    relevant information before answering. Include citations.
    """
)

# Generate grounded response
response = model.generate_content(
    "What was Google's revenue in Q4 2020?",
    generation_config={"temperature": 0.2, "max_output_tokens": 512}
)

print(response.text)

# Access grounding metadata
if response.candidates[0].grounding_metadata:
    print("\n=== GROUNDING SOURCES ===")
    for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
        print(f"Source: {chunk.web.uri if hasattr(chunk, 'web') else 'N/A'}")
```

### Method 2: LangChain Integration

```python
from langchain_google_community import VertexAISearchRetriever
from langchain_google_vertexai import VertexAI
from langchain.chains import RetrievalQAWithSourcesChain

# Create retriever
retriever = VertexAISearchRetriever(
    project_id=PROJECT_ID,
    location_id=LOCATION,
    data_store_id=DATASTORE_ID,

    # Retrieval configuration
    get_extractive_answers=True,
    max_documents=10,
    max_extractive_answer_count=5,
)

# Create LLM
llm = VertexAI(
    model_name="gemini-2.0-flash",
    temperature=0.2,
    max_output_tokens=512
)

# Create QA chain
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Query
result = qa_chain.invoke(
    "What was Google's revenue in Q4 2020?",
    return_only_outputs=True
)

print("Answer:", result['answer'])
print("Sources:", result['sources'])
```

### Method 3: Custom Grounding Pipeline

```python
import re
from google.cloud import discoveryengine
from vertexai.generative_models import GenerativeModel

class GroundedQA:
    def __init__(self, project_id, location, datastore_id, engine_id):
        self.search_client = self._create_search_client(location)
        self.serving_config = (
            f"projects/{project_id}/locations/{location}/"
            f"collections/default_collection/engines/{engine_id}/"
            f"servingConfigs/default_config"
        )
        self.llm = GenerativeModel("gemini-2.0-flash")

    def answer_question(self, question):
        # 1. Retrieve relevant documents
        search_response = self.search_client.search(
            discoveryengine.SearchRequest(
                serving_config=self.serving_config,
                query=question,
                page_size=5,
                content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                    snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                        return_snippet=True
                    ),
                ),
            )
        )

        # 2. Extract snippets
        snippets = []
        sources = []

        for i, result in enumerate(search_response.results):
            doc_data = result.document.derived_struct_data

            # Get snippets
            if 'snippets' in doc_data:
                for snippet in doc_data['snippets']:
                    text = re.sub("<[^>]*>", "", snippet.get('snippet', ''))
                    snippets.append(text)

            # Get source
            title = doc_data.get('title', 'Unknown')
            link = doc_data.get('link', 'N/A')
            sources.append({"title": title, "link": link, "index": i+1})

        # 3. Create grounded prompt
        context = "\n\n".join([
            f"Document {i+1}: {snippet}"
            for i, snippet in enumerate(snippets)
        ])

        prompt = f"""Answer the question using ONLY the provided context.
Include citations using [1], [2], etc.

Question: {question}

Context:
{context}

Answer:"""

        # 4. Generate answer
        response = self.llm.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 512}
        )

        return {
            "answer": response.text,
            "sources": sources
        }

# Usage
qa = GroundedQA(PROJECT_ID, LOCATION, DATASTORE_ID, ENGINE_ID)
result = qa.answer_question("What was Google's revenue in Q4 2020?")

print(result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(f"[{source['index']}] {source['title']} - {source['link']}")
```

---

## Search Tuning and Optimization

### 1. Relevance Tuning

**Boosting/Burying:**

```python
# Boost recent documents
boost_spec = discoveryengine.SearchRequest.BoostSpec(
    condition_boost_specs=[
        discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
            condition='publish_date > "2023-01-01"',
            boost=1.5
        ),
        discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
            condition='category: "featured"',
            boost=2.0
        ),
    ]
)

request = discoveryengine.SearchRequest(
    serving_config=serving_config,
    query=query,
    boost_spec=boost_spec
)
```

**Filtering:**

```python
# Filter syntax (similar to BigQuery WHERE clause)
filters = [
    'category: "electronics"',  # Exact match
    'price < 100',  # Numeric
    'publish_date > "2023-01-01"',  # Date
    'tags: ANY("sale", "clearance")',  # Array contains any
    'in_stock = true',  # Boolean
]

combined_filter = ' AND '.join(filters)

request = discoveryengine.SearchRequest(
    serving_config=serving_config,
    query=query,
    filter=combined_filter
)
```

### 2. Custom Ranking

**Upload Training Data (Console):**

1. Prepare CSV with query-document pairs:
   ```csv
   query,document_id,label
   "cloud storage","doc-123",3
   "cloud storage","doc-456",1
   ```

2. Upload via Console → Search App → Ranking

3. Train custom ranking model

**Query with Custom Ranking:**

```python
# Automatically uses custom ranking if configured
response = search_client.search(request)
```

### 3. Synonym Sets

**Create Synonyms (Console):**

```
# Synonyms for "car"
car,vehicle,automobile

# One-way synonym
smartphone => phone
```

### 4. Search Quality Metrics

```python
# Track metrics
from google.cloud import monitoring_v3

# Query latency
# Result clicks
# Zero-result queries
# Query reformulations

# Use Cloud Monitoring for tracking
```

---

## Cost and Performance

### Pricing Model

**Query-Based Pricing:**

| Tier | Price per 1000 Queries | Features |
|------|------------------------|----------|
| **Basic** | ~$3 | Keyword search, basic ranking |
| **Enterprise** | ~$45 | + Semantic search, LLM features, extractive answers |

**Additional Costs:**
- **Storage**: $0.024/GB/month for indexed documents
- **LLM Summaries**: Included in Enterprise tier (up to 5 results/summary)

**Estimator:**
- 100K queries/month × $45/1000 = **$4,500/month**
- 10GB indexed docs × $0.024 = **$0.24/month**
- **Total: ~$4,500/month**

### Performance Characteristics

| Metric | Typical Value |
|--------|---------------|
| **Query Latency** | 50-200ms (p50) |
| **Indexing Latency** | Minutes to hours |
| **Max Documents** | Billions |
| **Max Query Length** | 2048 characters |
| **Concurrent Queries** | Thousands QPS |

### Optimization Strategies

#### 1. Reduce Query Cost

```python
# Use summary only when needed
if user_needs_summary:
    enable_summary = True
else:
    enable_summary = False  # Save cost

# Cache frequent queries
import hashlib

query_hash = hashlib.md5(query.encode()).hexdigest()
if query_hash in cache:
    return cache[query_hash]
else:
    response = search_client.search(request)
    cache[query_hash] = response
    return response
```

#### 2. Reduce Latency

```python
# Request fewer results
page_size=5  # Instead of 50

# Disable unnecessary features
# - No summary for navigational queries
# - No extractive answers for known-item queries

# Use filters to reduce search space
filter='category: "electronics"'
```

---

## Integration Patterns

### Pattern 1: Chatbot with Grounded Answers

```python
class GroundedChatbot:
    def __init__(self, project_id, location, engine_id, datastore_id):
        self.search_client = self._create_search_client(location)
        self.serving_config = self._build_serving_config(
            project_id, location, engine_id
        )
        self.llm = GenerativeModel(
            "gemini-2.0-flash",
            tools=[self._create_grounding_tool(project_id, location, datastore_id)]
        )
        self.conversation_history = []

    def chat(self, user_message):
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Build prompt with history
        prompt = self._build_prompt_with_history()

        # Generate response
        response = self.llm.generate_content(prompt)

        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.text
        })

        return response.text

# Usage
chatbot = GroundedChatbot(PROJECT_ID, LOCATION, ENGINE_ID, DATASTORE_ID)
print(chatbot.chat("What is Cloud Run?"))
print(chatbot.chat("How much does it cost?"))  # Maintains context
```

### Pattern 2: Multi-Modal Search (Text + Metadata)

```python
# Index with rich metadata
{
  "id": "product-123",
  "content": {
    "mimeType": "text/plain",
    "rawBytes": "Product description..."
  },
  "structData": {
    "price": 99.99,
    "category": "electronics",
    "rating": 4.5,
    "in_stock": true,
    "tags": ["wireless", "portable"]
  }
}

# Search with filters and facets
response = search_client.search(
    discoveryengine.SearchRequest(
        serving_config=serving_config,
        query="wireless headphones",

        # Filter
        filter='price < 150 AND in_stock = true',

        # Facets
        facet_specs=[
            discoveryengine.SearchRequest.FacetSpec(
                facet_key=discoveryengine.SearchRequest.FacetSpec.FacetKey(
                    key="category"
                )
            ),
            discoveryengine.SearchRequest.FacetSpec(
                facet_key=discoveryengine.SearchRequest.FacetSpec.FacetKey(
                    key="price",
                    intervals=[
                        {"minimum": 0, "maximum": 50},
                        {"minimum": 50, "maximum": 100},
                        {"minimum": 100, "maximum": 200},
                    ]
                )
            ),
        ],
    )
)
```

---

## Troubleshooting

### Common Issues

#### Issue: No search results returned

**Causes:**
1. Documents not indexed yet
2. Query too specific
3. Filters too restrictive

**Solutions:**
```python
# Check indexing status
datastore = datastore_client.get_data_store(name=datastore_name)
print(f"Document count: {datastore.document_count}")

# Try without filters
response = search_client.search(
    discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        # Remove filter/facets temporarily
    )
)

# Enable query expansion
query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
    condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO
)
```

#### Issue: Poor search quality

**Solutions:**
1. Use Enterprise tier (vs Basic)
2. Add synonyms
3. Boost/bury specific documents
4. Train custom ranking model
5. Improve document metadata

#### Issue: High latency

**Solutions:**
```python
# Reduce page size
page_size=5

# Disable heavy features
# - summary_spec (if not needed)
# - extractive_content_spec (if not needed)

# Use caching for popular queries
```

#### Issue: Grounding not working

**Causes:**
1. Datastore ID incorrect
2. No relevant documents
3. LLM add-on not enabled

**Solutions:**
```python
# Verify datastore ID
print(f"Using datastore: {DATASTORE_ID}")

# Check if LLM add-on enabled
engine = engine_client.get_engine(name=engine_name)
print(f"Search add-ons: {engine.search_engine_config.search_add_ons}")

# Test retrieval directly
response = search_client.search(
    discoveryengine.SearchRequest(
        serving_config=serving_config,
        query="test query",
        page_size=5
    )
)
print(f"Results: {len(response.results)}")
```

---

## Best Practices

### 1. Data Store Design

✅ **DO:**
- Use descriptive display names
- Group related content in same datastore
- Include rich metadata (title, category, date, etc.)
- Use structured data for products/entities
- Set up multiple datastores for different content types

❌ **DON'T:**
- Mix unrelated content in one datastore
- Forget to add metadata
- Use generic IDs (use meaningful IDs)
- Store PII without proper controls

### 2. Search Configuration

✅ **DO:**
- Use Enterprise tier for RAG applications
- Enable query expansion and spell correction
- Request only needed features (save cost)
- Use filters to reduce search space
- Cache frequent queries

❌ **DON'T:**
- Enable all features for every query
- Ignore query performance
- Request more results than needed
- Skip A/B testing for ranking changes

### 3. Grounding

✅ **DO:**
- Include citations in prompts
- Use custom preambles for specific use cases
- Handle "no relevant docs" gracefully
- Monitor grounding quality
- Validate citations programmatically

❌ **DON'T:**
- Trust LLM without verification
- Ignore low-confidence answers
- Skip source attribution
- Allow hallucinations in production

### 4. Production Deployment

✅ **DO:**
- Monitor query latency and errors
- Set up alerts for indexing failures
- Version your search apps (v1, v2, ...)
- Test queries before deploying ranking changes
- Use separate environments (dev/staging/prod)

❌ **DON'T:**
- Deploy untested changes to production
- Ignore slow queries
- Skip monitoring and alerting
- Use production for testing

---

## Anti-Patterns

### ❌ Anti-Pattern 1: Using Vertex AI Search for Pure Vector Similarity

**Bad:**
```python
# Using Vertex AI Search only for semantic similarity
# (ignoring keyword search, ranking, etc.)
```

**Why:** Vertex AI Search is optimized for full-text + semantic hybrid search. For pure vector similarity, use Vector Search (cheaper, faster).

**Good:**
```python
# Use Vector Search for pure semantic similarity
# Use Vertex AI Search for hybrid search with keywords, filters, ranking
```

### ❌ Anti-Pattern 2: Not Using Filters

**Bad:**
```python
# Search everything, filter in application
results = search(query="shoes")
filtered = [r for r in results if r.price < 100]
```

**Good:**
```python
# Filter in search engine
results = search(query="shoes", filter="price < 100")
```

### ❌ Anti-Pattern 3: Ignoring Search Quality

**Bad:**
```python
# Deploy and forget
create_search_app()
# Never monitor or tune
```

**Good:**
```python
# Continuous improvement
create_search_app()
monitor_query_metrics()
analyze_zero_result_queries()
add_synonyms()
tune_ranking()
```

---

## Summary

Vertex AI Search is Google Cloud's **complete enterprise search solution** that combines keyword search, semantic search, and LLM capabilities. Key takeaways:

### When to Use
- ✅ Need out-of-box search quality
- ✅ Want hybrid search (keyword + semantic)
- ✅ Need LLM grounding with citations
- ✅ Have multiple data sources (GCS, BigQuery, websites)
- ❌ Need pure vector similarity (use Vector Search)
- ❌ Need < 10ms latency (use Vector Search)

### Key Features
- **Hybrid Search**: Automatic blending of keyword + semantic
- **LLM Integration**: Native Gemini grounding
- **Multi-Source**: Search across datastores simultaneously
- **Enterprise**: Access controls, compliance, audit logs

### Costs
- **Basic Tier**: ~$3 per 1000 queries
- **Enterprise Tier**: ~$45 per 1000 queries (includes LLM features)
- **Storage**: $0.024/GB/month

### Performance
- **Latency**: 50-200ms (p50)
- **Scale**: Billions of documents
- **Throughput**: Thousands QPS

### Next Steps
1. Try the [Quickstart](#quickstart-15-minutes)
2. Experiment with [Grounding](#grounding-with-gemini)
3. Read the [Comparison Guide](./guide_rag_search_comparison.md)
4. Explore [RAG Engine](./guide_rag_engine.md) for orchestration

---

## Additional Resources

### Documentation
- [Vertex AI Search Overview](https://cloud.google.com/generative-ai-app-builder/docs/enterprise-search-introduction)
- [Create Data Stores](https://cloud.google.com/generative-ai-app-builder/docs/create-data-store-es)
- [Search API Reference](https://cloud.google.com/generative-ai-app-builder/docs/reference/rest)
- [Grounding with Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/grounding)

### Tutorials
- [Create Datastore and Search](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/search/create_datastore_and_search.ipynb)
- [Data Blending with Gemini Summarization](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/search/search_data_blending_with_gemini_summarization.ipynb)
- [Vertex AI Search Options](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/search/vertexai-search-options/vertexai_search_options.ipynb)

---

**End of Manual** | [View Comparison Guide →](./guide_rag_search_comparison.md)

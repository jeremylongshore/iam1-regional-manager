# Agent Scaffolding Templates: Complete Guide for AI Agents

## Purpose

This manual explains **agent scaffolding** - reusable code templates for building AI agents. This is different from infrastructure-as-code (Terraform), which provisions cloud resources.

**What you'll learn**:
- Difference between scaffolds (code templates) and infrastructure (Terraform)
- Agent architecture patterns and when to use each
- Ready-to-use scaffolding systems (Agent Starter Pack)
- How to create custom agent templates
- Best practices for agent development

---

## Key Concept: Scaffolds vs Infrastructure

### Scaffolds (Agent Code Templates)
**What**: Pre-built code structure for building agents
**Purpose**: Speed up development, enforce best practices, ensure consistency
**Examples**: Google Agent Starter Pack, LangChain templates, custom templates
**Output**: Python/JavaScript code, agent definitions, tool integrations

```
Agent Scaffold Creates:
├── app/
│   ├── agent.py           # Agent logic
│   ├── tools.py           # Tool definitions
│   └── retrievers.py      # RAG components
├── tests/
│   ├── unit/
│   └── integration/
├── Makefile
└── pyproject.toml
```

### Infrastructure (Terraform, IaC)
**What**: Cloud resource provisioning
**Purpose**: Create and manage cloud infrastructure
**Examples**: Terraform, CloudFormation, Pulumi
**Output**: Cloud resources (databases, compute, storage, APIs)

```
Terraform Creates:
├── Vertex AI Search datastore
├── Cloud Run service
├── Service accounts & IAM roles
├── Cloud Storage buckets
├── BigQuery datasets
└── Secrets and environment variables
```

### How They Work Together

```
┌──────────────────────────────────────────┐
│ 1. Use Agent Scaffold                    │
│    → Generate agent code structure       │
│    → Get Makefile, tests, deployment     │
└─────────────┬────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────┐
│ 2. Use Terraform (included in scaffold)  │
│    → Provision cloud resources           │
│    → Create databases, datastores        │
│    → Setup IAM and secrets               │
└─────────────┬────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────┐
│ 3. Deploy Agent Code                     │
│    → Deploy to Cloud Run or Agent Engine │
│    → Connect to provisioned resources    │
└──────────────────────────────────────────┘
```

**Summary**: Scaffolds generate code, Terraform provisions infrastructure, then you deploy the code to use that infrastructure.

---

## Agent Architecture Patterns

### Pattern 1: Single-Agent ReAct (Reasoning + Acting)

**When to use**: General conversational agents, simple task execution

**Architecture**:
```
User Query
    ↓
Agent (Gemini/Claude)
    ↓
Reasoning Loop:
├─ Think (analyze query)
├─ Act (use tool)
├─ Observe (get result)
└─ Repeat or Answer
    ↓
Final Response
```

**Best for**:
- Chat assistants
- Simple Q&A with tool use
- Single-domain knowledge
- Low complexity workflows

**Scaffold**: `adk_base`, `langgraph_base_react`

**Example Code**:
```python
from google.adk.agents import Agent

# Simple ReAct agent
agent = Agent(
    name="helper_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant with access to tools.",
    tools=[get_weather, search_docs, calculate],
)

response = agent.send_message("What's the weather in SF?")
# Agent thinks → uses get_weather tool → returns answer
```

---

### Pattern 2: RAG Agent (Retrieval-Augmented Generation)

**When to use**: Knowledge-based Q&A, document chatbots

**Architecture**:
```
User Query
    ↓
Generate Embedding
    ↓
Vector/Enterprise Search
    ↓
Retrieve Top-K Documents
    ↓
Agent + Context
    ↓
Grounded Response
```

**Best for**:
- Documentation assistants
- Customer support chatbots
- Legal/medical knowledge systems
- Enterprise search

**Scaffold**: `agentic_rag`

**Example Code**:
```python
from google.adk.agents import Agent
from app.retrievers import get_retriever, get_compressor

# Setup RAG components
retriever = get_retriever(
    data_store_id="my-docs-datastore",
    max_documents=10,
)

compressor = get_compressor()  # Vertex AI Rank

def retrieve_docs(query: str) -> str:
    """Retrieve relevant documents for a query."""
    docs = retriever.invoke(query)
    ranked_docs = compressor.compress_documents(docs, query)
    return format_docs(ranked_docs)

# RAG agent
agent = Agent(
    name="doc_assistant",
    model="gemini-2.5-flash",
    instruction="Answer using the knowledge base. Always cite sources.",
    tools=[retrieve_docs],
)

response = agent.send_message("How do I deploy an agent?")
# Agent uses retrieve_docs → gets context → answers with citations
```

---

### Pattern 3: Multi-Agent Hierarchy (IAM1/IAM2 Pattern)

**When to use**: Complex tasks requiring specialization, enterprise workflows

**Architecture**:
```
User Query
    ↓
IAM1 (Manager Agent)
    ├─ Analyzes query
    ├─ Routes to specialists
    └─ Synthesizes results
         ├─→ IAM2 (Research Specialist)
         ├─→ IAM2 (Code Specialist)
         ├─→ IAM2 (Data Specialist)
         └─→ IAM2 (Slack Specialist)
    ↓
Coordinated Response
```

**Best for**:
- Enterprise automation
- Complex multi-step workflows
- Task delegation
- Specialized capabilities

**Scaffold**: Custom (see IAM1/IAM2 example below)

**Example Code**:
```python
from google.adk.agents import Agent
from app.sub_agents import AGENT_REGISTRY

# Specialist agents
research_agent = Agent(
    name="research_specialist",
    model="gemini-2.5-flash",
    instruction="You are a research specialist. Provide detailed analysis.",
    tools=[retrieve_docs, web_search],
)

code_agent = Agent(
    name="code_specialist",
    model="gemini-2.0-flash",
    instruction="You are a code specialist. Write clean, tested code.",
    tools=[],
)

AGENT_REGISTRY = {
    "research": research_agent,
    "code": code_agent,
}

def route_to_specialist(task_type: str, query: str) -> str:
    """Route task to appropriate specialist."""
    specialist = AGENT_REGISTRY[task_type]
    return specialist.send_message(query)

# Manager agent (IAM1)
manager_agent = Agent(
    name="manager",
    model="gemini-2.0-flash",
    instruction="""You are a manager agent. Delegate tasks to specialists:
    - 'research' for research tasks
    - 'code' for programming tasks
    """,
    tools=[route_to_specialist, retrieve_docs],
)

response = manager_agent.send_message("Research best practices for authentication and write example code")
# Manager delegates to research_specialist → then to code_specialist → synthesizes response
```

---

### Pattern 4: Multi-Agent Collaboration (CrewAI Pattern)

**When to use**: Complex projects requiring peer collaboration

**Architecture**:
```
User Query
    ↓
Crew Manager
    ↓
Task Distribution:
├─→ Engineer Agent (implements)
├─→ QA Agent (validates)
└─→ Designer Agent (optimizes)
    ↓
Sequential or Parallel Processing
    ↓
Collaborative Output
```

**Best for**:
- Code generation + review
- Content creation + editing
- Research + synthesis
- Iterative improvement workflows

**Scaffold**: `crewai_coding_crew`

**Example Code**:
```python
from crewai import Agent, Task, Crew

# Define specialized agents
engineer = Agent(
    role="Senior Engineer",
    goal="Write high-quality code",
    backstory="Expert Python developer with 10 years experience",
    tools=[code_executor, documentation_search],
)

qa_engineer = Agent(
    role="QA Engineer",
    goal="Ensure code quality and catch bugs",
    backstory="Testing specialist focused on edge cases",
    tools=[test_runner, lint_checker],
)

# Define tasks
code_task = Task(
    description="Implement user authentication system",
    agent=engineer,
)

review_task = Task(
    description="Review the authentication code for security issues",
    agent=qa_engineer,
)

# Create crew
crew = Crew(
    agents=[engineer, qa_engineer],
    tasks=[code_task, review_task],
    process="sequential",  # Engineer → QA
)

result = crew.kickoff()
# Engineer writes code → QA reviews → final validated code
```

---

### Pattern 5: Distributed Multi-Agent (A2A Protocol)

**When to use**: Cross-domain coordination, distributed systems

**Architecture**:
```
Engineering Domain
    IAM1 (Engineering) ──────┐
         ↓                    │
    IAM2 specialists          │
                              │ A2A Protocol
Sales Domain                  │ (JSON-RPC 2.0)
    IAM1 (Sales) ─────────────┤
         ↓                    │
    IAM2 specialists          │
                              │
Operations Domain             │
    IAM1 (Operations) ────────┘
         ↓
    IAM2 specialists
```

**Best for**:
- Multi-department coordination
- Distributed agent systems
- Framework-agnostic agents
- Enterprise-scale deployments

**Scaffold**: `adk_a2a_base`

**Example Code**:
```python
from a2a_sdk import A2AClient, Message

def coordinate_with_peer(domain: str, request: str) -> str:
    """Coordinate with peer IAM1 agent via A2A Protocol."""
    peer_url = os.getenv(f"IAM1_{domain.upper()}_URL")
    client = A2AClient(base_url=peer_url)

    message = Message(
        role="user",
        content=[{"type": "text", "text": request}],
    )

    task = client.tasks.create(messages=[message])
    task = client.tasks.wait_until_complete(task.id, timeout=30)

    return task.artifacts[0].parts[0].text

# Agent with A2A coordination
agent = Agent(
    name="engineering_iam1",
    model="gemini-2.0-flash",
    instruction="""You coordinate with peer IAM1s in other domains.
    Available peers: sales, operations, finance, hr, marketing""",
    tools=[coordinate_with_peer, route_to_specialist, retrieve_docs],
)

response = agent.send_message("Get sales forecast from Sales and check if Ops has capacity")
# Coordinates with sales IAM1 → coordinates with ops IAM1 → synthesizes answer
```

---

### Pattern 6: Real-Time Multimodal Agent

**When to use**: Voice/video interactions, real-time streaming

**Architecture**:
```
User (Audio/Video/Text)
    ↓
WebSocket Connection
    ↓
FastAPI Backend
    ↓
Gemini Live API
    ├─ Audio input
    ├─ Video input
    ├─ Text input
    └─ Tool calling
    ↓
Streaming Response
    ↓
React Frontend (Audio/Video/Text)
```

**Best for**:
- Voice assistants
- Video analysis
- Real-time conversations
- Multimodal applications

**Scaffold**: `adk_live`

**Example Code**:
```python
from fastapi import FastAPI, WebSocket
from google.adk.agents import Agent

app = FastAPI()

agent = Agent(
    name="live_assistant",
    model="gemini-2.0-flash-live",
    instruction="You are a helpful voice assistant.",
    tools=[get_weather, search_calendar],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async for message in websocket.iter_text():
        # Process audio/video/text
        response = agent.send_message(message)

        # Stream response back
        await websocket.send_text(response)
```

---

## Using Agent Starter Pack Scaffolds

### Quick Start (5 Minutes)

```bash
# Install
pip install agent-starter-pack

# Create agent from template
agent-starter-pack create my-agent \
  --agent adk_base \
  --deployment-target cloud_run

# Navigate and run
cd my-agent
make install
make playground
```

### Available Templates

| Template | Pattern | Use Case |
|----------|---------|----------|
| `adk_base` | Single-agent ReAct | General conversational agent |
| `adk_a2a_base` | Distributed ReAct | Multi-agent coordination |
| `agentic_rag` | RAG | Knowledge-based Q&A |
| `langgraph_base_react` | Graph-based ReAct | Complex workflows |
| `crewai_coding_crew` | Multi-agent collaboration | Code generation + review |
| `adk_live` | Real-time multimodal | Voice/video agents |

### Customizing Templates

**1. Modify Agent Logic** (`app/agent.py`):
```python
# Edit instruction
instruction = """
You are a customer support agent for Acme Corp.
Help users with billing, technical issues, and account management.
"""

# Add custom tools
def check_order_status(order_id: str) -> str:
    """Check the status of an order."""
    # Your logic here
    return f"Order {order_id} status: Shipped"

agent = Agent(
    name="support_agent",
    model="gemini-2.5-flash",
    instruction=instruction,
    tools=[check_order_status, retrieve_docs],
)
```

**2. Add Data Ingestion** (for RAG):
```bash
# Include RAG pipeline
agent-starter-pack create my-rag-agent \
  --agent agentic_rag \
  --include-data-ingestion \
  --datastore vertex_ai_search

# Add your documents
cp my-docs/* my-rag-agent/data_ingestion/sample_data/

# Ingest
cd my-rag-agent
make setup-dev-env   # Create infrastructure
make data-ingestion  # Index documents
```

**3. Customize Frontend** (Streamlit):
```python
# app/streamlit_app.py
import streamlit as st

st.title("Acme Support Agent")
st.sidebar.image("logo.png")

query = st.chat_input("How can I help you?")
if query:
    response = agent.send_message(query)
    st.chat_message("assistant").write(response)
```

---

## Creating Custom Agent Scaffolds

### DIY Template Structure

```
my-agent-template/
├── .template/
│   └── templateconfig.yaml    # Template metadata
├── cookiecutter.json           # Template variables
├── {{cookiecutter.project_name}}/
│   ├── app/
│   │   ├── agent.py.jinja2    # Templated agent code
│   │   ├── tools.py.jinja2
│   │   └── app_utils/
│   ├── tests/
│   │   ├── unit/
│   │   └── integration/
│   ├── deployment/
│   │   └── terraform/
│   ├── Makefile.jinja2
│   ├── pyproject.toml.jinja2
│   └── README.md.jinja2
└── README.md                   # Template documentation
```

### Example: Custom RAG Template

**cookiecutter.json**:
```json
{
    "project_name": "my-rag-agent",
    "agent_name": "rag_assistant",
    "model": ["gemini-2.5-flash", "gemini-2.0-flash", "claude-sonnet-4-5"],
    "datastore_type": ["vertex_ai_search", "vertex_ai_vector_search"],
    "deployment_target": ["cloud_run", "agent_engine"]
}
```

**agent.py.jinja2**:
```python
from google.adk.agents import Agent
{% if cookiecutter.datastore_type == "vertex_ai_search" %}
from app.retrievers import get_vertex_ai_search_retriever as get_retriever
{% elif cookiecutter.datastore_type == "vertex_ai_vector_search" %}
from app.retrievers import get_vector_search_retriever as get_retriever
{% endif %}

retriever = get_retriever(
    project_id="{{ cookiecutter.project_id }}",
    data_store_id="{{ cookiecutter.project_name }}-datastore",
)

def retrieve_docs(query: str) -> str:
    """Retrieve relevant documents."""
    docs = retriever.invoke(query)
    return format_docs(docs)

agent = Agent(
    name="{{ cookiecutter.agent_name }}",
    model="{{ cookiecutter.model }}",
    instruction="You are a helpful RAG assistant. Always cite sources.",
    tools=[retrieve_docs],
)
```

**Using Your Custom Template**:
```bash
# From GitHub
agent-starter-pack create my-project \
  --agent https://github.com/user/my-agent-template

# From local path
agent-starter-pack create my-project \
  --agent local@./my-agent-template
```

---

## IAM1/IAM2 Agent Scaffold Example

Here's a complete scaffold for the hierarchical IAM1/IAM2 pattern:

### Directory Structure
```
iam1-regional-manager/
├── app/
│   ├── agent.py              # IAM1 (manager)
│   ├── sub_agents.py         # IAM2 (specialists)
│   ├── a2a_tools.py          # Peer coordination
│   ├── retrievers.py         # RAG components
│   └── templates.py
├── deployment/
│   └── terraform/
│       ├── dev/
│       └── main/
├── tests/
│   ├── unit/
│   └── integration/
├── Makefile
└── pyproject.toml
```

### Core Files

**app/agent.py** (IAM1 Manager):
```python
from google.adk.agents import Agent
from app.sub_agents import AGENT_REGISTRY
from app.a2a_tools import coordinate_with_peer
from app.retrievers import retrieve_docs

def route_to_specialist(task_type: str, query: str) -> str:
    """Route to IAM2 specialist (research, code, data, slack)."""
    specialist = AGENT_REGISTRY[task_type]
    return specialist.send_message(query)

instruction = """
You are IAM1 - a Regional Manager AI agent.

YOUR TEAM (IAM2 Specialists):
- research: Deep research, knowledge synthesis
- code: Code generation, debugging
- data: SQL queries, data analysis
- slack: Slack formatting, integrations

PEER IAM1s (Coordinate via A2A):
- engineering, sales, operations, marketing, finance, hr

DECISION FRAMEWORK:
1. Simple questions → Answer directly
2. Knowledge questions → Use retrieve_docs
3. Cross-domain info → Coordinate with peer IAM1
4. Complex specialized tasks → Route to IAM2 specialist
"""

iam1_agent = Agent(
    name="regional_manager",
    model="gemini-2.0-flash",
    instruction=instruction,
    tools=[retrieve_docs, route_to_specialist, coordinate_with_peer],
)
```

**app/sub_agents.py** (IAM2 Specialists):
```python
from google.adk.agents import Agent

research_agent = Agent(
    name="research_iam2",
    model="gemini-2.5-flash",
    instruction="""You are a Research Specialist (IAM2).
    Report to IAM1 with thorough research and citations.""",
    tools=[retrieve_docs],
)

code_agent = Agent(
    name="code_iam2",
    model="gemini-2.0-flash",
    instruction="""You are a Code Specialist (IAM2).
    Write clean, tested code and explain your approach.""",
    tools=[],
)

data_agent = Agent(
    name="data_iam2",
    model="gemini-2.5-flash",
    instruction="""You are a Data Specialist (IAM2).
    Write SQL queries and provide data insights.""",
    tools=[],
)

slack_agent = Agent(
    name="slack_iam2",
    model="gemini-2.0-flash",
    instruction="""You are a Slack Specialist (IAM2).
    Format messages for Slack with proper markdown.""",
    tools=[],
)

AGENT_REGISTRY = {
    "research": research_agent,
    "code": code_agent,
    "data": data_agent,
    "slack": slack_agent,
}
```

**app/a2a_tools.py** (Peer Coordination):
```python
import os
from a2a_sdk import A2AClient, Message

PEER_REGISTRY = {
    "engineering": os.getenv("IAM1_ENGINEERING_URL", ""),
    "sales": os.getenv("IAM1_SALES_URL", ""),
    "operations": os.getenv("IAM1_OPERATIONS_URL", ""),
    "marketing": os.getenv("IAM1_MARKETING_URL", ""),
    "finance": os.getenv("IAM1_FINANCE_URL", ""),
    "hr": os.getenv("IAM1_HR_URL", ""),
}

def coordinate_with_peer(domain: str, request: str) -> str:
    """Coordinate with peer IAM1 (not subordinate, peer)."""
    peer_url = PEER_REGISTRY.get(domain)
    if not peer_url:
        return f"Peer IAM1 '{domain}' not configured"

    client = A2AClient(base_url=peer_url)
    message = Message(role="user", content=[{"type": "text", "text": request}])

    task = client.tasks.create(messages=[message])
    task = client.tasks.wait_until_complete(task.id, timeout=30)

    if task.status == "completed":
        return task.artifacts[0].parts[0].text
    else:
        return f"Peer coordination failed: {task.status}"
```

**Makefile**:
```makefile
.PHONY: install playground deploy

install:
	uv pip install -r requirements.txt

playground:
	streamlit run app/streamlit_app.py

deploy:
	gcloud run deploy regional-manager \
		--source . \
		--region us-central1 \
		--set-env-vars IAM1_ENGINEERING_URL=$(ENGINEERING_URL)

setup-dev-env:
	cd deployment/terraform/dev && terraform apply
```

---

## Best Practices for Agent Scaffolding

### 1. Separation of Concerns

```
✅ GOOD - Modular structure
app/
├── agent.py           # Agent definition only
├── tools.py           # Tool implementations
├── retrievers.py      # RAG logic
└── models.py          # Data models

❌ BAD - Everything in one file
app/
└── agent.py           # 2000 lines of mixed concerns
```

### 2. Configuration Management

```python
✅ GOOD - Environment variables
import os

PROJECT_ID = os.getenv("PROJECT_ID")
DATA_STORE_ID = os.getenv("DATA_STORE_ID", "default-datastore")
MODEL = os.getenv("MODEL", "gemini-2.5-flash")

❌ BAD - Hardcoded values
PROJECT_ID = "my-project-123"  # Never do this!
```

### 3. Error Handling

```python
✅ GOOD - Graceful degradation
def retrieve_docs(query: str) -> str:
    try:
        docs = retriever.invoke(query)
        return format_docs(docs)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return "Unable to retrieve documents. Using general knowledge."

❌ BAD - Unhandled errors
def retrieve_docs(query: str) -> str:
    docs = retriever.invoke(query)  # Crashes if error
    return format_docs(docs)
```

### 4. Testing Structure

```
✅ GOOD - Comprehensive tests
tests/
├── unit/
│   ├── test_agent.py         # Mock all external deps
│   ├── test_tools.py
│   └── test_retrievers.py
├── integration/
│   └── test_agent.py         # Test with real GCP
└── load_test/
    └── locustfile.py         # Performance testing

❌ BAD - No tests or only integration
tests/
└── test_everything.py        # Slow, brittle, hard to debug
```

### 5. Makefile for Consistency

```makefile
✅ GOOD - Standard commands
.PHONY: install test lint deploy

install:
	uv pip install -r requirements.txt

test:
	pytest tests/unit tests/integration

lint:
	ruff check app/
	mypy app/

deploy:
	make test && make lint && gcloud run deploy ...
```

---

## Deployment Patterns

### Pattern 1: Cloud Run (Containerized)

**When**: General-purpose agents, REST APIs, HTTP endpoints

```yaml
# deployment/cloud_run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: my-agent
spec:
  template:
    spec:
      containers:
      - image: gcr.io/project/my-agent:latest
        env:
        - name: PROJECT_ID
          value: "my-project"
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
```

**Scaffold**: Includes Dockerfile, Cloud Run config

### Pattern 2: Vertex AI Agent Engine

**When**: LLM-native agents, managed orchestration

```python
# deployment/agent_engine_deploy.py
from google.cloud.aiplatform import reasoning_engines

agent_engine = reasoning_engines.ReasoningEngine.create(
    reasoning_engine=agent,
    requirements=["google-cloud-aiplatform", "langchain"],
    display_name="My Agent",
)

print(f"Deployed: {agent_engine.resource_name}")
```

**Scaffold**: Includes agent_engine_app.py, deployment script

### Pattern 3: Hybrid (Cloud Run + Agent Engine)

**When**: Complex agents with custom backends

```
Cloud Run (Custom Logic)
    ↓
Calls Agent Engine (LLM)
    ↓
Returns Response
```

---

## Scaffold Comparison

| Feature | Agent Starter Pack | Custom Template | DIY (No Scaffold) |
|---------|-------------------|-----------------|-------------------|
| **Setup Time** | 5 minutes | 30 minutes | 2+ hours |
| **CI/CD** | ✅ Automated | ⚙️ Manual | ❌ None |
| **Terraform** | ✅ Included | ⚙️ Optional | ❌ None |
| **Testing** | ✅ Full suite | ⚙️ Partial | ❌ None |
| **Best Practices** | ✅ Enforced | ⚙️ Up to you | ❌ None |
| **Observability** | ✅ Built-in | ⚙️ Optional | ❌ None |
| **Flexibility** | ⚙️ Moderate | ✅ High | ✅ Complete |
| **Learning Curve** | Easy | Moderate | Hard |

**Recommendation**:
- **Start with Agent Starter Pack** for production projects
- **Use custom templates** for organization-specific patterns
- **DIY only** for experimental/research projects

---

## Summary

### Key Takeaways

1. **Scaffolds ≠ Infrastructure**
   - Scaffolds = Code templates (agent logic, tools, tests)
   - Infrastructure = Cloud resources (databases, compute, storage)
   - Both work together for production deployment

2. **Choose the Right Pattern**
   - Single-agent: General tasks
   - RAG: Knowledge-based Q&A
   - Multi-agent hierarchy: Complex enterprises
   - Collaboration: Peer review workflows
   - Distributed A2A: Cross-domain coordination

3. **Use Existing Scaffolds**
   - Agent Starter Pack: 6 production-ready templates
   - Saves hours/days of setup time
   - Includes CI/CD, Terraform, testing

4. **Customize for Your Needs**
   - Modify agent instructions
   - Add custom tools
   - Integrate your data sources
   - Brand your frontend

5. **Follow Best Practices**
   - Modular code structure
   - Environment-based configuration
   - Comprehensive testing
   - Observability from day one

### Next Steps

1. **Quick Start**: Use Agent Starter Pack
   ```bash
   uvx agent-starter-pack create my-agent -a adk_base
   ```

2. **Learn Patterns**: Study the 6 architecture patterns

3. **Build Custom**: Create templates for your organization

4. **Deploy**: Use included Terraform and CI/CD

**Resources**:
- Agent Starter Pack: https://github.com/GoogleCloudPlatform/agent-starter-pack
- ADK Documentation: https://google.github.io/adk-docs/
- A2A Protocol: https://a2aprotocol.org/

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**IAM1 Regional Manager** is a hierarchical multi-agent system built on Google ADK and Vertex AI. It implements a "Regional Manager" pattern where:

- **IAM1** = Regional Manager agent (sovereign within domain, orchestrates tasks)
- **IAM2** = Specialist agents (subordinates who execute specific tasks)
- **A2A Protocol** = Peer-to-peer coordination between IAM1 instances

Think of it as an organizational hierarchy: IAM1 can **command** IAM2 subordinates but can only **coordinate** with peer IAM1s.

## Essential Commands

### Development
```bash
# Install dependencies (uses uv package manager)
make install

# Launch local development playground
make playground
# Opens web UI at http://localhost:8501
# IMPORTANT: Select the 'app' folder to interact with your agent

# Run tests (unit + integration)
make test

# Run linting (codespell, ruff, mypy)
make lint
```

### Deployment
```bash
# Deploy to Google Cloud Vertex AI
make deploy

# Set up infrastructure (Terraform)
make setup-dev-env

# Ingest data for RAG capabilities
make data-ingestion
```

### Testing
```bash
# Run unit tests only
uv run pytest tests/unit

# Run integration tests only
uv run pytest tests/integration

# Run specific test file
uv run pytest tests/unit/test_dummy.py -v
```

## Architecture Overview

### Agent Hierarchy

```
IAM1 (Regional Manager)
├── Manages: IAM2 specialist agents via route_to_agent()
├── Coordinates: Peer IAM1s via A2A Protocol
└── Grounds: Decisions in RAG knowledge via retrieve_docs()

IAM2 Specialists (Subordinates)
├── Research Agent (research_iam2) - Deep research, knowledge synthesis
├── Code Agent (code_iam2) - Code generation, debugging
├── Data Agent (data_iam2) - BigQuery queries, data analysis
└── Slack Agent (slack_iam2) - Slack formatting, channel management
```

### Key Files & Responsibilities

**Core Agent Implementation:**
- `app/agent.py` - Main IAM1 orchestrator agent definition
- `app/sub_agents.py` - IAM2 specialist agent definitions and AGENT_REGISTRY
- `app/a2a_tools.py` - A2A Protocol integration for peer IAM1 coordination
- `app/agent_engine_app.py` - Entry point for Vertex AI Agent Engine deployment

**Configuration & Utilities:**
- `app/iam1_config.py` - Business model config, deployment scenarios, hierarchy rules
- `app/agent_card.py` - Agent metadata (name, version, capabilities)
- `app/retrievers.py` - Vertex AI Search RAG retrieval logic
- `app/templates.py` - Document formatting for RAG responses
- `app/app_utils/deploy.py` - Deployment utilities
- `app/app_utils/tracing.py` - Observability and telemetry

**Infrastructure:**
- `deployment/terraform/` - Infrastructure as Code for GCP resources
- `data_ingestion/` - Pipeline for uploading documents to Vertex AI Search

### Decision Framework

The IAM1 agent follows this routing logic (defined in `app/agent.py` instruction):

1. **Simple questions** (greetings, basic info) → Answer directly
2. **Knowledge questions** → Use `retrieve_docs()` tool (RAG)
3. **Cross-domain information** → Use `coordinate_with_peer_iam1()` (A2A)
4. **Specialized tasks** → Use `route_to_agent()` to delegate to IAM2
5. **Multi-step tasks** → Coordinate multiple agents

### IAM2 Agent Types (sub_agents.py)

When adding/modifying IAM2 agents:

- **research**: Uses `retrieve_docs` tool, provides research reports with citations
- **code**: No tools yet (future: code execution), generates clean code with explanations
- **data**: No tools yet (future: BigQuery), writes SQL queries and analysis
- **slack**: No tools yet (future: Slack API), formats messages for Slack

Registry: `AGENT_REGISTRY` dict in `app/sub_agents.py`

### A2A Peer Coordination (a2a_tools.py)

Peer IAM1 domains configured via environment variables:
- `IAM1_ENGINEERING_URL` - Engineering domain
- `IAM1_SALES_URL` - Sales domain
- `IAM1_OPERATIONS_URL` - Operations domain
- `IAM1_MARKETING_URL` - Marketing domain
- `IAM1_FINANCE_URL` - Finance domain
- `IAM1_HR_URL` - HR domain

Authentication: `IAM1_A2A_API_KEY` environment variable

**Important**: IAM1 peers are **equals**, not subordinates. They coordinate but don't command each other.

## RAG Knowledge Grounding

### How RAG Works

1. Documents uploaded to Google Cloud Storage bucket
2. Data ingestion pipeline (`make data-ingestion`) indexes to Vertex AI Search
3. IAM1 uses `retrieve_docs()` tool to query knowledge base
4. Results re-ranked using Vertex AI Rank for relevance

### Configuring RAG

Environment variables:
- `DATA_STORE_ID` - Vertex AI Search data store ID (default: "bob-vertex-agent-datastore")
- `DATA_STORE_REGION` - Region for data store (default: "us")

Both IAM1 and IAM2-research agents have access to `retrieve_docs()`.

## Business Model Context

This project is designed for **multi-tenant deployments**:

- Each client gets their own GCP project + IAM1 instance
- Knowledge bases are **never shared** between clients
- Pricing: Per-IAM1 subscription + per-IAM2 add-ons
- Deployment pattern: `{client}-{domain}-iam1` (e.g., "acme-sales-iam1")

See `app/iam1_config.py` for business model definitions and deployment scenarios.

See `DEPLOYMENT_GUIDE.md` for detailed deployment workflows.

## Key Configuration

### Python Project
- **Package Manager**: `uv` (modern, fast pip alternative)
- **Python Version**: 3.10 - 3.12
- **Framework**: Google ADK (Agent Development Kit)
- **LLM Models**:
  - IAM1: `gemini-2.0-flash` (orchestrator)
  - IAM2-research/data: `gemini-2.5-flash` (specialists)
  - IAM2-code/slack: `gemini-2.0-flash` (specialists)

### Code Quality
- **Linter**: Ruff (replaces flake8, black, isort)
- **Type Checker**: mypy (strict mode enabled)
- **Spell Check**: codespell

Line length: 88 characters

### Testing
- **Framework**: pytest with pytest-asyncio
- Test structure:
  - `tests/unit/` - Unit tests (fast, mocked)
  - `tests/integration/` - Integration tests (require GCP access)
  - `tests/load_test/` - Load testing utilities

## Important Patterns

### Adding a New IAM2 Agent

1. Define agent in `app/sub_agents.py`:
   ```python
   new_agent = Agent(
       name="new_iam2",
       model="gemini-2.5-flash",
       instruction="You are a [SPECIALTY] Specialist...",
       tools=[...],
   )
   ```

2. Add to `AGENT_REGISTRY`:
   ```python
   AGENT_REGISTRY = {
       ...
       "new_type": new_agent,
   }
   ```

3. Update IAM1 instruction in `app/agent.py` to document the new specialist

### Adding Tools to Agents

IAM1 tools are defined in `app/agent.py`:
```python
root_agent = Agent(
    ...
    tools=[retrieve_docs, route_to_agent, coordinate_with_peer_iam1],
)
```

IAM2 tools are defined in their respective agent definitions in `app/sub_agents.py`.

Tools must follow Google ADK function signature patterns (type hints required).

### Modifying Agent Instructions

Agent behavior is controlled by the `instruction` parameter. These are detailed system prompts that define:
- Identity and role
- Capabilities and tools available
- Decision-making framework
- Output format expectations
- Hierarchy relationships (IAM1 vs IAM2)

See `app/agent.py` (IAM1 instruction) and `app/sub_agents.py` (IAM2 instructions) for examples.

## Deployment Notes

### Local Development
- Use `make playground` for interactive testing
- Playground requires selecting the `app` folder from UI
- Changes to agent code require playground reload

### Google Cloud Deployment
- Deploys to Vertex AI Agent Engine (managed service)
- Requires GCP project with Vertex AI APIs enabled
- Deployment uses `gcloud` CLI and ADK deployment utilities
- Environment variables set via `--set-env-vars` in deploy command

### Terraform Infrastructure
- Located in `deployment/terraform/dev/`
- Creates: Vertex AI Search, BigQuery, Cloud Storage, IAM roles
- Variables in `deployment/terraform/dev/vars/env.tfvars`

## Documentation

- `README.md` - Marketing overview, features, quick start
- `DEPLOYMENT_GUIDE.md` - Detailed deployment scenarios and business model
- `GEMINI.md` - Gemini-specific implementation notes
- `claudes-docs/` - Detailed design documents:
  - `000-INDEX.md` - Index of all documents
  - `001-OD-CONF-github-template-setup.md` - GitHub template configuration
  - `002-AT-ADEC-iam1-fine-tuning.md` - IAM1 agent tuning decisions
  - `003-RA-ANLY-a2a-integration.md` - A2A Protocol integration analysis
  - `004-PP-PLAN-architecture-brainstorm.md` - Architecture planning
  - `005-OD-DEPL-github-pages.md` - GitHub Pages documentation
  - `006-OD-DEPL-template-deployment.md` - Template deployment guide
  - `007-RA-REPT-industry-examples.md` - Industry use cases

Interactive documentation available at: https://jeremylongshore.github.io/iam1-intent-agent-model-vertex-ai/

## Common Gotchas

1. **uv not installed**: Run `make install` which auto-installs uv if needed
2. **GCP authentication**: Ensure `gcloud auth application-default login` is run before deployment
3. **Environment variables**: IAM1 needs `PROJECT_ID` set for GCP operations
4. **A2A peers not configured**: Peer coordination fails gracefully with helpful error messages if URLs not set
5. **Data store not created**: Run `make setup-dev-env` before first deployment
6. **Mypy strict mode**: All functions need type hints, use `# type: ignore` sparingly
7. **Pytest asyncio**: Use `pytest-asyncio` fixtures for async tests

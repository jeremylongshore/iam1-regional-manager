# Multi-Agent Systems with Claude on Vertex AI
**Version:** 1.0
**Last Updated:** 2025-11-12
**Topics:** A2A Protocol, Claude, Vertex AI Agent Engine, MCP Tools, Multi-Agent Orchestration

---

## Overview

This guide demonstrates how to build sophisticated multi-agent systems that combine **Claude** (Anthropic) with **Gemini** (Google) models on Vertex AI, using the Agent2Agent (A2A) protocol for standardized communication and the Model Context Protocol (MCP) for tool integration.

**What You'll Learn:**
- Integrate Claude models with Vertex AI Agent Engine
- Build agents using different frameworks (Pydantic AI + Google ADK)
- Create custom tools with MCP (Model Context Protocol)
- Orchestrate multiple specialized agents via A2A protocol
- Deploy production multi-agent systems to Google Cloud

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Framework Integration](#framework-integration)
3. [MCP Tool Creation](#mcp-tool-creation)
4. [Building Agents](#building-agents)
5. [A2A Communication](#a2a-communication)
6. [Orchestration Patterns](#orchestration-patterns)
7. [Deployment to Agent Engine](#deployment-to-agent-engine)
8. [Complete Example](#complete-example)

---

## Architecture Overview

### Multi-Framework Architecture

The power of A2A protocol is that agents built with different frameworks can communicate seamlessly:

```
┌─────────────────────────────────────────────┐
│         Orchestrator Agent (ADK)             │
│              Gemini 2.5 Flash                │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐       ┌──────────────┐
│ Agent A      │       │ Agent B      │
│ Pydantic AI  │       │ Google ADK   │
│ + Claude     │       │ + Claude     │
│ + MCP Tools  │       │ + MCP Tools  │
└──────────────┘       └──────────────┘
```

**Key Components:**
1. **Agent Frameworks**: Pydantic AI, Google ADK, LangChain, etc.
2. **LLM Models**: Claude (via LiteLLM), Gemini (native)
3. **Tool Protocols**: MCP for custom capabilities
4. **Communication**: A2A protocol for agent-to-agent interaction
5. **Deployment**: Vertex AI Agent Engine for managed infrastructure

---

## Framework Integration

### Using Claude with Pydantic AI

Pydantic AI provides a clean interface for building agents with Claude on Vertex AI:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
import vertexai

# Initialize Vertex AI
project_id = "your-project-id"
location = "us-central1"
vertexai.init(project=project_id, location=location)

# Configure Claude model through Google provider
provider = GoogleProvider(vertexai=True)
model = GoogleModel("gemini-2.5-flash", provider=provider)  # or Claude when available

# Create agent
agent = Agent(
    model=model,
    system_prompt="You are a helpful AI assistant specialized in...",
    tools=[],  # MCP tools go here
    retries=2,
)

# Use the agent
result = await agent.run("Analyze this market data...")
print(result.output)
```

---

### Using Claude with Google ADK via LiteLLM

Google ADK can route to Claude using LiteLLM:

```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm, litellm
import os

# Configure LiteLLM for Vertex AI routing
litellm.vertex_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
litellm.vertex_location = "global"

# Create ADK agent with Claude
agent = LlmAgent(
    name="claude_specialist",
    model=LiteLlm("vertex_ai/claude-sonnet-4-5@20250929"),
    description="Specialist agent using Claude Sonnet 4.5",
    instruction="You are an expert analyst...",
    tools=[],  # Tools go here
)
```

**Model Options via LiteLLM:**
- `vertex_ai/claude-sonnet-4-5@20250929` - Claude Sonnet 4.5
- `vertex_ai/claude-opus-4@20250514` - Claude Opus 4
- `vertex_ai/claude-haiku-4@20250514` - Claude Haiku 4

---

## MCP Tool Creation

MCP (Model Context Protocol) extends agent capabilities with custom tools. Here's how to create and package them:

### Creating MCP Tools with FastMCP

```python
from mcp.server.fastmcp import FastMCP
import numpy as np

# Initialize MCP server
mcp = FastMCP("my-agent-tools")

@mcp.tool()
async def analyze_data(symbol: str, period: int = 30) -> str:
    """Analyze financial data for a given symbol.

    Args:
        symbol: Stock symbol to analyze (e.g., NVDA, AAPL)
        period: Number of days to analyze (default: 30)

    Returns:
        Analysis report as formatted string
    """
    # Your analysis logic here
    analysis = f"""
ANALYSIS FOR {symbol}
{'='*40}
Period: {period} days
Trend: BULLISH
Confidence: 85%

Key Metrics:
- Momentum Score: 78/100
- Risk Level: MEDIUM
- Entry Quality: HIGH
"""
    return analysis

@mcp.tool()
async def get_market_sentiment(symbol: str) -> str:
    """Retrieve market sentiment for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Sentiment analysis report
    """
    sentiment_score = np.random.uniform(40, 80)

    return f"""
SENTIMENT ANALYSIS: {symbol}
Sentiment Score: {sentiment_score:.1f}/100
Rating: {"POSITIVE" if sentiment_score > 60 else "NEUTRAL"}

Factors:
- Social media: Positive mentions increasing
- News coverage: Mostly favorable
- Analyst ratings: 70% buy recommendations
"""
```

---

### Packaging MCP Tools as Python Module

Create a proper package structure for deployment:

```bash
mcp_tools/
├── __init__.py
├── market_tools.py      # Tool definitions
└── mcp_server.py        # Server entry point
```

**`mcp_tools/__init__.py`:**
```python
"""MCP Tools package for trading agents."""
```

**`mcp_tools/market_tools.py`:**
```python
"""Market analysis MCP tools."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("market-tools")

@mcp.tool()
async def analyze_data(symbol: str) -> str:
    # Tool implementation
    pass
```

**`mcp_tools/mcp_server.py`:**
```python
"""MCP Server entry point."""

from market_tools import mcp

if __name__ == "__main__":
    # Run server with STDIO transport
    mcp.run(transport="stdio")
```

---

### Using MCP Tools in Pydantic AI Agents

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

# Configure MCP server connection
mcp_server = MCPServerStdio(
    "python",
    args=["mcp_tools/mcp_server.py"],
    timeout=60
)

# Create agent with MCP tools
agent = Agent(
    model=model,
    system_prompt="You are a market analyst...",
    toolsets=[mcp_server],  # MCP tools as toolset
    retries=3,
)
```

---

### Using MCP Tools in Google ADK Agents

```python
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioServerParameters,
)

# Create ADK agent with MCP tools
agent = LlmAgent(
    name="market_analyst",
    model="gemini-2.5-flash",
    instruction="You are a market analyst...",
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    "python",
                    args=["mcp_tools/mcp_server.py"],
                    timeout=60,
                ),
            ),
        )
    ],
)
```

---

## Building Agents

### Example: Risk Analysis Agent (Pydantic AI + Claude)

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.mcp import MCPServerStdio

# System prompt defining agent personality
risk_analyst_prompt = """You are a cautious risk analyst focused on
identifying potential downside catalysts, warning signals, and protective
strategies. You prioritize capital preservation.

When analyzing risks:
1. Use your tools to gather comprehensive data
2. Identify specific risk factors with severity levels
3. Provide actionable recommendations
4. Cite data sources and calculations

Always be thorough but concise in your analysis."""

# Configure Gemini/Claude model
provider = GoogleProvider(vertexai=True)
model = GoogleModel("gemini-2.5-flash", provider=provider)

# Configure MCP tools
mcp_server = MCPServerStdio(
    "python",
    args=["mcp_tools/risk_tools_server.py"],
    timeout=60
)

# Create Risk Analysis Agent
risk_agent = Agent(
    model=model,
    system_prompt=risk_analyst_prompt,
    toolsets=[mcp_server],
    retries=2,
)

# Test the agent
async def test_agent():
    result = await risk_agent.run("Analyze the risks for NVDA stock")
    print(result.output)

await test_agent()
```

---

### Example: Opportunity Agent (ADK + Claude via LiteLLM)

```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm, litellm
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioServerParameters,
)

# Configure LiteLLM for Claude
litellm.vertex_project = "your-project-id"
litellm.vertex_location = "global"

# Create Opportunity Analysis Agent
opportunity_agent = LlmAgent(
    name="opportunity_analyst",
    model=LiteLlm("vertex_ai/claude-sonnet-4-5@20250929"),
    description="Optimistic analyst focused on growth opportunities",
    instruction="""You are an optimistic market analyst focused on identifying
    growth opportunities, bullish patterns, and upside catalysts.

    Use available tools to:
    - Identify breakout patterns
    - Screen for momentum stocks
    - Detect optimal entry points

    Present findings with specific price targets and confidence levels.""",
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    "python",
                    args=["mcp_tools/opportunity_tools_server.py"],
                    timeout=60,
                ),
            ),
        )
    ],
)
```

---

## A2A Communication

### Creating Agent Cards for Discovery

Agent Cards advertise capabilities via A2A protocol:

```python
from a2a.types import AgentSkill
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card

# Define agent skills
risk_skills = [
    AgentSkill(
        id="risk_analysis",
        name="Risk Factor Scanner",
        description="Identifies potential downside catalysts and risk factors",
        tags=["risk-analysis", "market-analysis"],
        examples=[
            "What are the key risks for NVDA?",
            "Analyze downside catalysts for tech stocks",
        ],
    ),
    AgentSkill(
        id="divergence_detection",
        name="Divergence Detection",
        description="Finds bearish divergences and technical weakness",
        tags=["technical-analysis", "divergence"],
        examples=[
            "Find bearish divergences in AAPL",
        ],
    ),
]

# Create agent card
agent_card = create_agent_card(
    agent_name="Risk Analyst (Claude + MCP)",
    description="Cautious risk analyst using Claude for analysis",
    skills=risk_skills,
)
```

---

### Creating A2A Agent Executors

Bridge agents with A2A protocol using executors:

```python
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError

class RiskAgentExecutor(AgentExecutor):
    """A2A executor for Risk Analysis Agent."""

    def __init__(self):
        # Lazy initialization to avoid pickling issues
        self.agent = None

    def _init_agent(self):
        """Initialize agent when first needed on deployment."""
        if self.agent is None:
            # Import and create agent here
            # This happens on Agent Engine, not during packaging
            from pydantic_ai import Agent
            from pydantic_ai.mcp import MCPServerStdio
            # ... agent creation code ...
            self.agent = Agent(...)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """Execute agent analysis via A2A protocol."""
        # Initialize agent if needed
        if self.agent is None:
            self._init_agent()

        # Extract user query
        query = context.get_user_input()
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # Submit task
        if not context.current_task:
            await updater.submit()

        # Mark as working
        await updater.start_work()

        try:
            # Update status
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message("Analyzing risks...")
            )

            # Run agent
            result = await self.agent.run(query)

            # Format response
            response = f"""
RISK ANALYSIS
{'='*50}

{result.output}

Analysis completed
"""

            # Add artifact and complete
            await updater.add_artifact(
                [TextPart(text=response)],
                name="risk_analysis"
            )
            await updater.complete()

        except Exception as e:
            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"Analysis failed: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancellation not supported."""
        raise ServerError(error=UnsupportedOperationError())
```

---

## Orchestration Patterns

### Creating Multi-Agent Orchestrator

Coordinate multiple specialized agents:

```python
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

# Create remote agent proxies
remote_risk_agent = RemoteA2aAgent(
    name="risk_analyst",
    description="Analyzes risks and warning signals",
    agent_card="https://agent-endpoint/a2a/v1/card",
    httpx_client=authenticated_client,  # With Google Cloud auth
)

remote_opportunity_agent = RemoteA2aAgent(
    name="opportunity_analyst",
    description="Identifies growth opportunities",
    agent_card="https://agent-endpoint/a2a/v1/card",
    httpx_client=authenticated_client,
)

# Create orchestrator
orchestrator = LlmAgent(
    name="strategy_orchestrator",
    model="gemini-2.5-flash",
    instruction="""You are a trading strategy orchestrator that coordinates
    specialized agents to provide balanced market analysis.

    Workflow:
    1. Analyze user query to determine which agents to consult
    2. Invoke appropriate specialist agents
    3. Synthesize responses into coherent recommendation
    4. Provide balanced perspective considering both risks and opportunities

    Always consult both risk and opportunity analysts for major decisions.""",
    tools=[
        AgentTool(agent=remote_risk_agent),
        AgentTool(agent=remote_opportunity_agent),
    ],
)

# Create runner for orchestrator
runner = Runner(
    app_name=orchestrator.name,
    agent=orchestrator,
    session_service=InMemorySessionService(),
)

# Execute coordinated analysis
async def get_trading_recommendation(query: str):
    """Get balanced recommendation from multiple agents."""

    # Create session
    session = await runner.session_service.create_session(
        app_name=orchestrator.name,
        user_id="user_123",
        session_id="session_001",
    )

    # Format query
    content = types.Content(
        role="user",
        parts=[types.Part(text=query)]
    )

    # Run orchestrator
    final_response = None
    async for event in runner.run_async(
        session_id=session.id,
        user_id="user_123",
        new_message=content
    ):
        if event.is_final_response():
            final_response = event
            break

    # Extract response
    if final_response and final_response.content:
        response_text = " ".join(
            part.text for part in final_response.content.parts
            if hasattr(part, "text") and part.text
        )
        return response_text

    return "No response generated"

# Use the orchestrator
recommendation = await get_trading_recommendation(
    "Should I buy NVDA stock? Provide balanced analysis."
)
print(recommendation)
```

---

## Deployment to Agent Engine

### Deploying Pydantic AI Agent with Claude

```python
from vertexai.preview.reasoning_engines import A2aAgent
import vertexai

# Initialize client
client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

# Create A2A agent wrapper
a2a_agent = A2aAgent(
    agent_card=agent_card,
    agent_executor_builder=RiskAgentExecutor  # Your executor class
)

# Deploy to Agent Engine
deployed_agent = client.agent_engines.create(
    agent=a2a_agent,
    config={
        "display_name": "Risk Analyst (Claude + MCP)",
        "description": "Risk analysis agent using Claude",
        "requirements": [
            "a2a-sdk>=0.3.4",
            "google-cloud-aiplatform[agent_engines,adk]>=1.120.0",
            "fastmcp>=0.1.0",
            "pydantic>=2.0.0",
            "pydantic-ai>=0.1.0",
            "numpy>=1.24.0",
        ],
        "extra_packages": ["mcp_tools"],  # Your MCP tools package
        "staging_bucket": "gs://your-bucket",
    },
)

print(f"Deployed: {deployed_agent.api_resource.name}")
```

---

### Deploying ADK Agent with Claude (via LiteLLM)

```python
# Create A2A agent wrapper
a2a_opportunity_agent = A2aAgent(
    agent_card=opportunity_agent_card,
    agent_executor_builder=OpportunityAgentExecutor
)

# Deploy to Agent Engine
deployed_opportunity = client.agent_engines.create(
    agent=a2a_opportunity_agent,
    config={
        "display_name": "Opportunity Analyst (Claude + MCP)",
        "description": "Opportunity analysis using Claude via LiteLLM",
        "requirements": [
            "a2a-sdk>=0.3.4",
            "google-cloud-aiplatform[agent_engines,adk]>=1.120.0",
            "fastmcp>=0.1.0",
            "litellm>=1.0.0",  # For Claude routing
            "numpy>=1.24.0",
        ],
        "extra_packages": ["mcp_tools"],
        "staging_bucket": "gs://your-bucket",
    },
)
```

---

### Accessing Deployed Agents via A2A

```python
import httpx
from google.auth import default
from google.auth.transport.requests import Request as AuthRequest

# Create authenticated client
class GoogleAuth(httpx.Auth):
    """Google Cloud authentication for httpx."""

    def __init__(self):
        self.credentials, _ = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.auth_request = AuthRequest()

    def auth_flow(self, request: httpx.Request):
        if not self.credentials.valid:
            self.credentials.refresh(self.auth_request)

        request.headers["Authorization"] = f"Bearer {self.credentials.token}"
        yield request

# Create authenticated client
auth_client = httpx.AsyncClient(
    timeout=120,
    auth=GoogleAuth(),
)

# Build agent endpoint
api_endpoint = f"https://{LOCATION}-aiplatform.googleapis.com"
agent_resource_name = deployed_agent.api_resource.name
agent_endpoint = f"{api_endpoint}/v1beta1/{agent_resource_name}/a2a"

# Create remote proxy
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

remote_agent = RemoteA2aAgent(
    name="deployed_risk_analyst",
    description="Production risk analysis agent",
    agent_card=f"{agent_endpoint}/v1/card",
    httpx_client=auth_client,
)

# Use the deployed agent
result = await remote_agent.send_message("Analyze NVDA risks")
print(result)
```

---

## Complete Example

### Trading Analysis Multi-Agent System

Here's a complete example combining everything:

```python
import asyncio
import vertexai
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool

# Configuration
PROJECT_ID = "your-project-id"
LOCATION = "us-central1"

# Initialize
vertexai.init(project=PROJECT_ID, location=LOCATION)
client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

# Step 1: Deploy specialized agents (assume already deployed)
risk_agent_endpoint = "https://...../risk-agent/a2a"
opportunity_agent_endpoint = "https://...../opportunity-agent/a2a"

# Step 2: Create authenticated client
auth_client = httpx.AsyncClient(timeout=120, auth=GoogleAuth())

# Step 3: Create remote proxies
remote_risk = RemoteA2aAgent(
    name="risk_analyst",
    description="Risk analysis specialist",
    agent_card=f"{risk_agent_endpoint}/v1/card",
    httpx_client=auth_client,
)

remote_opportunity = RemoteA2aAgent(
    name="opportunity_analyst",
    description="Opportunity analysis specialist",
    agent_card=f"{opportunity_agent_endpoint}/v1/card",
    httpx_client=auth_client,
)

# Step 4: Create orchestrator
orchestrator = LlmAgent(
    name="trading_orchestrator",
    model="gemini-2.5-flash",
    instruction="""Coordinate risk and opportunity analysts to provide
    balanced trading recommendations.""",
    tools=[
        AgentTool(agent=remote_risk),
        AgentTool(agent=remote_opportunity),
    ],
)

# Step 5: Create runner
runner = Runner(
    app_name=orchestrator.name,
    agent=orchestrator,
    session_service=InMemorySessionService(),
)

# Step 6: Execute analysis
async def analyze_stock(symbol: str):
    """Get comprehensive stock analysis."""

    session = await runner.session_service.create_session(
        app_name=orchestrator.name,
        user_id="trader_001",
        session_id=f"analysis_{symbol}",
    )

    query = f"Provide comprehensive analysis for {symbol}. Include both risks and opportunities."

    content = types.Content(
        role="user",
        parts=[types.Part(text=query)]
    )

    async for event in runner.run_async(
        session_id=session.id,
        user_id="trader_001",
        new_message=content
    ):
        if event.is_final_response():
            response = " ".join(
                part.text for part in event.content.parts
                if hasattr(part, "text") and part.text
            )
            return response

    return "Analysis failed"

# Run the analysis
result = await analyze_stock("NVDA")
print(result)
```

---

## Best Practices

### 1. Framework Selection

**Use Pydantic AI when:**
- You need strict type validation
- Working primarily with Claude models
- Building data-focused agents
- Prefer lightweight dependencies

**Use Google ADK when:**
- Building complex multi-agent systems
- Need session and memory management
- Deploying to Agent Engine
- Want native Vertex AI integration

---

### 2. MCP Tool Design

**Guidelines:**
- Keep tools focused on single responsibilities
- Provide clear docstrings with examples
- Return structured, parseable data
- Handle errors gracefully
- Include timeout configurations

**Example:**
```python
@mcp.tool()
async def analyze_sentiment(
    symbol: str,
    timeframe: str = "1d"
) -> str:
    """Analyze market sentiment for a symbol.

    Args:
        symbol: Stock ticker (e.g., 'NVDA')
        timeframe: Analysis period ('1d', '1w', '1m')

    Returns:
        JSON-formatted sentiment report

    Example:
        >>> await analyze_sentiment("NVDA", "1w")
        '{"score": 0.75, "rating": "POSITIVE", ...}'
    """
    try:
        # Analysis logic
        result = {"score": 0.75, "rating": "POSITIVE"}
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
```

---

### 3. A2A Communication

**Best Practices:**
- Always implement proper agent cards
- Use agent executors for clean separation
- Handle task lifecycle properly (submit → working → complete)
- Provide informative error messages
- Test locally before deploying

---

### 4. Claude Model Selection

**Model Characteristics:**

| Model | Best For | Speed | Cost |
|-------|----------|-------|------|
| Claude Haiku 4 | Simple tasks, high volume | Fastest | Lowest |
| Claude Sonnet 4.5 | Balanced performance | Fast | Medium |
| Claude Opus 4 | Complex reasoning | Slower | Highest |

**When to use Claude vs Gemini:**
- **Claude**: Better at reasoning, analysis, creative writing
- **Gemini**: Better at code generation, structured data, multimodal

---

## Troubleshooting

### Common Issues

**1. MCP Tools Not Loading**
```python
# Ensure correct server startup
mcp_server = MCPServerStdio(
    "python",  # Use python, not python3
    args=["mcp_tools/server.py"],  # Correct path
    timeout=60  # Sufficient timeout
)
```

**2. A2A Authentication Failures**
```python
# Verify authentication
from google.auth import default

credentials, project = default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
print(f"Authenticated: {credentials.valid}")
print(f"Project: {project}")
```

**3. Claude Model Not Found**
```python
# Check LiteLLM configuration
import litellm

litellm.vertex_project = "your-project"
litellm.vertex_location = "global"  # Use 'global', not region

# Verify model name
model_name = "vertex_ai/claude-sonnet-4-5@20250929"
```

---

## Resources

### Documentation
- [A2A Protocol Specification](https://a2aprotocol.ai/)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [LiteLLM Vertex AI](https://docs.litellm.ai/docs/providers/vertex)

### Related Guides
- `tutorial_a2a_on_agent_engine.ipynb` - A2A fundamentals
- `guide_multi_agent_orchestration.md` - Orchestration patterns
- `guide_security_authentication.md` - Production security

### Example Code
- Bear/Bull trading agent example (reference implementation)
- Multi-framework orchestration patterns
- MCP tool packaging examples

---

## Next Steps

1. **Start Simple**: Build a single agent with one MCP tool
2. **Test Locally**: Use local A2A servers for development
3. **Add Complexity**: Introduce multi-agent orchestration
4. **Deploy to Production**: Use Agent Engine for managed deployment
5. **Monitor & Iterate**: Add observability and refine based on usage

---

**Version:** 1.0
**Last Updated:** 2025-11-12
**Maintainer:** IntentSolutions
**Status:** Production Ready

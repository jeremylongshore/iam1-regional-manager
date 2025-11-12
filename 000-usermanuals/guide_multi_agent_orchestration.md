# Multi-Agent Orchestration Guide
**Version:** 1.0
**Last Updated:** 2025-11-12
**Applies To:** A2A Protocol, Google ADK, IAM1/IAM2 Patterns

---

## Overview

This guide covers patterns and best practices for orchestrating multiple AI agents to work together, based on the A2A (Agent2Agent) Protocol and hierarchical agent architectures like IAM1/IAM2.

**Key Concepts:**
- **Coordination** - Agents working together as peers
- **Delegation** - Manager agents assigning tasks to specialists
- **Discovery** - Agents finding and learning about other agents
- **Communication** - Standard protocols for agent-to-agent interaction

---

## Table of Contents

1. [Agent Hierarchies](#agent-hierarchies)
2. [A2A Protocol Communication](#a2a-protocol-communication)
3. [Coordination Patterns](#coordination-patterns)
4. [Implementation Examples](#implementation-examples)
5. [Best Practices](#best-practices)
6. [Performance & Scaling](#performance--scaling)

---

## Agent Hierarchies

### Pattern 1: Manager-Specialist (IAM1/IAM2)

**Architecture:**
```
IAM1 (Manager Agent)
├── IAM2 Specialist 1 (Research)
├── IAM2 Specialist 2 (Code)
├── IAM2 Specialist 3 (Data)
└── IAM2 Specialist 4 (Communication)
```

**Characteristics:**
- IAM1 **commands** IAM2 specialists (subordinates)
- IAM1 **coordinates** with peer IAM1s (equals)
- IAM2 specialists report only to their IAM1
- Clear hierarchy and responsibility

**Use Cases:**
- Department-specific AI assistants
- Domain-specialized agent teams
- Enterprise multi-agent systems

---

### Pattern 2: Peer-to-Peer Network

**Architecture:**
```
┌─────────┐     A2A      ┌─────────┐
│ Agent A │ ◄─────────► │ Agent B │
└─────────┘             └─────────┘
     ▲ A2A                   ▲ A2A
     │                       │
     ▼                       ▼
┌─────────┐             ┌─────────┐
│ Agent C │ ◄─────────► │ Agent D │
└─────────┘     A2A      └─────────┘
```

**Characteristics:**
- All agents are equals
- No central controller
- Collaborative problem-solving
- Distributed decision-making

**Use Cases:**
- Research collaboration
- Multi-domain data aggregation
- Distributed task execution

---

### Pattern 3: Pipeline/Sequential

**Architecture:**
```
Input → Agent 1 → Agent 2 → Agent 3 → Output
        (Ingest)  (Process) (Format)
```

**Characteristics:**
- Linear data flow
- Each agent specializes in one step
- Output of one is input to next
- Clear transformation pipeline

**Use Cases:**
- Data processing pipelines
- Multi-stage analysis
- Content generation workflows

---

## A2A Protocol Communication

### Agent Discovery via Agent Cards

**Agent Card Structure:**
```python
from a2a.types import AgentCard, AgentSkill

# Define agent capabilities
research_skill = AgentSkill(
    id="deep_research",
    name="Deep Research",
    description="Conduct comprehensive research using multiple sources",
    tags=["research", "analysis", "synthesis"],
    examples=[
        "Research market trends for AI agents",
        "Analyze competitive landscape",
        "Synthesize findings from multiple reports"
    ],
    input_modes=["text/plain"],
    output_modes=["text/plain", "application/json"]
)

# Create agent card
agent_card = AgentCard(
    name="Research Specialist",
    description="Expert research agent with web search and document analysis",
    skills=[research_skill],
    url="https://agent-engine.googleapis.com/v1/agents/research-specialist"
)
```

**Discovery Workflow:**
```python
from a2a.client import ClientFactory, ClientConfig

class AgentDiscovery:
    """Discover and register available agents."""

    def __init__(self):
        self.agent_registry = {}

    async def discover_agent(self, agent_url: str):
        """Discover agent capabilities via A2A protocol."""
        factory = ClientFactory(ClientConfig())
        client = factory.create_from_url(agent_url)

        # Fetch agent card
        agent_card = await client.get_card()

        # Register agent
        self.agent_registry[agent_card.name] = {
            'card': agent_card,
            'client': client,
            'skills': {skill.id: skill for skill in agent_card.skills}
        }

        return agent_card

    def find_agents_by_skill(self, skill_tag: str):
        """Find all agents with a specific skill."""
        matching_agents = []

        for agent_name, agent_info in self.agent_registry.items():
            for skill in agent_info['skills'].values():
                if skill_tag in skill.tags:
                    matching_agents.append((agent_name, skill))

        return matching_agents
```

---

### Peer Agent Coordination

**IAM1-to-IAM1 Coordination (Equals):**
```python
from a2a_sdk import A2AClient, Message
from a2a.types import Role, Part, TextPart

class PeerCoordinator:
    """Coordinate with peer IAM1 agents."""

    def __init__(self, peer_registry: dict):
        self.peers = peer_registry  # {domain: url}

    async def coordinate_with_peer(
        self,
        domain: str,
        request: str
    ) -> str:
        """Request information from peer IAM1 (coordination, not command)."""

        if domain not in self.peers:
            raise ValueError(f"Unknown peer domain: {domain}")

        peer_url = self.peers[domain]

        # Initialize A2A client
        client = A2AClient(base_url=peer_url)

        # Create coordination request
        message = Message(
            message_id=f"coord-{uuid.uuid4()}",
            role=Role.user,
            parts=[Part(root=TextPart(text=request))]
        )

        # Send request
        response = await client.send_message(message)

        # Extract task ID
        async for chunk in response:
            task = chunk[0]
            task_id = task.id
            break

        # Poll for completion
        task_result = await client.get_task(TaskQueryParams(id=task_id))

        # Extract response
        if task_result.artifacts:
            return task_result.artifacts[0].parts[0].root.text

        return "No response from peer"

# Usage example
coordinator = PeerCoordinator({
    'engineering': 'https://engineering-iam1.example.com',
    'sales': 'https://sales-iam1.example.com',
    'operations': 'https://ops-iam1.example.com'
})

# Coordinate (request information, don't command)
roadmap = await coordinator.coordinate_with_peer(
    domain='engineering',
    request='What is the Q2 product roadmap?'
)
```

---

### Hierarchical Delegation

**IAM1-to-IAM2 Delegation (Manager to Specialist):**
```python
from typing import Dict
from google.adk.agents import Agent

class SpecialistRegistry:
    """Registry of IAM2 specialist agents."""

    def __init__(self):
        self.specialists: Dict[str, Agent] = {}

    def register(self, specialist_type: str, agent: Agent):
        """Register a specialist agent."""
        self.specialists[specialist_type] = agent

    def get(self, specialist_type: str) -> Agent:
        """Get specialist agent by type."""
        return self.specialists.get(specialist_type)

class ManagerAgent:
    """IAM1 manager agent that delegates to IAM2 specialists."""

    def __init__(self, registry: SpecialistRegistry):
        self.registry = registry

    async def delegate_task(
        self,
        task_type: str,
        query: str
    ) -> str:
        """Delegate task to appropriate specialist (command, not coordinate)."""

        specialist = self.registry.get(task_type)

        if not specialist:
            raise ValueError(f"No specialist found for: {task_type}")

        # Execute task with specialist
        response = specialist.send_message(query)

        return response

# Setup
registry = SpecialistRegistry()

# Register specialists
registry.register('research', research_agent)
registry.register('code', code_agent)
registry.register('data', data_agent)

manager = ManagerAgent(registry)

# Delegate to specialist
result = await manager.delegate_task(
    task_type='research',
    query='Research best practices for agent orchestration'
)
```

---

## Coordination Patterns

### Pattern 1: Sequential Coordination

**Use Case:** Multi-step workflow where each agent builds on previous

```python
class SequentialOrchestrator:
    """Execute agents in sequence, passing outputs forward."""

    def __init__(self, agent_pipeline: list):
        self.pipeline = agent_pipeline

    async def execute(self, initial_input: str):
        """Execute pipeline sequentially."""
        current_input = initial_input
        results = []

        for agent_config in self.pipeline:
            agent_name = agent_config['name']
            agent = agent_config['agent']

            # Execute current agent
            response = await agent.send_message(current_input)

            results.append({
                'agent': agent_name,
                'output': response
            })

            # Use output as input for next agent
            current_input = response

        return results

# Define pipeline
pipeline = [
    {'name': 'Data Collector', 'agent': collector_agent},
    {'name': 'Data Analyzer', 'agent': analyzer_agent},
    {'name': 'Report Generator', 'agent': reporter_agent}
]

orchestrator = SequentialOrchestrator(pipeline)

# Execute
final_report = await orchestrator.execute("Analyze Q4 sales data")
```

---

### Pattern 2: Parallel Coordination

**Use Case:** Independent tasks that can run simultaneously

```python
import asyncio

class ParallelOrchestrator:
    """Execute multiple agents in parallel."""

    def __init__(self, agents: dict):
        self.agents = agents

    async def execute_parallel(self, tasks: dict):
        """Execute multiple agents concurrently.

        Args:
            tasks: {agent_name: query}

        Returns:
            {agent_name: response}
        """

        async def run_agent(agent_name, query):
            """Helper to run single agent."""
            agent = self.agents[agent_name]
            response = await agent.send_message(query)
            return agent_name, response

        # Create tasks for all agents
        agent_tasks = [
            run_agent(name, query)
            for name, query in tasks.items()
        ]

        # Execute all in parallel
        results = await asyncio.gather(*agent_tasks)

        # Convert to dict
        return dict(results)

# Setup
agents = {
    'market_research': market_agent,
    'competitor_analysis': competitor_agent,
    'customer_sentiment': sentiment_agent
}

orchestrator = ParallelOrchestrator(agents)

# Execute in parallel
results = await orchestrator.execute_parallel({
    'market_research': 'Research AI agent market size',
    'competitor_analysis': 'Analyze top 5 competitors',
    'customer_sentiment': 'Summarize customer feedback'
})
```

---

### Pattern 3: Conditional Coordination

**Use Case:** Route to different agents based on task characteristics

```python
class ConditionalOrchestrator:
    """Route tasks to agents based on conditions."""

    def __init__(self, router_agent, specialist_registry):
        self.router = router_agent
        self.registry = specialist_registry

    async def execute(self, user_query: str):
        """Conditionally route to appropriate specialist."""

        # Use router agent to classify task
        classification = await self.router.send_message(
            f"Classify this task: {user_query}"
        )

        # Extract task type from classification
        task_type = self._parse_classification(classification)

        # Route to appropriate specialist
        specialist = self.registry.get(task_type)

        if not specialist:
            return f"No specialist available for: {task_type}"

        # Execute with specialist
        response = await specialist.send_message(user_query)

        return {
            'task_type': task_type,
            'response': response
        }

    def _parse_classification(self, classification: str) -> str:
        """Extract task type from classification response."""
        # Implementation depends on router agent output format
        # Example: "TASK_TYPE: research"
        if "research" in classification.lower():
            return "research"
        elif "code" in classification.lower():
            return "code"
        elif "data" in classification.lower():
            return "data"
        else:
            return "general"
```

---

### Pattern 4: Hierarchical Aggregation

**Use Case:** Collect information from multiple specialists, synthesize

```python
class AggregationOrchestrator:
    """Collect from multiple specialists, synthesize with manager."""

    def __init__(self, specialists: dict, synthesizer):
        self.specialists = specialists
        self.synthesizer = synthesizer

    async def execute(self, base_query: str):
        """Gather from specialists, synthesize final answer."""

        # Step 1: Query all specialists in parallel
        specialist_queries = {
            name: f"{base_query} (focus: {name})"
            for name in self.specialists.keys()
        }

        async def query_specialist(name, query):
            agent = self.specialists[name]
            response = await agent.send_message(query)
            return name, response

        tasks = [
            query_specialist(name, query)
            for name, query in specialist_queries.items()
        ]

        specialist_results = dict(await asyncio.gather(*tasks))

        # Step 2: Synthesize all specialist responses
        synthesis_prompt = f"""
        Original Query: {base_query}

        Specialist Responses:
        {self._format_responses(specialist_results)}

        Synthesize a comprehensive answer that integrates all specialist insights.
        """

        final_answer = await self.synthesizer.send_message(synthesis_prompt)

        return {
            'specialist_responses': specialist_results,
            'synthesized_answer': final_answer
        }

    def _format_responses(self, responses: dict) -> str:
        """Format specialist responses for synthesis."""
        formatted = []
        for name, response in responses.items():
            formatted.append(f"**{name}:**\n{response}\n")
        return "\n".join(formatted)
```

---

## Implementation Examples

### Complete Multi-Agent System

```python
from google.adk.agents import Agent, LlmAgent
from a2a.types import AgentSkill

class MultiAgentSystem:
    """Complete multi-agent orchestration system."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.manager = None
        self.specialists = {}
        self.peers = {}

    def setup_manager(self):
        """Create IAM1 manager agent."""
        self.manager = LlmAgent(
            model="gemini-2.5-flash",
            name="iam1_manager",
            description="Regional manager coordinating specialist team",
            instruction="""You are IAM1, a Regional Manager agent.

            You can:
            1. Delegate to specialist IAM2 agents (subordinates)
            2. Coordinate with peer IAM1 agents (equals)
            3. Synthesize information from multiple sources

            Decision framework:
            - Simple questions → Answer directly
            - Specialized tasks → Delegate to IAM2
            - Cross-domain info → Coordinate with peer IAM1
            - Complex tasks → Coordinate multiple agents
            """,
            tools=[self.delegate_to_specialist, self.coordinate_with_peer]
        )

    def add_specialist(self, specialist_type: str, agent: Agent):
        """Register IAM2 specialist agent."""
        self.specialists[specialist_type] = agent

    def add_peer(self, domain: str, peer_url: str):
        """Register peer IAM1 agent."""
        self.peers[domain] = peer_url

    async def delegate_to_specialist(self, task_type: str, query: str) -> str:
        """Delegate to IAM2 specialist."""
        specialist = self.specialists.get(task_type)
        if not specialist:
            return f"No specialist for: {task_type}"

        response = await specialist.send_message(query)
        return f"[IAM2 {task_type}]: {response}"

    async def coordinate_with_peer(self, domain: str, request: str) -> str:
        """Coordinate with peer IAM1."""
        if domain not in self.peers:
            return f"Unknown peer: {domain}"

        peer_client = A2AClient(base_url=self.peers[domain])

        message = Message(
            message_id=f"peer-{uuid.uuid4()}",
            role=Role.user,
            parts=[Part(root=TextPart(text=request))]
        )

        response = await peer_client.send_message(message)

        # Extract response (simplified)
        return f"[IAM1 {domain}]: <response>"

# Setup complete system
system = MultiAgentSystem(project_id="my-project")

# Setup manager
system.setup_manager()

# Add specialists
system.add_specialist('research', research_agent)
system.add_specialist('code', code_agent)
system.add_specialist('data', data_agent)

# Add peers
system.add_peer('engineering', 'https://eng-iam1.example.com')
system.add_peer('sales', 'https://sales-iam1.example.com')

# Use system
response = await system.manager.send_message(
    "Research Q2 product features and check with engineering for feasibility"
)
```

---

## Best Practices

### 1. Clear Responsibility Boundaries

**DO:**
- Define clear roles for each agent
- Document what each agent can/cannot do
- Establish hierarchy (manager vs specialist)

**DON'T:**
- Create overlapping agent responsibilities
- Allow circular dependencies
- Mix peer coordination with hierarchical command

---

### 2. Error Handling

```python
class RobustOrchestrator:
    """Orchestrator with comprehensive error handling."""

    async def execute_with_fallback(self, primary_agent, fallback_agent, query):
        """Try primary agent, fall back to secondary if fails."""

        try:
            response = await asyncio.wait_for(
                primary_agent.send_message(query),
                timeout=30.0
            )
            return response

        except asyncio.TimeoutError:
            logging.warning(f"Primary agent timeout, using fallback")
            return await fallback_agent.send_message(query)

        except Exception as e:
            logging.error(f"Primary agent failed: {e}")
            return await fallback_agent.send_message(query)
```

---

### 3. Monitoring & Observability

```python
from datetime import datetime
import logging

class ObservableOrchestrator:
    """Orchestrator with comprehensive monitoring."""

    def __init__(self):
        self.logger = logging.getLogger('orchestration')
        self.metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'agent_usage': {}
        }

    async def execute_with_monitoring(self, agent_name, agent, query):
        """Execute agent with full monitoring."""

        start_time = datetime.now()

        self.logger.info(f"Starting task", extra={
            'agent': agent_name,
            'query_length': len(query),
            'timestamp': start_time.isoformat()
        })

        try:
            response = await agent.send_message(query)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.metrics['total_tasks'] += 1
            self.metrics['successful_tasks'] += 1
            self.metrics['agent_usage'][agent_name] = \
                self.metrics['agent_usage'].get(agent_name, 0) + 1

            self.logger.info(f"Task completed", extra={
                'agent': agent_name,
                'duration_seconds': duration,
                'response_length': len(response)
            })

            return response

        except Exception as e:
            self.metrics['total_tasks'] += 1
            self.metrics['failed_tasks'] += 1

            self.logger.error(f"Task failed", extra={
                'agent': agent_name,
                'error': str(e)
            })

            raise
```

---

## Performance & Scaling

### Parallel Execution for Speed

```python
# DON'T: Sequential (slow)
result1 = await agent1.send_message(query1)
result2 = await agent2.send_message(query2)
result3 = await agent3.send_message(query3)

# DO: Parallel (fast)
results = await asyncio.gather(
    agent1.send_message(query1),
    agent2.send_message(query2),
    agent3.send_message(query3)
)
```

### Caching for Efficiency

```python
from functools import lru_cache
import hashlib

class CachedOrchestrator:
    """Orchestrator with response caching."""

    def __init__(self):
        self.cache = {}

    def _cache_key(self, agent_name: str, query: str) -> str:
        """Generate cache key."""
        content = f"{agent_name}:{query}"
        return hashlib.md5(content.encode()).hexdigest()

    async def execute_with_cache(self, agent_name, agent, query, ttl=300):
        """Execute with caching."""

        cache_key = self._cache_key(agent_name, query)

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_response = self.cache[cache_key]

            if (datetime.now() - cached_time).total_seconds() < ttl:
                logging.info(f"Cache hit for {agent_name}")
                return cached_response

        # Execute agent
        response = await agent.send_message(query)

        # Store in cache
        self.cache[cache_key] = (datetime.now(), response)

        return response
```

---

## Resources

- `tutorial_a2a_on_agent_engine.ipynb` - A2A protocol implementation
- `000-docs/003-RA-ANLY-a2a-integration.md` - A2A integration analysis
- `app/a2a_tools.py` - IAM1 peer coordination implementation
- `app/sub_agents.py` - IAM2 specialist registry
- [A2A Protocol Specification](https://a2aprotocol.ai/)
- [Google ADK Multi-Agent Systems](https://cloud.google.com/blog/products/ai-machine-learning/build-multi-agentic-systems-using-google-adk)

---

**Next Steps:**
1. Design your agent hierarchy
2. Define agent responsibilities
3. Implement coordination patterns
4. Add monitoring and error handling
5. Test multi-agent workflows
6. Deploy and scale

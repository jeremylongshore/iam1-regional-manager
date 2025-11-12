"""
Query deployed Supreme Commander Agent.

Interactive script to query the deployed agent.
"""

import os
import sys

from google.cloud import aiplatform


def interactive_query():
    """Interactive query interface for Supreme Commander."""

    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION", "us-central1")

    if not project_id:
        print("❌ ERROR: PROJECT_ID environment variable not set")
        sys.exit(1)

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Get agent resource name
    agent_resource_name = os.getenv("AGENT_RESOURCE_NAME")

    if not agent_resource_name:
        # Try to load from file
        try:
            with open("agent_resource_name.txt") as f:
                agent_resource_name = f.read().strip()
        except FileNotFoundError:
            print("❌ ERROR: Agent resource name not found")
            print("   Set AGENT_RESOURCE_NAME environment variable or run 'make deploy' first")
            sys.exit(1)

    print("🤖 Supreme Commander Agent - Interactive Query")
    print(f"   Agent: {agent_resource_name}")
    print()
    print("Commands:")
    print("  - Type your query and press Enter")
    print("  - Type 'exit' or 'quit' to quit")
    print("  - Type 'clear' to clear screen")
    print()

    # Get agent
    try:
        agent = aiplatform.ReasoningEngine(agent_resource_name)
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to agent: {e}")
        sys.exit(1)

    # Interactive loop
    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! 👋")
                break

            if query.lower() == "clear":
                os.system("clear" if os.name != "nt" else "cls")
                continue

            # Query agent
            print("🤔 Supreme Commander is thinking...")
            response = agent.query(input=query)

            print()
            print(f"Supreme Commander: {response.get('output', 'No response')}")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ ERROR: {e}\n")


def single_query(query: str):
    """Query agent with a single message."""

    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION", "us-central1")
    agent_resource_name = os.getenv("AGENT_RESOURCE_NAME")

    if not project_id:
        print("❌ ERROR: PROJECT_ID not set")
        sys.exit(1)

    if not agent_resource_name:
        try:
            with open("agent_resource_name.txt") as f:
                agent_resource_name = f.read().strip()
        except FileNotFoundError:
            print("❌ ERROR: Agent resource name not found")
            sys.exit(1)

    # Initialize and query
    aiplatform.init(project=project_id, location=region)
    agent = aiplatform.ReasoningEngine(agent_resource_name)

    response = agent.query(input=query)
    print(response.get("output", "No response"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query from command line
        query = " ".join(sys.argv[1:])
        single_query(query)
    else:
        # Interactive mode
        interactive_query()

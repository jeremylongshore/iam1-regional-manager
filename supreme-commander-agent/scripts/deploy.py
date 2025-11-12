"""
Deploy Supreme Commander Agent to Vertex AI Agent Engine.

This script packages and deploys the agent to Vertex AI.
"""

import os
import sys

from google.cloud import aiplatform


def deploy_agent():
    """Deploy Supreme Commander to Vertex AI Agent Engine."""

    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION", "us-central1")
    agent_name = os.getenv("AGENT_NAME", "supreme-commander")

    if not project_id:
        print("❌ ERROR: PROJECT_ID environment variable not set")
        sys.exit(1)

    print("🚀 Deploying Supreme Commander Agent...")
    print(f"   Project: {project_id}")
    print(f"   Region: {region}")
    print(f"   Agent Name: {agent_name}")
    print()

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Import agent
    try:
        from app.agent import supreme_commander
    except ImportError as e:
        print(f"❌ ERROR: Failed to import agent: {e}")
        print("   Make sure you're in the supreme-commander-agent directory")
        sys.exit(1)

    # Deploy to Agent Engine
    try:
        from google.cloud.aiplatform import reasoning_engines

        print("📦 Packaging agent...")

        deployed_agent = reasoning_engines.ReasoningEngine.create(
            reasoning_engine=supreme_commander,
            requirements=[
                "google-cloud-aiplatform>=1.60.0",
                "google-adk-python>=0.1.0",
                "slack-bolt>=1.18.0",
                "slack-sdk>=3.27.0",
                "firebase-admin>=6.5.0",
                "google-cloud-discoveryengine>=0.11.0",
                "google-cloud-firestore>=2.16.0",
                "google-cloud-storage>=2.16.0",
                "opentelemetry-api>=1.25.0",
                "opentelemetry-sdk>=1.25.0",
                "opentelemetry-exporter-gcp-trace>=1.6.0",
                "pydantic>=2.8.0",
            ],
            display_name=f"{agent_name}-{os.getenv('ENVIRONMENT', 'dev')}",
            description="Supreme Commander Agent - Boss of all boss agents",
            extra_packages=["."],
        )

        print()
        print("✅ Agent deployed successfully!")
        print()
        print(f"Resource name: {deployed_agent.resource_name}")
        print()
        print("Test the agent:")
        print("```python")
        print("from google.cloud import aiplatform")
        print(f"agent = aiplatform.ReasoningEngine('{deployed_agent.resource_name}')")
        print("response = agent.query(input='Are you operational?')")
        print("print(response['output'])")
        print("```")

        # Save resource name
        with open("agent_resource_name.txt", "w") as f:
            f.write(deployed_agent.resource_name)

        return deployed_agent.resource_name

    except Exception as e:
        print(f"❌ ERROR: Deployment failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    deploy_agent()

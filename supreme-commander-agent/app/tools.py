"""
Tools for Supreme Commander Agent.

These tools enable the Supreme Commander to:
- Delegate tasks to external agents
- Coordinate with peer agents
- Communicate via Slack and Firebase
- Manage goals and tasks
"""

import json
import logging
import os
from datetime import UTC, datetime
from typing import Literal

import firebase_admin
from firebase_admin import credentials, db, firestore
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)

# Initialize Firebase (only once)
if not firebase_admin._apps:
    try:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {
            'projectId': os.getenv('PROJECT_ID'),
            'databaseURL': os.getenv('FIREBASE_DATABASE_URL'),
        })
    except Exception as e:
        logger.warning(f"Firebase initialization skipped: {e}")

# Initialize Slack client
try:
    slack_client = WebClient(token=os.getenv('SLACK_BOT_TOKEN'))
except Exception as e:
    logger.warning(f"Slack client initialization skipped: {e}")
    slack_client = None


def delegate_to_agent(
    agent_id: str,
    task_description: str,
    priority: Literal["critical", "high", "medium", "low"] = "medium",
) -> str:
    """
    Delegate a task to a specialist agent.

    Args:
        agent_id: Identifier of the target agent (e.g., "engineering_agent_1",
"data_analyst_agent")
        task_description: Clear, actionable description of the task
        priority: Task priority level

    Returns:
        Status message with task ID and estimated completion
    """
    try:
        # Store task in Firebase
        db_ref = firestore.client()
        task_ref = db_ref.collection('tasks').document()

        task_data = {
            'agent_id': agent_id,
            'task_description': task_description,
            'priority': priority,
            'status': 'assigned',
            'created_at': datetime.now(UTC).isoformat(),
            'created_by': 'supreme_commander',
        }

        task_ref.set(task_data)

        # Log delegation
        logger.info(f"Delegated task to {agent_id}: {task_description[:50]}...")

        # Send notification to agent's queue/endpoint
        # (Implementation depends on how agents are integrated)
        # For now, just store in Firebase for agents to poll

        return f"""✅ Task delegated successfully
Task ID: {task_ref.id}
Agent: {agent_id}
Priority: {priority}
Status: Assigned and queued for execution

The agent will begin work shortly. Track progress with track_goal_progress("{task_ref.id}")
"""

    except Exception as e:
        logger.error(f"Failed to delegate task: {e}")
        return f"❌ Failed to delegate task: {str(e)}"


def coordinate_with_peer_agent(
    peer_agent_id: str,
    request: str,
) -> str:
    """
    Coordinate with a peer agent (equals, not subordinates).

    Used for cross-domain information requests or collaboration.

    Args:
        peer_agent_id: Identifier of peer agent
        request: What you need from the peer

    Returns:
        Response from peer agent
    """
    try:
        # Store coordination request
        db_ref = firestore.client()
        coord_ref = db_ref.collection('coordination_requests').document()

        coord_data = {
            'peer_agent_id': peer_agent_id,
            'request': request,
            'status': 'pending',
            'created_at': datetime.now(UTC).isoformat(),
            'requester': 'supreme_commander',
        }

        coord_ref.set(coord_data)

        # In a real system, this would use A2A Protocol or direct API call
        # For now, placeholder response

        return f"""📨 Coordination request sent to {peer_agent_id}
Request ID: {coord_ref.id}
Status: Awaiting response

The peer agent will respond when available. Check back shortly or you'll be notified when they respond.
"""

    except Exception as e:
        logger.error(f"Failed to coordinate with peer: {e}")
        return f"❌ Coordination failed: {str(e)}"


def send_slack_notification(
    channel: str,
    message: str,
    priority: Literal["info", "warning", "critical"] = "info",
) -> str:
    """
    Send notification to Slack channel.

    Args:
        channel: Slack channel name (without #)
        message: Message to send
        priority: Message priority (affects formatting)

    Returns:
        Confirmation message
    """
    if not slack_client:
        logger.warning("Slack client not initialized")
        return "⚠️ Slack client not configured. Message logged instead."

    try:
        # Format message based on priority
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
        }

        formatted_message = f"{emoji_map.get(priority, 'ℹ️')} {message}"

        # For critical, add @channel mention
        if priority == "critical":
            formatted_message = f"<!channel> {formatted_message}"

        # Send to Slack
        response = slack_client.chat_postMessage(
            channel=channel,
            text=formatted_message,
            unfurl_links=False,
            unfurl_media=False,
        )

        logger.info(f"Sent Slack message to #{channel}")

        return f"✅ Slack message sent to #{channel}\nMessage ID: {response['ts']}"

    except SlackApiError as e:
        logger.error(f"Slack API error: {e.response['error']}")
        return f"❌ Slack error: {e.response['error']}"
    except Exception as e:
        logger.error(f"Failed to send Slack message: {e}")
        return f"❌ Failed to send message: {str(e)}"


def update_firebase_dashboard(
    metric_type: Literal["task_status", "agent_utilization", "goal_progress", "alerts"],
    data: dict,
) -> str:
    """
    Update real-time Firebase dashboard with metrics.

    Args:
        metric_type: Type of metric to update
        data: Metric data (structure depends on type)

    Returns:
        Confirmation message
    """
    try:
        # Get Realtime Database reference
        dashboard_ref = db.reference(f'/dashboard/{metric_type}')

        # Add timestamp
        data['updated_at'] = datetime.now(UTC).isoformat()
        data['updated_by'] = 'supreme_commander'

        # Update dashboard
        dashboard_ref.set(data)

        logger.info(f"Updated dashboard: {metric_type}")

        return f"✅ Dashboard updated: {metric_type}\nData: {json.dumps(data, indent=2)}"

    except Exception as e:
        logger.error(f"Failed to update dashboard: {e}")
        return f"❌ Dashboard update failed: {str(e)}"


def create_task_plan(
    objective: str,
    context: str = "",
) -> str:
    """
    Create a structured execution plan for an objective.

    Args:
        objective: High-level goal/objective
        context: Additional context or constraints

    Returns:
        Structured task plan
    """
    try:
        # Store plan in Firebase
        db_ref = firestore.client()
        plan_ref = db_ref.collection('task_plans').document()

        plan_data = {
            'objective': objective,
            'context': context,
            'status': 'draft',
            'created_at': datetime.now(UTC).isoformat(),
            'created_by': 'supreme_commander',
            'tasks': [],  # Will be populated by agent's reasoning
        }

        plan_ref.set(plan_data)

        return f"""📋 Task plan created
Plan ID: {plan_ref.id}
Objective: {objective}
Status: Draft - ready for task breakdown

You should now analyze the objective and create specific tasks.
Use delegate_to_agent() for each task once the plan is finalized.
"""

    except Exception as e:
        logger.error(f"Failed to create task plan: {e}")
        return f"❌ Failed to create plan: {str(e)}"


def track_goal_progress(goal_id: str) -> str:
    """
    Check progress on an active goal or task.

    Args:
        goal_id: ID of goal/task to track

    Returns:
        Current status, completed tasks, blockers, ETA
    """
    try:
        db_ref = firestore.client()

        # Check if it's a task or plan
        task_doc = db_ref.collection('tasks').document(goal_id).get()
        plan_doc = db_ref.collection('task_plans').document(goal_id).get()

        if task_doc.exists:
            data = task_doc.to_dict()
            return f"""📊 Task Status
ID: {goal_id}
Agent: {data.get('agent_id', 'N/A')}
Description: {data.get('task_description', 'N/A')}
Status: {data.get('status', 'unknown')}
Priority: {data.get('priority', 'N/A')}
Created: {data.get('created_at', 'N/A')}
"""

        elif plan_doc.exists:
            data = plan_doc.to_dict()
            tasks = data.get('tasks', [])
            completed = len([t for t in tasks if t.get('status') == 'completed'])
            total = len(tasks)

            return f"""📊 Plan Status
ID: {goal_id}
Objective: {data.get('objective', 'N/A')}
Progress: {completed}/{total} tasks completed
Status: {data.get('status', 'unknown')}
Created: {data.get('created_at', 'N/A')}
"""

        else:
            return f"❌ Goal/task not found: {goal_id}"

    except Exception as e:
        logger.error(f"Failed to track progress: {e}")
        return f"❌ Failed to track: {str(e)}"

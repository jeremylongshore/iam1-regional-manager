"""
Firebase interface for Supreme Commander dashboard.

Provides real-time dashboard with:
- Active goals and tasks
- Agent utilization metrics
- System alerts
- Performance KPIs
"""

import logging
import os
from datetime import UTC, datetime
from typing import Any

import firebase_admin
from firebase_admin import credentials, db, firestore

from app.agent import handle_message

logger = logging.getLogger(__name__)


def initialize_firebase():
    """Initialize Firebase Admin SDK if not already initialized."""
    if not firebase_admin._apps:
        try:
            # Use Application Default Credentials in production
            cred = credentials.ApplicationDefault()

            firebase_admin.initialize_app(cred, {
                'projectId': os.getenv('PROJECT_ID'),
                'databaseURL': os.getenv('FIREBASE_DATABASE_URL'),
            })

            logger.info("Firebase initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise


def handle_dashboard_request(request_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """
    Handle incoming requests from Firebase dashboard.

    Args:
        request_type: Type of request ("query", "command", "status_check")
        data: Request data

    Returns:
        Response data for dashboard
    """
    try:
        if request_type == "query":
            # Dashboard user asking a question
            query = data.get("query", "")
            user_id = data.get("user_id", "dashboard_user")

            response = handle_message(
                message=query,
                user_id=user_id,
                channel="dashboard",
            )

            return {
                "status": "success",
                "response": response,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        elif request_type == "command":
            # Dashboard executing a command (e.g., "pause all tasks")
            command = data.get("command", "")
            user_id = data.get("user_id", "dashboard_user")

            response = handle_message(
                message=f"[COMMAND] {command}",
                user_id=user_id,
                channel="dashboard",
            )

            return {
                "status": "success",
                "response": response,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        elif request_type == "status_check":
            # Dashboard requesting system status
            return get_system_status()

        else:
            return {
                "status": "error",
                "error": f"Unknown request type: {request_type}",
            }

    except Exception as e:
        logger.error(f"Error handling dashboard request: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def get_system_status() -> dict[str, Any]:
    """
    Get current system status for dashboard.

    Returns:
        System status including active tasks, goals, alerts
    """
    try:
        db_ref = firestore.client()

        # Get active tasks
        active_tasks = db_ref.collection('tasks').where(
            'status', 'in', ['assigned', 'in_progress']
        ).limit(10).get()

        task_count = {
            'assigned': 0,
            'in_progress': 0,
            'completed': 0,
            'failed': 0,
        }

        for task in active_tasks:
            status = task.to_dict().get('status', 'unknown')
            task_count[status] = task_count.get(status, 0) + 1

        # Get active plans
        active_plans = db_ref.collection('task_plans').where(
            'status', '==', 'active'
        ).limit(5).get()

        # Get recent alerts
        recent_alerts = db_ref.collection('alerts').order_by(
            'created_at', direction=firestore.Query.DESCENDING
        ).limit(5).get()

        return {
            "status": "success",
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics": {
                "active_tasks": task_count['assigned'] + task_count['in_progress'],
                "completed_today": task_count['completed'],
                "failed_tasks": task_count['failed'],
                "active_plans": len(list(active_plans)),
                "recent_alerts": len(list(recent_alerts)),
            },
            "tasks": [
                {
                    "id": task.id,
                    "agent": task.to_dict().get('agent_id'),
                    "status": task.to_dict().get('status'),
                    "priority": task.to_dict().get('priority'),
                }
                for task in active_tasks
            ],
        }

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def update_dashboard_metric(
    metric_name: str,
    value: Any,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """
    Update a specific dashboard metric in realtime.

    Args:
        metric_name: Name of metric to update
        value: Metric value
        metadata: Additional metadata

    Returns:
        True if successful
    """
    try:
        # Update in Realtime Database for instant dashboard updates
        dashboard_ref = db.reference(f'/metrics/{metric_name}')

        data = {
            'value': value,
            'updated_at': datetime.now(UTC).isoformat(),
        }

        if metadata:
            data['metadata'] = metadata

        dashboard_ref.set(data)

        logger.info(f"Updated dashboard metric: {metric_name} = {value}")
        return True

    except Exception as e:
        logger.error(f"Failed to update dashboard metric: {e}")
        return False


def send_dashboard_alert(
    alert_type: str,
    message: str,
    severity: str = "info",
    metadata: dict[str, Any] | None = None,
) -> bool:
    """
    Send an alert to the Firebase dashboard.

    Args:
        alert_type: Type of alert ("task_failed", "system_error", "goal_completed", etc.)
        message: Alert message
        severity: "info", "warning", "critical"
        metadata: Additional context

    Returns:
        True if successful
    """
    try:
        db_ref = firestore.client()

        alert_data = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'created_at': datetime.now(UTC).isoformat(),
            'status': 'active',
        }

        if metadata:
            alert_data['metadata'] = metadata

        # Store in Firestore for persistence
        alert_ref = db_ref.collection('alerts').document()
        alert_ref.set(alert_data)

        # Also send to Realtime Database for instant notification
        realtime_ref = db.reference(f'/alerts/{alert_ref.id}')
        realtime_ref.set(alert_data)

        logger.info(f"Sent dashboard alert: {alert_type} - {message}")
        return True

    except Exception as e:
        logger.error(f"Failed to send dashboard alert: {e}")
        return False


# Initialize on module import
initialize_firebase()

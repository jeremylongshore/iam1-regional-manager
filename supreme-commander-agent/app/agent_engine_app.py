"""
Entry point for Vertex AI Agent Engine deployment.

This module packages the Supreme Commander agent for deployment
to Vertex AI Agent Engine (serverless, managed agent hosting).
"""

import logging

from app.agent import handle_message, supreme_commander

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def query(request: dict) -> dict:
    """
    Query endpoint for Agent Engine.

    Agent Engine calls this function when queried.

    Args:
        request: Dict with 'input' key containing user query

    Returns:
        Dict with 'output' key containing agent response
    """
    try:
        user_input = request.get('input', '')

        if not user_input:
            return {
                'output': 'Error: No input provided',
                'status': 'error',
            }

        logger.info(f"Received query: {user_input[:100]}...")

        # Get response from Supreme Commander
        response = handle_message(
            message=user_input,
            user_id=request.get('user_id'),
            channel=request.get('channel'),
        )

        logger.info(f"Generated response: {response[:100]}...")

        return {
            'output': response,
            'status': 'success',
        }

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return {
            'output': f'Error: {str(e)}',
            'status': 'error',
        }


# Export the agent for Agent Engine deployment
__all__ = ['supreme_commander', 'query']

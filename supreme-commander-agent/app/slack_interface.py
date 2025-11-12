"""
Slack interface for Supreme Commander using Slack Bolt.

Handles:
- Incoming messages/commands from Slack
- Sending responses back to Slack
- Real-time updates and notifications
"""

import logging
import os

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from app.agent import handle_message

logger = logging.getLogger(__name__)

# Initialize Slack Bolt app
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
)


@app.message(".*")
def handle_slack_message(message, say, client):
    """
    Handle incoming messages from Slack.

    Sends message to Supreme Commander and returns response.
    """
    try:
        user_id = message.get("user")
        channel = message.get("channel")
        text = message.get("text", "")

        # Ignore bot messages
        if message.get("bot_id"):
            return

        logger.info(f"Received message from {user_id} in {channel}: {text[:50]}...")

        # Show typing indicator
        client.chat_postMessage(
            channel=channel,
            text="🤔 Supreme Commander is thinking...",
            thread_ts=message.get("ts"),
        )

        # Get response from Supreme Commander
        response = handle_message(
            message=text,
            user_id=user_id,
            channel=channel,
        )

        # Send response
        say(
            text=response,
            thread_ts=message.get("ts"),  # Keep in thread
        )

        logger.info(f"Sent response to {user_id}")

    except Exception as e:
        logger.error(f"Error handling Slack message: {e}")
        say(
            text=f"❌ Error: {str(e)}",
            thread_ts=message.get("ts"),
        )


@app.command("/supreme")
def handle_supreme_command(ack, command, say):
    """
    Handle /supreme slash command.

    Usage: /supreme <your request>
    """
    try:
        ack()  # Acknowledge command

        user_id = command.get("user_id")
        channel = command.get("channel_id")
        text = command.get("text", "")

        if not text:
            say("Usage: /supreme <your request>")
            return

        logger.info(f"Received /supreme command from {user_id}: {text[:50]}...")

        # Get response from Supreme Commander
        response = handle_message(
            message=text,
            user_id=user_id,
            channel=channel,
        )

        # Send response
        say(response)

        logger.info(f"Sent /supreme response to {user_id}")

    except Exception as e:
        logger.error(f"Error handling /supreme command: {e}")
        say(f"❌ Error: {str(e)}")


@app.event("app_mention")
def handle_mention(event, say):
    """
    Handle @SupremeCommander mentions.
    """
    try:
        user_id = event.get("user")
        channel = event.get("channel")
        text = event.get("text", "")

        # Remove the mention from text
        text = text.split(">", 1)[-1].strip()

        logger.info(f"Mentioned by {user_id} in {channel}: {text[:50]}...")

        # Get response from Supreme Commander
        response = handle_message(
            message=text,
            user_id=user_id,
            channel=channel,
        )

        # Send response
        say(
            text=response,
            thread_ts=event.get("ts"),
        )

        logger.info(f"Sent mention response to {user_id}")

    except Exception as e:
        logger.error(f"Error handling mention: {e}")
        say(f"❌ Error: {str(e)}")


@app.event("reaction_added")
def handle_reaction(event):
    """
    Handle reactions to messages (for feedback/approval).
    """
    try:
        user = event.get("user")
        reaction = event.get("reaction")
        item = event.get("item", {})

        logger.info(f"Reaction {reaction} added by {user} to message {item.get('ts')}")

        # Could use reactions for:
        # ✅ - Approve a plan
        # ❌ - Reject/cancel
        # 🔄 - Retry
        # etc.

    except Exception as e:
        logger.error(f"Error handling reaction: {e}")


def start_slack_interface():
    """
    Start the Slack interface using Socket Mode.

    Socket Mode allows the app to receive events without exposing
    a public HTTP endpoint.
    """
    try:
        app_token = os.environ.get("SLACK_APP_TOKEN")

        if not app_token:
            logger.error("SLACK_APP_TOKEN not set. Slack interface disabled.")
            return

        handler = SocketModeHandler(app, app_token)

        logger.info("Starting Slack interface in Socket Mode...")
        handler.start()

    except Exception as e:
        logger.error(f"Failed to start Slack interface: {e}")
        raise


if __name__ == "__main__":
    # For local testing
    logging.basicConfig(level=logging.INFO)
    start_slack_interface()

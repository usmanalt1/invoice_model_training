from loguru import logger
from slack_sdk import WebClient
from config.settings import Settings

def send_slack_update(message):
    """
    Send update notification to health monitor channel

    Args:
        message(str): Message to include in notification
    """

    if Settings.ENABLE_SLACK == "True":
        logger.info("Sending slack notification")

        slack_channel = "client-svp-pfleiderer-health"
        bot_name = "Pfleiderer Pipeline Notifier"
        slack_client = WebClient(token=Settings.SLACK_TOKEN)

        try:
            slack_client.chat_postMessage(
                channel=slack_channel, text=message, username=bot_name
            )
        except Exception as e:
            logger.info(f"Error sending slack notification: {e}")

    else:
        logger.info("Slack notifications not enabled")

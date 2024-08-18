import logging
import os
from datetime import datetime

def setup_logging():
    # Check if the logs directory exists, and create it if it doesn't
    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_filename = f"logs/discord_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
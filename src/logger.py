import logging
import os
import sys
from datetime import datetime

# Create a log file with a timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create a 'logs' directory in the current working directory
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Full path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)  # This enables terminal output
    ]
)

if __name__ == "__main__":
    logging.info("Logging has started and is visible in the terminal!")

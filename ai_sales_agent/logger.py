import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent.log"),       # logs to a file
        logging.StreamHandler()                 # logs to console
    ]
)

logger = logging.getLogger("AI_SALES_AGENT")

import logging
import os
from datetime import datetime
from typing import Optional

class AgentLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Create a new log file for each day
        log_file = os.path.join(self.log_dir, f"agent_{datetime.now().strftime('%Y%m%d')}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("AI_SALES_AGENT")
        
    def log_interaction(self, customer_id: str, interaction_type: str, details: str):
        """Log customer interactions"""
        self.logger.info(f"Customer {customer_id} - {interaction_type}: {details}")
        
    def log_objection(self, customer_id: str, objection_type: str, response: str):
        """Log customer objections and responses"""
        self.logger.info(f"Customer {customer_id} - Objection ({objection_type}): {response}")
        
    def log_error(self, error_type: str, error_message: str, customer_id: Optional[str] = None):
        """Log errors with optional customer context"""
        context = f"Customer {customer_id} - " if customer_id else ""
        self.logger.error(f"{context}{error_type}: {error_message}")
        
    def log_insight(self, customer_id: str, insight_type: str, insight: str):
        """Log generated insights"""
        self.logger.info(f"Customer {customer_id} - Insight ({insight_type}): {insight}")
        
    def log_proposal(self, customer_id: str, proposal_type: str, status: str):
        """Log proposal generation and status"""
        self.logger.info(f"Customer {customer_id} - Proposal ({proposal_type}): {status}")

# Create a global logger instance
logger = AgentLogger()

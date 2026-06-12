import os
import logging
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

def setup_audit_logger():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    log_file = os.path.join(LOG_DIR, "audit.log")
    
    logger = logging.getLogger("audit_logger")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

audit = setup_audit_logger()

def log_interaction(query: str, retrieved_chunks: int, response: str):
    """Log a user interaction and system response."""
    audit.info(f"QUERY: {query}")
    audit.info(f"RETRIEVED_CHUNKS: {retrieved_chunks}")
    
    formatted_response = response.replace('\n', '\n\t')
    audit.info(f"RESPONSE:\n\t{formatted_response}")
    audit.info("-" * 80)

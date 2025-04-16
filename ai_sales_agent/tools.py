import requests
from logger import logger

SCORING_MODEL_URL = "https://your-deployed-model.com/score"

def get_lead_score(contact_id: str) -> str:
    logger.info(f"Fetching lead score for contact: {contact_id}")
    try:
        res = requests.get(f"{SCORING_MODEL_URL}?contact_id={contact_id}")
        res.raise_for_status()
        data = res.json()
        score = data.get("score", "No score found")
        logger.info(f"Lead score for {contact_id}: {score}")
        return f"Lead score for {contact_id} is {score}"
    except Exception as e:
        logger.error(f"Error retrieving lead score for {contact_id}: {e}")
        return f"Failed to retrieve lead score for {contact_id}"
        

def generate_document(contact_id: str) -> str:
    logger.info(f"Generating document for {contact_id}")
    return f"Document for {contact_id} has been generated!"

def suggest_next_action(contact_id: str) -> str:
    logger.info(f"Suggesting next action for {contact_id}")
    return f"Based on profile of {contact_id}, suggest a follow-up call or offer a discount."

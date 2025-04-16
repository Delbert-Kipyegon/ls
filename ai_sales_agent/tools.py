import requests

def get_lead_score(contact_id: str) -> str:
    # Simulate calling external scoring API
    res = requests.get(f"http://localhost:8001/score/{contact_id}")
    return res.json().get("score", "No score found")

def generate_document(contact_id: str) -> str:
    return f"Document for {contact_id} has been generated!"

def suggest_next_action(contact_id: str) -> str:
    return f"Based on profile of {contact_id}, suggest a follow-up call or offer a discount."

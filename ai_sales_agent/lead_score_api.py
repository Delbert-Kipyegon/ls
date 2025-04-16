from fastapi import FastAPI

api = FastAPI()

@api.get("/score/{contact_id}")
def get_score(contact_id: str):
    return {"score": 87 if contact_id == "CT004491" else 42}

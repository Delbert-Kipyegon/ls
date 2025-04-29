from fastapi import FastAPI
from agent import agent

app = FastAPI()

@app.get("/agent")
def talk_to_agent(input: str):
    return {"response": agent.run(input)}

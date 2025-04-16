from fastapi import FastAPI
from agent import agent
from logger import logger

app = FastAPI()

@app.get("/agent")
def talk_to_agent(input: str):
    logger.info(f"User input: {input}")
    try:
        response = agent.run(input)
        logger.info(f"Agent response: {response}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Agent failed to respond: {e}")
        return {"response": "An error occurred while processing the request."}

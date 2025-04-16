from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from tools import get_lead_score, generate_document, suggest_next_action

llm = ChatOpenAI(temperature=0)

tools = [
    Tool(name="Get Lead Score", func=get_lead_score, description="Fetches lead score for a contact"),
    Tool(name="Generate Document", func=generate_document, description="Creates a document for the user"),
    Tool(name="Suggest Next Action", func=suggest_next_action, description="Suggests follow-up actions for the salesperson"),
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

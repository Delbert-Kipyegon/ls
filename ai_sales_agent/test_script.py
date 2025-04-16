import requests
import time

# Agent URL
AGENT_URL = "http://localhost:8000/agent"

# Test inputs to simulate user requests
test_inputs = [
    "Get lead score for CT004491",
    "Generate document for CT004491",
    "Suggest next action for CT004491",
    "Get lead score for CT000000",
    "Suggest follow-up for lead CT004491"
]

def test_agent():
    for i, input_text in enumerate(test_inputs):
        print(f"\n🔹 Test {i+1}: {input_text}")
        try:
            response = requests.get(AGENT_URL, params={"input": input_text})
            print("✅ Response:", response.json()["response"])
        except Exception as e:
            print("❌ Error:", e)
        time.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    test_agent()

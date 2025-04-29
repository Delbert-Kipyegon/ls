import unittest
from ai_sales_agent.agent_core import SalesConversionAgent
from ai_sales_agent.logger import logger

class TestSalesConversionAgent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.agent = SalesConversionAgent()
        self.test_customer_id = "test_customer_123"
        self.test_customer = {
            "name": "Test Company",
            "industry": "Technology",
            "size": "Mid-Market",
            "interactions": [],
            "objections": [],
            "stage": "prospecting",
            "sentiment_score": 0
        }
        self.agent.customers[self.test_customer_id] = self.test_customer

    def test_analyze_interaction(self):
        """Test interaction analysis"""
        interaction_text = "We're interested in your product but concerned about the implementation timeline."
        result = self.agent.analyze_interaction(self.test_customer_id, interaction_text)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("sentiment", result)
        self.assertIn("topics", result)
        self.assertIn("recommended_actions", result)

    def test_handle_objections(self):
        """Test objection handling"""
        objection_text = "Your product is too expensive for our budget."
        result = self.agent.handle_objections(self.test_customer_id, objection_text)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("objection_type", result)
        self.assertIn("response", result)
        self.assertIn("follow_up", result)

    def test_generate_proposal(self):
        """Test proposal generation"""
        result = self.agent.generate_proposal(self.test_customer_id)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("proposal_content", result)
        self.assertIn("executive_summary", result)
        self.assertIn("pricing_section", result)

if __name__ == '__main__':
    unittest.main() 
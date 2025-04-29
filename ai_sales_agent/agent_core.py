import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
from .logger import logger
load_dotenv()  

class SalesConversionAgent:
    """AI-powered Sales Conversion Assistant for Sales Engineers"""
    
    def __init__(self, crm_data_path=None, use_llm=True):
        """Initialize the agent with CRM data if available"""
        self.customers = {}
        self.use_llm = use_llm
        self.logger = logger
        
        if crm_data_path:
            try:
                # Try to load and process real CRM data
                self._load_real_crm_data(crm_data_path)
            except Exception as e:
                print(f"Error loading CRM data: {e}")
                # Fall back to mock data if real data load fails
                self._load_mock_data()
        else:
            # Use mock data if no CRM data provided
            self._load_mock_data()
            
        # Load templates and other resources
        self._load_resources()
            
    def _load_real_crm_data(self, filepath):
        """Load and process real CRM data from CSV"""
        df = pd.read_csv(filepath)
        
        # Basic data cleaning
        # Convert date and time if possible
        try:
            df['Interaction_DateTime'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time_of_Interaction'], 
                errors='coerce'
            )
        except:
            # If date parsing fails, create a dummy datetime column
            df['Interaction_DateTime'] = pd.NaT
            
        # Use Document_No as Customer_ID for grouping
        df['Customer_ID'] = df['Document_No'].astype(str)
        
        # Process each customer
        for customer_id in df['Customer_ID'].unique():
            customer_data = df[df['Customer_ID'] == customer_id]
            
            # Extract company and contact info
            company_name = customer_data['Contact_Company_Name'].mode().iloc[0] if not customer_data['Contact_Company_Name'].mode().empty else f"Company {customer_id}"
            try:
                contact_name = customer_data['Contact_Name'].mode().iloc[0] if not customer_data['Contact_Name'].mode().empty else "Unknown Contact"
            except:
                contact_name = "Unknown Contact"
            
            # Create customer record
            self.customers[customer_id] = {
                'name': company_name,
                'contact_name': contact_name,
                'industry': self._guess_industry(company_name),
                'size': self._guess_company_size(customer_data),
                'annual_revenue': np.random.randint(500000, 10000000),  # Placeholder
                'interactions': [],
                'objections': [],
                'stage': 'evaluation',  # Default
                'sentiment_score': 0
            }
            
            # Process interactions
            interactions = []
            sentiment_scores = []
            topics_mentioned = set()
            objections = set()
            
            for _, row in customer_data.iterrows():
                # Extract description
                description = row['Description']
                if 'TRIM(WorkDescription)' in row and pd.notna(row['TRIM(WorkDescription)']):
                    description += ": " + row['TRIM(WorkDescription)']
                
                # Map evaluation to sentiment
                sentiment = 'neutral'
                if pd.notna(row['Evaluation']):
                    eval_lower = row['Evaluation'].lower()
                    if any(term in eval_lower for term in ['success', 'good', 'excellent']):
                        sentiment = 'positive'
                        sentiment_scores.append(1)
                    elif any(term in eval_lower for term in ['fail', 'poor', 'rejected']):
                        sentiment = 'negative'
                        sentiment_scores.append(-1)
                    else:
                        sentiment_scores.append(0)
                
                # Extract topics from description
                interaction_topics = self._extract_topics(description)
                topics_mentioned.update(interaction_topics)
                
                # Check for objections
                if sentiment == 'negative':
                    detected_objections = self._detect_objections(description)
                    objections.update(detected_objections)
                
                # Create interaction record
                interaction = {
                    'date': row['Interaction_DateTime'].strftime('%Y-%m-%d') if pd.notna(row['Interaction_DateTime']) else '2025-01-01',
                    'type': row['Document_Type'],
                    'sentiment': sentiment,
                    'topics': interaction_topics,
                    'description': description,
                    'duration': row['Duration_Min'] if 'Duration_Min' in row else 0
                }
                interactions.append(interaction)
            
            # Update customer record
            self.customers[customer_id]['interactions'] = interactions
            self.customers[customer_id]['objections'] = list(objections)
            self.customers[customer_id]['sentiment_score'] = np.mean(sentiment_scores) if sentiment_scores else 0
            self.customers[customer_id]['topics'] = list(topics_mentioned)
            
            # Determine stage based on interactions and sentiment
            if len(interactions) <= 1:
                self.customers[customer_id]['stage'] = 'prospecting'
            elif np.mean(sentiment_scores) < -0.2 if sentiment_scores else False:
                self.customers[customer_id]['stage'] = 'at_risk'
            elif 'pricing' in topics_mentioned or 'price' in topics_mentioned:
                self.customers[customer_id]['stage'] = 'negotiation'
            else:
                self.customers[customer_id]['stage'] = 'evaluation'
    
    def _extract_topics(self, text):
        """Extract topics from interaction text"""
        if not isinstance(text, str):
            return []
            
        text = text.lower()
        topics = []
        
        topic_keywords = {
            'pricing': ['price', 'cost', 'budget', 'expensive', 'afford', 'discount'],
            'features': ['feature', 'functionality', 'capability', 'does it', 'can it'],
            'competition': ['competitor', 'alternative', 'other solution', 'instead'],
            'timeline': ['timeline', 'deadline', 'when can', 'how soon', 'schedule'],
            'support': ['support', 'maintenance', 'help', 'assistance', 'training'],
            'integration': ['integrate', 'connection', 'api', 'work with', 'compatible'],
            'contract': ['contract', 'agreement', 'terms', 'conditions', 'legal']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
                
        return topics
    
    def _detect_objections(self, text):
        """Detect sales objections from text"""
        if not isinstance(text, str):
            return []
            
        text = text.lower()
        objections = []
        
        objection_patterns = {
            'price': ['too expensive', 'high price', 'over budget', 'can\'t afford', 'costly'],
            'implementation_time': ['takes too long', 'slow implementation', 'too much time'],
            'contract_terms': ['contract issue', 'terms problem', 'agreement concern'],
            'features': ['missing feature', 'doesn\'t have', 'lack of'],
            'complexity': ['too complex', 'complicated', 'difficult to use'],
            'authority': ['need approval', 'decision maker', 'not my decision']
        }
        
        for objection_type, patterns in objection_patterns.items():
            if any(pattern in text for pattern in patterns):
                objections.append(objection_type)
                
        return objections
            
    def _guess_industry(self, company_name):
        """Guess industry from company name"""
        if not isinstance(company_name, str):
            return "Unknown"
            
        company_name = company_name.lower()
        
        industry_keywords = {
            'Technology': ['tech', 'software', 'digital', 'data', 'cyber', 'computer', 'it '],
            'Manufacturing': ['manufacturing', 'factory', 'production', 'industrial'],
            'Healthcare': ['health', 'medical', 'care', 'pharma', 'clinic', 'hospital'],
            'Retail': ['retail', 'shop', 'store', 'mart', 'market'],
            'Financial': ['bank', 'finance', 'invest', 'capital', 'financial'],
            'Education': ['edu', 'school', 'university', 'college', 'academy']
        }
        
        for industry, keywords in industry_keywords.items():
            if any(keyword in company_name for keyword in keywords):
                return industry
                
        return "Other"
    
    def _guess_company_size(self, customer_data):
        """Guess company size from interaction data"""
        # Simple heuristic: more interactions or higher costs might indicate larger company
        interaction_count = len(customer_data)
        total_cost = customer_data['Cost_LCY'].sum() if 'Cost_LCY' in customer_data else 0
        
        if interaction_count > 10 or total_cost > 10000:
            return "Enterprise"
        elif interaction_count > 5 or total_cost > 5000:
            return "Mid-Market"
        else:
            return "SMB"
        
    def _load_mock_data(self):
        """Load mock data for demonstration purposes"""
        # Customer data
        self.customers = {
            "customer_123": {
                "name": "Acme Corp",
                "contact_name": "John Smith",
                "industry": "Manufacturing",
                "size": "Enterprise",
                "annual_revenue": 5000000,
                "interactions": [
                    {"date": "2025-03-15", "type": "email", "sentiment": "positive", "topics": ["pricing", "features"], 
                     "description": "Customer inquired about product pricing and key features."},
                    {"date": "2025-03-20", "type": "call", "sentiment": "neutral", "topics": ["competition", "timeline"],
                     "description": "Discussed how our solution compares to competitor X. Customer concerned about implementation timeline."}
                ],
                "objections": ["price", "implementation_time"],
                "stage": "evaluation",
                "sentiment_score": 0.5
            },
            "customer_456": {
                "name": "TechStart Inc",
                "contact_name": "Jane Doe",
                "industry": "Technology",
                "size": "SMB",
                "annual_revenue": 800000,
                "interactions": [
                    {"date": "2025-04-01", "type": "meeting", "sentiment": "positive", "topics": ["features", "support"],
                     "description": "Initial meeting to discuss product features. Customer very interested in support options."},
                    {"date": "2025-04-10", "type": "email", "sentiment": "negative", "topics": ["pricing", "contract"],
                     "description": "Customer expressed concerns about pricing structure and contract requirements."}
                ],
                "objections": ["contract_terms"],
                "stage": "negotiation",
                "sentiment_score": -0.2
            }
        }
        
    def _load_resources(self):
        """Load resources needed by the agent"""
        # Product data
        self.products = {
            "product_a": {
                "name": "Enterprise Solution", 
                "target_industries": ["Manufacturing", "Healthcare"],
                "features": [
                    "Advanced analytics dashboard",
                    "Enterprise-grade security",
                    "24/7 dedicated support",
                    "Custom integration services",
                    "Unlimited users"
                ],
                "base_price": 50000,
                "implementation_time": "4-6 weeks"
            },
            "product_b": {
                "name": "SMB Package", 
                "target_industries": ["Technology", "Retail"],
                "features": [
                    "Essential analytics",
                    "Standard security features",
                    "Business hours support",
                    "API access",
                    "Up to 50 users"
                ],
                "base_price": 10000,
                "implementation_time": "2-3 weeks"
            }
        }
        
        # Sales strategies
        self.strategies = {
            "value_based": {
                "description": "Focus on ROI and long-term value",
                "recommended_for": ["Enterprise", "Mid-Market"],
                "talking_points": [
                    "Long-term cost savings",
                    "Productivity improvements",
                    "Strategic advantage",
                    "Total cost of ownership"
                ]
            },
            "competitive": {
                "description": "Highlight advantages over competitors",
                "recommended_for": ["All"],
                "talking_points": [
                    "Direct feature comparisons",
                    "Market leadership position",
                    "Customer success stories",
                    "Performance benchmarks"
                ]
            },
            "relationship": {
                "description": "Build trust and long-term partnership",
                "recommended_for": ["Enterprise"],
                "talking_points": [
                    "Strategic alignment",
                    "Future roadmap",
                    "Executive sponsorship",
                    "Co-innovation opportunities"
                ]
            },
            "solution": {
                "description": "Focus on solving specific problems",
                "recommended_for": ["SMB", "Mid-Market"],
                "talking_points": [
                    "Pain point resolution",
                    "Quick wins",
                    "Implementation simplicity",
                    "Immediate value delivery"
                ]
            }
        }
        
        # Objection handling templates
        self.objection_responses = {
            "price": [
                "Our solution provides significant ROI through {benefit_1} and {benefit_2}",
                "While initial investment is higher, total cost of ownership is lower due to {reason}",
                "We offer flexible payment options that can align with your budget cycles"
            ],
            "implementation_time": [
                "We can provide a phased implementation approach focusing first on {key_feature}",
                "Our implementation team has experience with similar companies and can expedite the process",
                "We offer a quick-start program that can have you operational within {shorter_time}"
            ],
            "contract_terms": [
                "We can customize certain terms to align with your procurement requirements",
                "Many clients find value in our {specific_term} which provides {benefit}",
                "We can explore alternative agreement structures that meet both our needs"
            ],
            "features": [
                "While that feature isn't currently available, our {alternative_feature} provides similar benefits",
                "That feature is on our near-term roadmap with expected delivery in {timeline}",
                "We can discuss a custom development option for critical functionality"
            ],
            "complexity": [
                "We offer comprehensive training and onboarding to ensure smooth adoption",
                "Our interface is designed based on extensive user testing to simplify complex operations",
                "We provide dedicated support during the first {time_period} to assist with any challenges"
            ],
            "authority": [
                "We'd be happy to prepare materials specifically addressing the concerns of decision makers",
                "We can schedule a brief executive overview session with the key stakeholders",
                "Many clients find our ROI calculator helpful when presenting to decision makers"
            ]
        }
        
        # Document templates
        self.templates = {
            "proposal": "# Sales Proposal for {customer_name}\n\n## Executive Summary\n{executive_summary}\n\n## Understanding Your Needs\n{pain_points}\n\n## Proposed Solution\n{solution_details}\n\n## Why Choose Us\n{differentiators}\n\n## Implementation Approach\n{implementation_details}\n\n## Investment Summary\n{pricing_details}\n\n## Next Steps\n{next_steps}",
            
            "follow_up": "Subject: Follow-up regarding {topic} - {customer_name}\n\nDear {contact_name},\n\nThank you for discussing {topic} with us. As promised, {follow_up_details}.\n\n{objection_response}\n\n{next_steps}\n\nBest regards,\n{sales_engineer_name}"
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process an incoming query from a sales engineer"""
        intent = self._detect_intent(query)
        
        # Process based on intent
        if intent == "analyze_customer":
            customer_id = self._extract_customer_id(query)
            return self._analyze_customer(customer_id)
        
        elif intent == "score_lead":
            customer_id = self._extract_customer_id(query)
            return self._score_lead(customer_id)
        
        elif intent == "recommend_strategy":
            customer_id = self._extract_customer_id(query)
            return self._recommend_strategy(customer_id)
        
        elif intent == "generate_document":
            doc_type = self._extract_document_type(query)
            customer_id = self._extract_customer_id(query)
            return self._generate_document(doc_type, customer_id)
        
        elif intent == "identify_objections":
            customer_id = self._extract_customer_id(query)
            return self._identify_objections(customer_id)
        
        elif intent == "list_customers":
            return self._list_customers()
        
        else:
            return {"response": "I'm not sure how to help with that request. You can ask me to analyze customers, score leads, recommend strategies, generate documents, or identify objections."}
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the sales engineer's query"""
        query = query.lower()
        
        # Simple keyword matching for demo purposes
        if any(term in query for term in ["analyze", "review", "examine", "look at", "check", "tell me about"]):
            return "analyze_customer"
        
        elif any(term in query for term in ["score", "qualify", "rate", "evaluate", "what's the score"]):
            return "score_lead"
        
        elif any(term in query for term in ["recommend", "suggest", "strategy", "approach", "best way"]):
            return "recommend_strategy"
        
        elif any(term in query for term in ["generate", "create", "document", "proposal", "email"]):
            return "generate_document"
        
        elif any(term in query for term in ["objection", "concern", "hesitation", "pushback", "obstacle"]):
            return "identify_objections"
            
        elif any(term in query for term in ["list", "show", "all customers", "companies"]):
            return "list_customers"
        
        return "unknown"
    
    def _extract_customer_id(self, query: str) -> str:
        """Extract customer ID from query"""
        query_lower = query.lower()
        
        # Match customer by name mentions in the query
        for customer_id, customer in self.customers.items():
            if customer['name'].lower() in query_lower:
                return customer_id
        
        # If specific customer mentioned in the query can't be found
        # Just return the first customer ID for demo purposes
        if self.customers:
            return list(self.customers.keys())[0]
        
        return ""
    
    def _extract_document_type(self, query: str) -> str:
        """Extract document type from query"""
        query_lower = query.lower()
        
        if "proposal" in query_lower:
            return "proposal"
        elif any(term in query_lower for term in ["follow", "email", "message"]):
            return "follow_up"
        
        # Default
        return "proposal"
    
    def _list_customers(self) -> Dict[str, Any]:
        """List all available customers"""
        customer_list = []
        
        for customer_id, customer in self.customers.items():
            customer_list.append({
                "id": customer_id,
                "name": customer["name"],
                "industry": customer["industry"],
                "stage": customer["stage"],
                "interactions": len(customer["interactions"])
            })
            
        return {
            "customer_count": len(customer_list),
            "customers": customer_list
        }
    
    def _analyze_customer(self, customer_id: str) -> Dict[str, Any]:
        """Analyze customer interactions and data"""
        if customer_id not in self.customers:
            return {"error": "Customer not found"}
        
        customer = self.customers[customer_id]
        
        # Count topic mentions
        topic_count = {}
        for interaction in customer["interactions"]:
            for topic in interaction["topics"]:
                topic_count[topic] = topic_count.get(topic, 0) + 1
        
        # Generate insights
        insights = []
        
        # Sentiment-based insights
        if customer["sentiment_score"] > 0.5:
            insights.append("Customer shows positive engagement, good opportunity for upselling")
        elif customer["sentiment_score"] < -0.2:
            insights.append("Customer shows signs of dissatisfaction, focus on addressing concerns")
        
        # Topic-based insights
        for topic, count in topic_count.items():
            if count >= 2:
                insights.append(f"{topic.capitalize()} is a recurring theme, consider focusing on this area")
        
        # Stage-based insights
        if customer["stage"] == "negotiation":
            insights.append("Customer is in negotiation stage, consider focusing on value proposition")
        
        # Objection-based insights
        if customer["objections"]:
            objection_text = ", ".join(customer["objections"])
            insights.append(f"Address key objections: {objection_text}")
        
        # Use LLM to generate additional insights if enabled
        if self.use_llm:
            llm_insights = self._get_llm_insights(customer)
            if llm_insights:
                insights.extend(llm_insights)
        
        return {
            "customer_name": customer["name"],
            "industry": customer["industry"],
            "size": customer["size"],
            "current_stage": customer["stage"],
            "sentiment_analysis": {
                "score": round(customer["sentiment_score"], 2),
                "interpretation": "Positive" if customer["sentiment_score"] > 0 else "Negative" if customer["sentiment_score"] < 0 else "Neutral"
            },
            "key_topics": sorted(topic_count.items(), key=lambda x: x[1], reverse=True),
            "recent_interactions": sorted(customer["interactions"], key=lambda x: x["date"], reverse=True)[:3],
            "insights": insights,
            "recommended_actions": self._generate_recommended_actions(customer)
        }
    
    def _score_lead(self, customer_id: str) -> Dict[str, Any]:
        """Score a lead based on various factors"""
        if customer_id not in self.customers:
            return {"error": "Customer not found"}
        
        customer = self.customers[customer_id]
        
        # Scoring factors with weights
        factors = {
            "size": {
                "weight": 0.2,
                "score": {"Enterprise": 1.0, "Mid-Market": 0.7, "SMB": 0.4}
            },
            "industry_fit": {
                "weight": 0.15,
                "score": self._calculate_industry_fit(customer["industry"])
            },
            "engagement": {
                "weight": 0.25,
                "score": min(len(customer["interactions"]) / 5, 1.0)  # Cap at 5 interactions
            },
            "sentiment": {
                "weight": 0.2,
                "score": (customer["sentiment_score"] + 1) / 2  # Convert from [-1,1] to [0,1]
            },
            "stage": {
                "weight": 0.2,
                "score": {"prospecting": 0.2, "qualification": 0.4, "evaluation": 0.6, "negotiation": 0.8, "closing": 1.0, "at_risk": 0.3}
            }
        }
        
        # Calculate weighted score
        score = 0
        factor_scores = {}
        
        for factor, details in factors.items():
            if factor == "size":
                factor_value = details["score"].get(customer["size"], 0.5)
            elif factor == "stage":
                factor_value = details["score"].get(customer["stage"], 0.5)
            else:
                factor_value = details["score"]
                
            factor_scores[factor] = factor_value
            score += factor_value * details["weight"]
        
        # Normalize to 0-100
        normalized_score = int(score * 100)
        
        # Generate reasoning
        reasoning = []
        for factor, factor_score in factor_scores.items():
            if factor == "size":
                reasoning.append(f"{customer['size']}-sized company ({int(factor_score * 100)} points)")
            elif factor == "industry_fit":
                reasoning.append(f"Industry fit: {int(factor_score * 100)}%")
            elif factor == "engagement":
                reasoning.append(f"Engagement level: {len(customer['interactions'])} interactions ({int(factor_score * 100)} points)")
            elif factor == "sentiment":
                sentiment_text = "Positive" if customer["sentiment_score"] > 0 else "Negative" if customer["sentiment_score"] < 0 else "Neutral"
                reasoning.append(f"Customer sentiment: {sentiment_text} ({int(factor_score * 100)} points)")
            elif factor == "stage":
                reasoning.append(f"Sales stage: {customer['stage']} ({int(factor_score * 100)} points)")
        
        # Generate next steps based on score
        if normalized_score >= 70:
            category = "Hot Lead"
            next_steps = ["Schedule a demo", "Prepare customized proposal", "Involve solution architect"]
        elif normalized_score >= 40:
            category = "Warm Lead"
            next_steps = ["Send additional information", "Schedule follow-up call", "Address identified objections"]
        else:
            category = "Cold Lead"
            next_steps = ["Nurture with marketing content", "Re-engage in 3 months", "Consider different solution package"]
        
        return {
            "customer_name": customer["name"],
            "score": normalized_score,
            "category": category,
            "reasoning": reasoning,
            "next_steps": next_steps
        }
    
    def _calculate_industry_fit(self, industry: str) -> float:
        """Calculate how well the industry fits with available products"""
        product_industries = set()
        for product in self.products.values():
            product_industries.update(product["target_industries"])
            
        if industry in product_industries:
            return 1.0
        elif industry == "Other":
            return 0.5
        else:
            return 0.3
    
    def _generate_recommended_actions(self, customer: Dict[str, Any]) -> List[str]:
        """Generate recommended actions based on customer data"""
        recommendations = []
        
        # Stage-based recommendations
        stage_recommendations = {
            "prospecting": ["Send company overview", "Propose initial discovery call"],
            "qualification": ["Arrange technical demo", "Provide case studies for similar companies"],
            "evaluation": ["Schedule solution deep-dive", "Arrange stakeholder meeting"],
            "negotiation": ["Prepare ROI analysis", "Discuss implementation timeline"],
            "closing": ["Draft final agreement", "Introduce implementation team"],
            "at_risk": ["Escalate to senior representative", "Address outstanding concerns"]
        }
        
        if customer["stage"] in stage_recommendations:
            recommendations.extend(stage_recommendations[customer["stage"]])
        
        # Add objection-specific recommendations
        if customer["objections"]:
            for objection in customer["objections"][:2]:  # Limit to top 2 objections
                if objection == "price":
                    recommendations.append("Prepare value-based ROI analysis")
                elif objection == "implementation_time":
                    recommendations.append("Propose phased implementation plan")
                elif objection == "contract_terms":
                    recommendations.append("Review contract flexibility options")
                elif objection == "features":
                    recommendations.append("Highlight alternative capabilities or roadmap")
                elif objection == "complexity":
                    recommendations.append("Arrange simplified product demonstration")
                elif objection == "authority":
                    recommendations.append("Prepare executive summary for decision makers")
        
        # Add sentiment-based recommendations
        if customer["sentiment_score"] < -0.2:
            recommendations.append("Schedule relationship repair meeting")
        elif customer["sentiment_score"] > 0.5:
            recommendations.append("Explore upselling opportunities")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _recommend_strategy(self, customer_id: str) -> Dict[str, Any]:
        """Recommend a sales strategy for the customer"""
        if customer_id not in self.customers:
            return {"error": "Customer not found"}
        
        customer = self.customers[customer_id]
        
        # Choose primary strategy based on customer characteristics
        primary_strategy = self._select_primary_strategy(customer)
        
        # Choose secondary strategy as complementary approach
        secondary_strategy = self._select_secondary_strategy(customer, primary_strategy)
        
        # Generate talking points
        talking_points = self._generate_talking_points(customer, primary_strategy, secondary_strategy)
        
        # Generate product recommendations
        recommended_product = self._recommend_product(customer)
        
        # Generate custom tips
        custom_tips = self._generate_strategy_tips(customer, primary_strategy)
        
        # Use LLM for additional personalized strategy if enabled
        llm_strategy = None
        if self.use_llm:
            llm_strategy = self._get_llm_strategy(customer, primary_strategy, secondary_strategy)
        
        return {
            "customer_name": customer["name"],
            "primary_strategy": {
                "name": primary_strategy,
                "description": self.strategies[primary_strategy]["description"]
            },
            "secondary_strategy": {
                "name": secondary_strategy,
                "description": self.strategies[secondary_strategy]["description"]
            },
            "talking_points": talking_points,
            "recommended_product": recommended_product,
            "custom_tips": custom_tips,
            "ai_strategy_recommendation": llm_strategy
        }
    
    def _select_primary_strategy(self, customer: Dict[str, Any]) -> str:
        """Select the primary sales strategy based on customer attributes"""
        # Default strategy
        default_strategy = "value_based"
        
        # Check if there's a clear strategy match based on customer size
        size_strategy_map = {
            "Enterprise": ["value_based", "relationship"],
            "Mid-Market": ["value_based", "solution"],
            "SMB": ["solution", "competitive"]
        }
        
        potential_strategies = size_strategy_map.get(customer["size"], ["value_based", "solution"])
        
        # Refine based on customer stage
        if customer["stage"] == "negotiation":
            # In negotiation, value-based approaches often work better
            if "value_based" in potential_strategies:
                return "value_based"
            else:
                return potential_strategies[0]
        elif customer["stage"] == "at_risk":
            # For at-risk customers, relationship-building can be critical
            return "relationship"
        elif "competition" in customer.get("topics", []):
            # If competition is a frequent topic, use competitive strategy
            return "competitive"
        
        # Otherwise use the first strategy for customer size
        return potential_strategies[0]
    
    def _select_secondary_strategy(self, customer: Dict[str, Any], primary_strategy: str) -> str:
        """Select a complementary secondary strategy"""
        # Do not select the same strategy as primary
        available_strategies = [s for s in self.strategies.keys() if s != primary_strategy]
        
        # For enterprises, relationship is important if not already primary
        if customer["size"] == "Enterprise" and primary_strategy != "relationship":
            return "relationship"
        
        # For customers with price objections, value-based is helpful if not already primary
        if "price" in customer.get("objections", []) and primary_strategy != "value_based":
            return "value_based"
            
        # For customers in negotiation, competitive can be useful if not already primary
        if customer["stage"] == "negotiation" and primary_strategy != "competitive":
            return "competitive"
            
        # For SMBs, solution-focused is often helpful if not already primary
        if customer["size"] == "SMB" and primary_strategy != "solution":
            return "solution"
            
        # Default to the first available non-primary strategy
        return available_strategies[0]
    
    def _generate_talking_points(self, customer: Dict[str, Any], primary_strategy: str, secondary_strategy: str) -> List[str]:
        """Generate talking points based on selected strategies and customer data"""
        talking_points = []
        
        # Add points from primary strategy
        if primary_strategy in self.strategies:
            talking_points.extend(self.strategies[primary_strategy]["talking_points"])
            
        # Add points from secondary strategy (limit to 2)
        if secondary_strategy in self.strategies:
            talking_points.extend(self.strategies[secondary_strategy]["talking_points"][:2])
            
        # Add customer-specific talking points
        if "price" in customer.get("objections", []):
            talking_points.append("Cost-benefit analysis showing ROI within first year")
            
        if "implementation_time" in customer.get("objections", []):
            talking_points.append("Phased implementation approach with quick early wins")
            
        if customer["industry"] == "Healthcare":
            talking_points.append("HIPAA compliance and security features")
            
        if customer["industry"] == "Financial":
            talking_points.append("Regulatory compliance and audit trail capabilities")
            
        if customer["size"] == "Enterprise":
            talking_points.append("Enterprise-wide scalability and integration")
            
        # Remove duplicates and limit to 6 total points
        unique_points = []
        for point in talking_points:
            if point not in unique_points:
                unique_points.append(point)
                
        return unique_points[:6]
    
    def _recommend_product(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend the most suitable product based on customer attributes"""
        # Simple product recommendation logic based on company size and industry
        product_id = None
        
        # Size-based recommendation
        if customer["size"] == "Enterprise":
            product_id = "product_a"  # Enterprise Solution
        else:
            product_id = "product_b"  # SMB Package
            
        # Check if industry aligns with product target industries
        if product_id and customer["industry"] in self.products[product_id]["target_industries"]:
            confidence = "high"
        else:
            confidence = "medium"
            
        # Add special notes based on customer attributes
        special_notes = []
        
        if customer["size"] == "Enterprise" and product_id == "product_b":
            special_notes.append("Consider enterprise upgrades to meet scale requirements")
            
        if customer["size"] == "SMB" and product_id == "product_a":
            special_notes.append("May be oversized for current needs - consider starting with basic package")
            
        if "features" in customer.get("objections", []) and product_id == "product_b":
            special_notes.append("May need feature customization to address specific requirements")
            
        # Return recommendation data
        if product_id:
            product = self.products[product_id]
            return {
                "product_name": product["name"],
                "confidence": confidence,
                "key_features": product["features"][:3],  # Top 3 features
                "base_price": product["base_price"],
                "implementation_time": product["implementation_time"],
                "special_notes": special_notes
            }
        else:
            return {
                "product_name": "Custom Solution",
                "confidence": "low",
                "key_features": ["To be determined based on requirements"],
                "base_price": "Custom",
                "implementation_time": "Variable",
                "special_notes": ["Needs detailed discovery to determine exact requirements"]
            }
    
    def _generate_strategy_tips(self, customer: Dict[str, Any], primary_strategy: str) -> List[str]:
        """Generate custom strategy tips based on customer data and selected strategy"""
        tips = []
        
        # General tips based on customer stage
        stage_tips = {
            "prospecting": ["Focus on establishing credibility", "Research the customer's business challenges"],
            "qualification": ["Confirm budget and timeline expectations", "Identify all decision makers"],
            "evaluation": ["Provide proof points specific to their industry", "Address technical concerns early"],
            "negotiation": ["Be prepared with pricing flexibility options", "Focus on value, not features"],
            "closing": ["Prepare for implementation questions", "Discuss success metrics"],
            "at_risk": ["Escalate to management if needed", "Consider concessions on key sticking points"]
        }
        
        if customer["stage"] in stage_tips:
            tips.extend(stage_tips[customer["stage"]])
        
        # Strategy-specific tips
        if primary_strategy == "value_based":
            tips.append(f"Prepare ROI calculator specific to {customer['industry']} industry")
            
        elif primary_strategy == "competitive":
            tips.append("Research which competitors they're considering")
            
        elif primary_strategy == "relationship":
            tips.append("Identify executive sponsor within your organization")
            
        elif primary_strategy == "solution":
            tips.append("Map specific product capabilities to their stated challenges")
        
        # Customer-specific tips
        if customer["sentiment_score"] < 0:
            tips.append("Address negative sentiment before pushing too hard on next steps")
            
        if len(customer["interactions"]) <= 2:
            tips.append("Still early in relationship - focus on building trust")
            
        if customer["size"] == "Enterprise" and customer["stage"] in ["evaluation", "negotiation"]:
            tips.append("Be prepared for procurement and legal team involvement")
            
        return tips[:5]  # Limit to 5 tips
    
    def _identify_objections(self, customer_id: str) -> Dict[str, Any]:
        """Identify and analyze customer objections"""
        if customer_id not in self.customers:
            return {"error": "Customer not found"}
        
        customer = self.customers[customer_id]
        
        objections = customer.get("objections", [])
        responses = {}
        evidence = {}
        priority = {}
        
        # If no objections detected, try to infer potential objections
        if not objections:
            potential_objections = self._infer_potential_objections(customer)
            if potential_objections:
                result = {
                    "customer_name": customer["name"],
                    "detected_objections": [],
                    "potential_objections": [{
                        "type": obj_type,
                        "reason": reason,
                        "suggested_response": self._get_objection_response(obj_type, customer)
                    } for obj_type, reason in potential_objections.items()]
                }
                return result
            else:
                return {
                    "customer_name": customer["name"],
                    "message": "No objections detected or inferred",
                    "detected_objections": [],
                    "potential_objections": []
                }
        
        # Process each objection
        for obj in objections:
            # Find evidence in interactions
            evidence[obj] = self._find_objection_evidence(customer, obj)
            
            # Get response templates
            responses[obj] = self._get_objection_response(obj, customer)
            
            # Assign priority based on frequency and recency
            priority[obj] = self._calculate_objection_priority(customer, obj)
        
        # Use LLM to generate personalized responses if enabled
        llm_responses = {}
        if self.use_llm:
            llm_responses = self._get_llm_objection_responses(customer, objections)
        
        # Sort objections by priority
        prioritized_objections = sorted(
            [(obj, priority[obj]) for obj in objections], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Format result
        result = {
            "customer_name": customer["name"],
            "detected_objections": [{
                "type": obj_type,
                "priority": priority,
                "evidence": evidence[obj_type][:2],  # Limit to top 2 pieces of evidence
                "suggested_response": responses[obj_type],
                "ai_response": llm_responses.get(obj_type, None)
            } for obj_type, priority in prioritized_objections]
        }
        
        return result
    
    def _infer_potential_objections(self, customer: Dict[str, Any]) -> Dict[str, str]:
        """Infer potential objections based on customer characteristics"""
        potential_objections = {}
        
        # Size-based potential objections
        if customer["size"] == "Enterprise":
            potential_objections["implementation_time"] = "Enterprise customers often have concerns about implementation complexity"
            potential_objections["integration"] = "Enterprise customers typically need extensive integration with existing systems"
        
        elif customer["size"] == "SMB":
            potential_objections["price"] = "SMB customers are often price-sensitive"
            potential_objections["support"] = "SMB customers may worry about adequate support resources"
        
        # Industry-based potential objections
        if customer["industry"] == "Healthcare":
            potential_objections["compliance"] = "Healthcare companies typically have strict compliance requirements"
            
        elif customer["industry"] == "Financial":
            potential_objections["security"] = "Financial companies are particularly concerned with security"
        
        # Stage-based potential objections
        if customer["stage"] == "evaluation":
            potential_objections["features"] = "Customers in evaluation stage often compare features with competitors"
            
        elif customer["stage"] == "negotiation":
            potential_objections["contract_terms"] = "Negotiation stage often involves contract term discussions"
        
        # Sentiment-based potential objections
        if customer["sentiment_score"] < 0:
            potential_objections["satisfaction"] = "Customer sentiment is negative, suggesting general dissatisfaction"
        
        return potential_objections
    
    def _find_objection_evidence(self, customer: Dict[str, Any], objection_type: str) -> List[str]:
        """Find evidence of objection in customer interactions"""
        evidence = []
        
        for interaction in customer["interactions"]:
            description = interaction.get("description", "")
            objection_keywords = {
                "price": ["expensive", "cost", "price", "budget", "afford"],
                "implementation_time": ["time", "timeline", "long", "delay", "quick"],
                "contract_terms": ["contract", "terms", "agreement", "legal"],
                "features": ["feature", "functionality", "capability", "missing"],
                "complexity": ["complex", "complicated", "difficult", "confusing"],
                "authority": ["approval", "decision", "authorize", "boss", "committee"]
            }
            
            # Check if the interaction description contains objection keywords
            if objection_type in objection_keywords:
                keywords = objection_keywords[objection_type]
                if any(keyword in description.lower() for keyword in keywords):
                    evidence.append(f"{interaction['date']} ({interaction['type']}): {description}")
        
        return evidence
    
    def _get_objection_response(self, objection_type: str, customer: Dict[str, Any]) -> str:
        """Get appropriate response template for an objection"""
        if objection_type not in self.objection_responses:
            return "No standard response available for this objection type."
        
        # Get templates for this objection type
        templates = self.objection_responses[objection_type]
        
        # Choose the most appropriate template based on customer attributes
        if objection_type == "price":
            if customer["size"] == "Enterprise":
                template = templates[1]  # Total cost of ownership
                return template.replace("{reason}", "decreased maintenance costs and higher productivity")
            else:
                template = templates[2]  # Flexible payment
                return template
        
        elif objection_type == "implementation_time":
            if customer["size"] == "Enterprise":
                template = templates[0]  # Phased approach
                return template.replace("{key_feature}", "core functionality")
            else:
                template = templates[2]  # Quick start
                return template.replace("{shorter_time}", "2 weeks")
        
        elif objection_type == "features":
            template = templates[1]  # Roadmap
            return template.replace("{timeline}", "Q3 2025")
        
        # Default to first template for other objection types
        return templates[0]
    
    def _calculate_objection_priority(self, customer: Dict[str, Any], objection_type: str) -> float:
        """Calculate the priority of an objection based on frequency and recency"""
        priority = 0.0
        recent_interaction_count = 0
        recent_objection_count = 0
        
        # Sort interactions by date
        sorted_interactions = sorted(
            customer["interactions"], 
            key=lambda x: x["date"], 
            reverse=True
        )
        
        # Get the count of recent interactions (last 3)
        recent_interactions = sorted_interactions[:3]
        recent_interaction_count = len(recent_interactions)
        
        # Check for objection evidence in recent interactions
        objection_keywords = {
            "price": ["expensive", "cost", "price", "budget", "afford"],
            "implementation_time": ["time", "timeline", "long", "delay", "quick"],
            "contract_terms": ["contract", "terms", "agreement", "legal"],
            "features": ["feature", "functionality", "capability", "missing"],
            "complexity": ["complex", "complicated", "difficult", "confusing"],
            "authority": ["approval", "decision", "authorize", "boss", "committee"]
        }
        
        if objection_type in objection_keywords:
            keywords = objection_keywords[objection_type]
            for interaction in recent_interactions:
                description = interaction.get("description", "").lower()
                if any(keyword in description for keyword in keywords):
                    recent_objection_count += 1
        
        # Calculate priority based on recency and frequency
        if recent_interaction_count > 0:
            recency_factor = recent_objection_count / recent_interaction_count
        else:
            recency_factor = 0
            
        # Adjust priority based on customer stage
        stage_factors = {
            "negotiation": 1.5,  # Objections in negotiation stage are critical
            "evaluation": 1.2,   # Objections in evaluation are important
            "at_risk": 2.0,      # Objections for at-risk customers are highest priority
            "prospecting": 0.7   # Objections in early stages are less urgent
        }
        
        stage_factor = stage_factors.get(customer["stage"], 1.0)
        
        # Calculate final priority
        priority = (recency_factor * 0.7 + 0.3) * stage_factor
        
        # Ensure priority is between 0 and 10
        return min(max(priority * 5, 1), 10)
    
    def _generate_document(self, doc_type: str, customer_id: str) -> Dict[str, Any]:
        """Generate a sales document based on customer data"""
        if customer_id not in self.customers:
            return {"error": "Customer not found"}
        
        customer = self.customers[customer_id]
        
        if doc_type == "proposal":
            return self._generate_proposal(customer)
        elif doc_type == "follow_up":
            return self._generate_follow_up(customer)
        else:
            return {"error": f"Unknown document type: {doc_type}"}
    
    def _generate_proposal(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a sales proposal for a customer"""
        # Recommend product
        product_rec = self._recommend_product(customer)
        product_name = product_rec["product_name"]
        
        # Get product details
        product_id = None
        for pid, product in self.products.items():
            if product["name"] == product_name:
                product_id = pid
                break
        
        if not product_id:
            product_id = "product_a" if customer["size"] == "Enterprise" else "product_b"
        
        product = self.products[product_id]
        
        # Generate executive summary
        executive_summary = f"This proposal outlines how {product['name']} can address {customer['name']}'s specific needs in the {customer['industry']} industry. Based on our discussions, we've tailored a solution that addresses your key challenges while providing excellent value and ROI."
        
        # Identify pain points based on interactions and objections
        pain_points = self._identify_pain_points(customer)
        pain_points_text = "Based on our conversations, we understand that you are facing the following challenges:\n\n"
        for i, point in enumerate(pain_points, 1):
            pain_points_text += f"{i}. {point}\n"
        
        # Generate solution details
        solution_details = f"Our {product['name']} provides a comprehensive solution tailored to your needs:\n\n"
        for feature in product["features"]:
            solution_details += f"- {feature}\n"
        
        # Generate differentiators
        differentiators = self._generate_differentiators(customer)
        differentiators_text = "We stand apart from other solutions in the following ways:\n\n"
        for diff in differentiators:
            differentiators_text += f"- {diff}\n"
        
        # Generate implementation details
        implementation_details = f"Our implementation approach for {customer['name']} includes:\n\n"
        implementation_details += "1. Initial discovery and requirements validation (1-2 weeks)\n"
        implementation_details += "2. Solution configuration and customization (2-3 weeks)\n"
        implementation_details += "3. Integration with existing systems (1-2 weeks)\n"
        implementation_details += "4. User acceptance testing and training (1 week)\n"
        implementation_details += f"5. Go-live and post-implementation support ({product['implementation_time']})\n"
        
        # Generate pricing details
        base_price = product["base_price"]
        discount = 0
        
        # Apply potential discounts based on customer characteristics
        if customer["size"] == "Enterprise":
            discount = 0.05  # 5% enterprise discount
        
        if "price" in customer.get("objections", []):
            discount += 0.03  # Additional 3% if price is an objection
            
        final_price = base_price * (1 - discount)
        
        pricing_details = f"Investment Summary:\n\n"
        pricing_details += f"- Base {product['name']}: ${base_price:,}\n"
        
        if discount > 0:
            pricing_details += f"- Applied discount: ${base_price * discount:,.2f} ({discount*100:.0f}%)\n"
            
        pricing_details += f"- Total investment: ${final_price:,.2f}\n\n"
        pricing_details += "Optional add-ons:\n"
        pricing_details += "- Premium support package: $5,000/year\n"
        pricing_details += "- Additional user licenses: $500 per user/year\n"
        
        # Generate next steps
        next_steps = "Recommended next steps:\n\n"
        next_steps += "1. Review proposal and provide feedback\n"
        next_steps += "2. Schedule technical deep-dive session\n"
        next_steps += "3. Finalize contract details\n"
        next_steps += "4. Kick off implementation planning\n"
        
        # Fill template
        template = self.templates["proposal"]
        proposal_content = template.format(
            customer_name=customer["name"],
            executive_summary=executive_summary,
            pain_points=pain_points_text,
            solution_details=solution_details,
            differentiators=differentiators_text,
            implementation_details=implementation_details,
            pricing_details=pricing_details,
            next_steps=next_steps
        )
        
        # Use LLM to enhance proposal if enabled
        if self.use_llm:
            enhanced_content = self._get_llm_enhanced_document(proposal_content, "proposal", customer)
            if enhanced_content:
                proposal_content = enhanced_content
        
        return {
            "document_type": "proposal",
            "customer_name": customer["name"],
            "content": proposal_content,
            "recommended_product": product["name"],
            "pricing": {
                "base_price": base_price,
                "discount_percentage": discount * 100,
                "final_price": final_price
            }
        }
    
    def _identify_pain_points(self, customer: Dict[str, Any]) -> List[str]:
        """Identify customer pain points based on interactions and objections"""
        pain_points = []
        
        # Map topics and objections to potential pain points
        topic_pain_map = {
            "pricing": "Cost management and budget constraints",
            "features": "Need for specific functionality to improve operations",
            "competition": "Competitive pressure requiring improved capabilities",
            "timeline": "Tight timeline for implementation and deployment",
            "support": "Concerns about ongoing support and maintenance",
            "integration": "Seamless integration with existing systems",
            "contract": "Flexibility in contract terms and conditions"
        }
        
        objection_pain_map = {
            "price": "Budget limitations requiring cost-effective solutions",
            "implementation_time": "Need for rapid deployment to meet business deadlines",
            "contract_terms": "Specific contractual requirements to meet company policies",
            "features": "Feature gaps in current solutions affecting productivity",
            "complexity": "Need for intuitive, easy-to-use systems to ensure adoption",
            "authority": "Complex decision-making process requiring consensus"
        }
        
        # Add pain points based on topics in interactions
        topics = set()
        for interaction in customer["interactions"]:
            topics.update(interaction.get("topics", []))
            
        for topic in topics:
            if topic in topic_pain_map:
                pain_points.append(topic_pain_map[topic])
        
        # Add pain points based on objections
        for objection in customer.get("objections", []):
            if objection in objection_pain_map and objection_pain_map[objection] not in pain_points:
                pain_points.append(objection_pain_map[objection])
        
        # Add industry-specific pain points
        industry_pain_points = {
            "Technology": "Rapid innovation cycles requiring scalable solutions",
            "Manufacturing": "Operational efficiency and production optimization",
            "Healthcare": "Regulatory compliance and patient data security",
            "Retail": "Customer experience enhancement and inventory management",
            "Financial": "Data security and regulatory reporting requirements",
            "Education": "Resource optimization and student experience improvement"
        }
        
        if customer["industry"] in industry_pain_points:
            pain_points.append(industry_pain_points[customer["industry"]])
        
        # Add size-specific pain points
        size_pain_points = {
            "Enterprise": "Enterprise-wide standardization and governance",
            "Mid-Market": "Growth scalability and resource optimization",
            "SMB": "Maximizing impact with limited resources"
        }
        
        if customer["size"] in size_pain_points:
            pain_points.append(size_pain_points[customer["size"]])
            
        return pain_points[:5]  # Limit to top 5 pain points
    
    def _generate_differentiators(self, customer: Dict[str, Any]) -> List[str]:
        """Generate differentiators based on customer characteristics"""
        differentiators = [
            "Industry-leading customer satisfaction rates",
            "Seamless integration with existing systems",
            "Regular product updates based on customer feedback"
        ]
        
        # Add industry-specific differentiators
        industry_differentiators = {
            "Technology": "Built by technologists for technology companies",
            "Manufacturing": "Specialized modules for production optimization",
            "Healthcare": "HIPAA-compliant and designed for healthcare workflows",
            "Retail": "Retail-specific analytics and customer insights",
            "Financial": "SOC 2 compliant with financial institution security",
            "Education": "Education-specific templates and workflows"
        }
        
        if customer["industry"] in industry_differentiators:
            differentiators.append(industry_differentiators[customer["industry"]])
        
        # Add size-specific differentiators
        size_differentiators = {
            "Enterprise": "Enterprise-grade scalability supporting thousands of users",
            "Mid-Market": "Perfect balance of features and cost for growing companies",
            "SMB": "Affordable solutions with enterprise-level capabilities"
        }
        
        if customer["size"] in size_differentiators:
            differentiators.append(size_differentiators[customer["size"]])
            
        # Add objection-countering differentiators
        objection_differentiators = {
            "price": "Industry-leading ROI with typical payback in under 12 months",
            "implementation_time": "Fastest implementation timeline in the industry",
            "contract_terms": "Flexible contract options tailored to your needs",
            "features": "Most comprehensive feature set with regular updates",
            "complexity": "Intuitive interface with minimal training required",
            "authority": "Phased adoption approach to minimize organizational disruption"
        }
        
        for objection in customer.get("objections", []):
            if objection in objection_differentiators:
                differentiators.append(objection_differentiators[objection])
                
        return differentiators[:5]  # Limit to top 5 differentiators
    
    def _generate_follow_up(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a follow-up email for a customer"""
        # Identify most recent interaction
        recent_interactions = sorted(customer["interactions"], key=lambda x: x["date"], reverse=True)
        
        # If no interactions, use a generic follow-up
        if not recent_interactions:
            topic = "our initial discussion"
            follow_up_details = "I wanted to touch base regarding our solutions and how they might help your business"
        else:
            # Use the most recent interaction
            recent = recent_interactions[0]
            
            # Determine the topic based on the interaction
            if recent["topics"]:
                topic = recent["topics"][0]  # Use the first topic
            else:
                topic = "our recent discussion"
                
            # Generate follow-up details
            if topic == "pricing":
                follow_up_details = "I'm sharing additional pricing information and ROI analysis as discussed"
            elif topic == "features":
                follow_up_details = "I wanted to provide more details about the features we discussed"
            elif topic == "competition":
                follow_up_details = "I've attached a detailed comparison with competing solutions"
            elif topic == "timeline":
                follow_up_details = "I wanted to address your timeline concerns with our implementation schedule"
            elif topic == "support":
                follow_up_details = "I'm sharing our support options and SLA details"
            elif topic == "integration":
                follow_up_details = "I wanted to provide more information about our integration capabilities"
            elif topic == "contract":
                follow_up_details = "I wanted to address the contract questions you raised"
            else:
                follow_up_details = "I wanted to follow up on the key points from our conversation"
        
        # Generate objection response if applicable
        objection_response = ""
        if customer.get("objections"):
            # Get the primary objection
            primary_objection = customer["objections"][0]
            
            # Get the template response
            objection_template = self._get_objection_response(primary_objection, customer)
            
            # Add to email
            objection_response = f"Regarding your concerns about {primary_objection}, {objection_template}\n\n"
        
        # Generate next steps
        if customer["stage"] == "prospecting":
            next_steps = "I'd like to suggest a brief discovery call to better understand your specific needs. Would you have 30 minutes available next week?"
        elif customer["stage"] == "evaluation":
            next_steps = "I'd be happy to schedule a product demo focusing on the specific capabilities we discussed. Would that be helpful for your evaluation process?"
        elif customer["stage"] == "negotiation":
            next_steps = "To move forward, I've attached a draft proposal for your review. I'm available to discuss any questions you might have about the terms or pricing."
        elif customer["stage"] == "at_risk":
            next_steps = "I understand you may have some concerns. I'd appreciate the opportunity to discuss these in more detail to ensure our solution is the right fit for your needs."
        else:
            next_steps = "I'd be happy to schedule a follow-up call to discuss next steps. Please let me know what would work best for your schedule."
        
        # Fill template
        template = self.templates["follow_up"]
        email_content = template.format(
            topic=topic,
            customer_name=customer["name"],
            contact_name=customer.get("contact_name", "Valued Customer"),
            follow_up_details=follow_up_details,
            objection_response=objection_response,
            next_steps=next_steps,
            sales_engineer_name="Alex Sales Engineer"
        )
        
        # Use LLM to enhance email if enabled
        if self.use_llm:
            enhanced_content = self._get_llm_enhanced_document(email_content, "follow_up", customer)
            if enhanced_content:
                email_content = enhanced_content
        
        return {
            "document_type": "follow_up",
            "customer_name": customer["name"],
            "subject": f"Follow-up regarding {topic} - {customer['name']}",
            "content": email_content,
            "topic": topic,
            "primary_objection": customer["objections"][0] if customer.get("objections") else None
        }
    
    def _get_llm_insights(self, customer: Dict[str, Any]) -> List[str]:
        """Use LLM to generate additional customer insights"""
        if not self.use_llm:
            return []
            
        try:
            # Prepare customer data for LLM
            customer_summary = {
                "name": customer["name"],
                "industry": customer["industry"],
                "size": customer["size"],
                "stage": customer["stage"],
                "sentiment": customer["sentiment_score"],
                "recent_interactions": [i for i in customer["interactions"][:3]],
                "objections": customer["objections"]
            }
            
            # Create prompt for LLM
            prompt = f"""
            As an AI sales assistant, analyze this customer data and provide 2-3 unique insights that could help a sales engineer:
            
            {json.dumps(customer_summary, indent=2)}
            
            Focus on non-obvious insights that could help close the deal. Provide each insight as a concise bullet point starting with a hyphen (-).
            """
            
            # Use cached model and tokenizer if available
            if not hasattr(self, '_llm_model') or not hasattr(self, '_llm_tokenizer'):
                import os
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                # Check for HF token
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    raise ValueError("Hugging Face token not found. Set HF_TOKEN environment variable.")
                
                # Get device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Initialize the model and tokenizer
                model_name = "google/gemma-3-1b-it"
                self._llm_tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                self._llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    token=hf_token,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                self._llm_device = device
                
            # Tokenize input with proper padding
            inputs = self._llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Move inputs to device
            if hasattr(self, '_llm_device') and self._llm_device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate response with proper parameters
            with torch.no_grad():
                outputs = self._llm_model.generate(
                    **inputs, 
                    max_new_tokens=800,  # Longer output to ensure we get full insights
                    num_return_sequences=1,
                    temperature=0.7,  # Add some creativity but not too random
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.1
                )
            
            # Decode the response
            response = self._llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Better parsing of insights using regex
            import re
            insights = []
            
            # Extract the generated portion (after the prompt)
            if prompt in response:
                response = response[response.find(prompt) + len(prompt):]
            
            # Find bullet points
            bullet_pattern = r'[-*•]\s*(.*?)(?=\n[-*•]|\n\n|$)'
            numbered_pattern = r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|$)'
            
            bullet_matches = re.findall(bullet_pattern, response, re.DOTALL)
            numbered_matches = re.findall(numbered_pattern, response, re.DOTALL)
            
            # Combine and clean insights
            all_matches = bullet_matches + numbered_matches
            for match in all_matches:
                insight = match.strip()
                if insight and len(insight) > 10:  # Reasonable length for an insight
                    insights.append(insight)
            
            # If no structured insights found, try to extract paragraphs
            if not insights:
                paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
                for p in paragraphs:
                    if len(p) > 20 and len(p) < 200:  # Reasonable insight length
                        insights.append(p)
            
            return insights[:3]  # Return at most 3 insights
            
        except Exception as e:
            import traceback
            print(f"Error generating LLM insights: {str(e)}")
            print(traceback.format_exc())
            return []

    def analyze_interaction(self, customer_id: str, interaction_text: str) -> Dict[str, Any]:
        """Analyze a customer interaction and provide insights"""
        if customer_id not in self.customers:
            return {"error": "Customer not found"}
            
        try:
            if not self._validate_customer(customer_id):
                self.logger.log_error("Customer Validation", f"Customer {customer_id} not found")
                return None
                
            self.logger.log_interaction(customer_id, "New Interaction", interaction_text)
            
            # Prepare interaction data
            customer = self.customers[customer_id]
            interaction_data = {
                "customer_details": customer,
                "interaction_text": interaction_text,
                "previous_sentiment": customer.get("sentiment", "neutral")
            }
            
            # Create prompt for analysis
            prompt = self._create_interaction_prompt(interaction_data)
            
            # Get LLM insights
            insights = self._get_llm_insights(prompt)
            self.logger.log_insight(customer_id, "Interaction Analysis", str(insights))
            
            # Update customer record
            self._update_customer_record(customer_id, interaction_text, insights)
            
            # Return structured response with expected keys
            return {
                "sentiment": self._analyze_sentiment(interaction_text),
                "topics": self._extract_topics(interaction_text),
                "recommended_actions": self._generate_recommended_actions(customer)
            }
            
        except Exception as e:
            self.logger.log_error("Interaction Analysis", str(e), customer_id)
            return None

    def _validate_customer(self, customer_id: str) -> bool:
        return customer_id in self.customers

    def _create_interaction_prompt(self, data: Dict[str, Any]) -> str:
        return f"""
        Analyze this customer interaction and provide insights:
        
        {json.dumps(data, indent=2)}
        
        Consider:
        1. Sentiment and tone
        2. Key topics discussed
        3. Potential objections or concerns
        4. Next best actions
        5. Any red flags or opportunities
        
        Provide a structured analysis with specific recommendations.
        """

    def _update_customer_record(self, customer_id: str, interaction_text: str, insights: List[str]):
        customer = self.customers[customer_id]
        new_interaction = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "type": "conversation",
            "sentiment": self._analyze_sentiment(interaction_text),
            "topics": self._extract_topics(interaction_text),
            "description": interaction_text
        }
        customer["interactions"].append(new_interaction)
        customer["sentiment_score"] = self._calculate_sentiment_score(customer["interactions"])
        self._update_customer_state(customer_id)

    def _calculate_sentiment_score(self, interactions: List[Dict[str, Any]]) -> float:
        sentiment_scores = [i.get("sentiment_score", 0) for i in interactions]
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    def _update_customer_state(self, customer_id: str):
        try:
            customer = self.customers[customer_id]
            interactions = customer.get("interactions", [])
            
            if not interactions:
                return
                
            # Calculate average sentiment
            sentiment_scores = [i.get("sentiment_score", 0) for i in interactions]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Update stage based on sentiment and interaction count
            new_stage = self._determine_customer_stage(len(interactions), avg_sentiment)
            
            if new_stage != customer.get("stage"):
                self.logger.log_insight(
                    customer_id,
                    "Stage Transition",
                    f"From {customer.get('stage')} to {new_stage}"
                )
                customer["stage"] = new_stage
                
        except Exception as e:
            self.logger.log_error("State Update", str(e), customer_id)

    def _determine_customer_stage(self, interaction_count: int, avg_sentiment: float) -> str:
        if interaction_count <= 1:
            return "prospecting"
        elif avg_sentiment < -0.2:
            return "at_risk"
        elif "pricing" in self.customers[customer_id]["topics"] or "price" in self.customers[customer_id]["topics"]:
            return "negotiation"
        else:
            return "evaluation"

    def handle_objections(self, customer_id: str, objection_text: str) -> Dict[str, Any]:
        """Identify and address customer objections"""
        if customer_id not in self.customers:
            return {"error": "Customer not found"}
            
        try:
            if not self._validate_customer(customer_id):
                self.logger.log_error("Customer Validation", f"Customer {customer_id} not found")
                return None
                
            self.logger.log_objection(customer_id, "New Objection", objection_text)
            
            # Prepare objection analysis data
            customer = self.customers[customer_id]
            objection_data = {
                "customer_details": customer,
                "objection_text": objection_text,
                "previous_objections": customer.get("objection_history", [])
            }
            
            # Classify objection
            objection_type = self._classify_objection(objection_text)
            self.logger.log_objection(customer_id, objection_type, objection_text)
            
            # Generate response strategy
            response = self._generate_response_strategy(objection_data)
            self.logger.log_insight(customer_id, "Objection Response", response)
            
            # Update customer record
            self._update_customer_objection_history(customer_id, objection_type, response)
            
            # Generate follow-up strategy
            follow_up = self._generate_follow_up_strategy(customer_id, objection_type)
            self.logger.log_insight(customer_id, "Follow-up Strategy", follow_up)
            
            # Return structured response with expected keys
            return {
                "objection_type": objection_type,
                "response": response,
                "follow_up": follow_up
            }
            
        except Exception as e:
            self.logger.log_error("Objection Handling", str(e), customer_id)
            return None

    def _classify_objection(self, text: str) -> Optional[str]:
        """Classify the type of objection"""
        try:
            prompt = f"""
            Classify this sales objection into one of these categories:
            - price
            - features
            - timing
            - competition
            - authority
            - other
            
            Objection: {text}
            
            Respond with only the category name.
            """
            
            model_name = "google/gemma-3-1b-it"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=20, num_return_sequences=1)
            objection_type = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            valid_types = ["price", "features", "timing", "competition", "authority", "other"]
            return objection_type if objection_type in valid_types else "other"
            
        except Exception as e:
            print(f"Error classifying objection: {e}")
            return "other"
            
    def _generate_follow_up_strategy(self, customer_id: str, objection_type: str) -> Dict[str, Any]:
        """Generate follow-up strategy based on objection type"""
        try:
            prompt = f"""
            Generate a follow-up strategy for this customer:
            
            Customer: {self.customers[customer_id]["name"]}
            Industry: {self.customers[customer_id]["industry"]}
            Size: {self.customers[customer_id]["size"]}
            Stage: {self.customers[customer_id]["stage"]}
            Objection Type: {objection_type}
            
            Consider:
            1. Appropriate timing for follow-up
            2. Best communication channel
            3. Key points to address
            4. Required materials or resources
            5. Success metrics
            
            Provide a structured follow-up plan.
            """
            
            model_name = "google/gemma-3-1b-it"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=300, num_return_sequences=1)
            strategy = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "strategy": strategy,
                "timing": self._determine_follow_up_timing(self.customers[customer_id]["stage"], objection_type),
                "channel": self._determine_follow_up_channel(self.customers[customer_id]["size"], objection_type)
            }
            
        except Exception as e:
            print(f"Error generating follow-up strategy: {e}")
            return {"error": "Failed to generate follow-up strategy"}
            
    def _determine_follow_up_timing(self, stage: str, objection_type: str) -> str:
        """Determine appropriate timing for follow-up"""
        timing_map = {
            "prospecting": {
                "price": "1-2 days",
                "features": "2-3 days",
                "timing": "1 week",
                "competition": "2-3 days",
                "authority": "3-4 days",
                "other": "2-3 days"
            },
            "evaluation": {
                "price": "2-3 days",
                "features": "1-2 days",
                "timing": "3-4 days",
                "competition": "1-2 days",
                "authority": "2-3 days",
                "other": "2-3 days"
            },
            "negotiation": {
                "price": "1-2 days",
                "features": "1 day",
                "timing": "2-3 days",
                "competition": "1 day",
                "authority": "1-2 days",
                "other": "1-2 days"
            },
            "at_risk": {
                "price": "1 day",
                "features": "1 day",
                "timing": "1-2 days",
                "competition": "1 day",
                "authority": "1 day",
                "other": "1 day"
            }
        }
        
        return timing_map.get(stage, {}).get(objection_type, "2-3 days")
        
    def _determine_follow_up_channel(self, size: str, objection_type: str) -> str:
        """Determine appropriate communication channel for follow-up"""
        channel_map = {
            "Enterprise": {
                "price": "video call",
                "features": "video call",
                "timing": "email",
                "competition": "video call",
                "authority": "video call",
                "other": "video call"
            },
            "Mid-Market": {
                "price": "video call",
                "features": "video call",
                "timing": "phone call",
                "competition": "video call",
                "authority": "video call",
                "other": "phone call"
            },
            "SMB": {
                "price": "phone call",
                "features": "video call",
                "timing": "email",
                "competition": "phone call",
                "authority": "phone call",
                "other": "email"
            }
        }
        
        return channel_map.get(size, {}).get(objection_type, "email")

    def generate_proposal(self, customer_id: str) -> Dict[str, Any]:
        """Generate a personalized proposal for a customer"""
        if customer_id not in self.customers:
            return {"error": "Customer not found"}
            
        try:
            customer = self.customers[customer_id]
            
            # Prepare proposal data
            proposal_data = {
                "customer_name": customer["name"],
                "industry": customer["industry"],
                "size": customer["size"],
                "stage": customer["stage"],
                "sentiment_score": customer["sentiment_score"],
                "objections": customer["objections"],
                "recent_interactions": customer["interactions"][-3:] if customer["interactions"] else [],
                "recommended_product": self._recommend_product(customer)
            }
            
            # Create prompt for proposal generation
            prompt = f"""
            Generate a personalized sales proposal for this customer:
            
            {json.dumps(proposal_data, indent=2)}
            
            Consider:
            1. Customer's industry-specific needs
            2. Company size and budget considerations
            3. Previous objections and concerns
            4. Current sales stage
            5. Product features and benefits
            6. Implementation timeline
            7. Pricing and payment terms
            
            Provide a comprehensive proposal that addresses the customer's specific needs and concerns.
            """
            
            # Get LLM-generated proposal
            model_name = "google/gemma-3-1b-it"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=1000, num_return_sequences=1)
            proposal_content = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Generate executive summary
            summary_prompt = f"""
            Generate an executive summary for this proposal:
            
            Customer: {customer["name"]}
            Industry: {customer["industry"]}
            Size: {customer["size"]}
            Key Benefits: {proposal_data["recommended_product"]["key_features"]}
            
            Focus on:
            1. Business value
            2. Key differentiators
            3. Implementation approach
            4. ROI potential
            
            Keep it concise and impactful.
            """
            
            inputs = tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=200, num_return_sequences=1)
            executive_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Generate pricing section
            pricing_prompt = f"""
            Generate a pricing section for this proposal:
            
            Customer: {customer["name"]}
            Size: {customer["size"]}
            Base Price: ${proposal_data["recommended_product"]["base_price"]}
            Implementation Time: {proposal_data["recommended_product"]["implementation_time"]}
            
            Consider:
            1. Customer's budget sensitivity
            2. Payment terms
            3. Optional add-ons
            4. Implementation costs
            5. Support and maintenance
            
            Provide clear pricing structure with options.
            """
            
            inputs = tokenizer(pricing_prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=300, num_return_sequences=1)
            pricing_section = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return structured response with expected keys
            return {
                "proposal_content": proposal_content,
                "executive_summary": executive_summary,
                "pricing_section": pricing_section
            }
            
        except Exception as e:
            print(f"Error generating proposal: {e}")
            return {"error": "Failed to generate proposal"}
            
    def _generate_proposal_next_steps(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """Generate next steps after proposal submission"""
        try:
            prompt = f"""
            Generate next steps for this proposal:
            
            Customer: {customer["name"]}
            Stage: {customer["stage"]}
            Sentiment: {customer["sentiment_score"]}
            
            Consider:
            1. Appropriate follow-up timing
            2. Key stakeholders to involve
            3. Required materials or demos
            4. Success metrics
            5. Potential objections to address
            
            Provide a structured action plan.
            """
            
            model_name = "google/gemma-3-1b-it"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=300, num_return_sequences=1)
            next_steps = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "action_plan": next_steps,
                "timing": self._determine_proposal_follow_up_timing(customer["stage"]),
                "stakeholders": self._identify_key_stakeholders(customer["size"])
            }
            
        except Exception as e:
            print(f"Error generating next steps: {e}")
            return {"error": "Failed to generate next steps"}
            
    def _determine_proposal_follow_up_timing(self, stage: str) -> str:
        """Determine appropriate timing for proposal follow-up"""
        timing_map = {
            "prospecting": "3-4 days",
            "evaluation": "2-3 days",
            "negotiation": "1-2 days",
            "at_risk": "1 day"
        }
        
        return timing_map.get(stage, "2-3 days")
        
    def _identify_key_stakeholders(self, size: str) -> List[str]:
        """Identify key stakeholders based on company size"""
        stakeholder_map = {
            "Enterprise": [
                "Executive Sponsor",
                "Technical Decision Maker",
                "Business Unit Head",
                "Procurement Manager",
                "End Users"
            ],
            "Mid-Market": [
                "Business Owner/CEO",
                "Technical Lead",
                "Department Manager",
                "End Users"
            ],
            "SMB": [
                "Business Owner",
                "Technical Contact",
                "Primary Users"
            ]
        }
        
        return stakeholder_map.get(size, ["Primary Contact"])

    def _initialize_gemma_model(self):
        """Initialize the Gemma model and tokenizer with proper authentication"""
        if not hasattr(self, '_llm_model') or not hasattr(self, '_llm_tokenizer'):
            try:            
                # Check for HF token
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    raise ValueError("Hugging Face token not found. Set HF_TOKEN environment variable.")
                
                # Get device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Initialize the model and tokenizer
                model_name = "google/gemma-3-1b-it"
                self._llm_tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                self._llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    token=hf_token,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                self._llm_device = device
                self.logger.log_info("Model Initialization", f"Gemma model initialized on {device}")
                return True
            except Exception as e:
                self.logger.log_error("Model Initialization", f"Failed to initialize Gemma model: {str(e)}")
                return False
        return True

    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of interaction text"""
        try:
            # Initialize model if not already done
            if not self._initialize_gemma_model():
                return "neutral"  # Fallback if initialization fails
                
            prompt = f"""
            Analyze the sentiment of this text and classify it as positive, negative, or neutral:
            
            {text}
            
            Respond with only one word: positive, negative, or neutral.
            """
            
            # Tokenize input with proper padding
            inputs = self._llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Move inputs to device
            if hasattr(self, '_llm_device') and self._llm_device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate response with proper parameters
            with torch.no_grad():
                outputs = self._llm_model.generate(
                    **inputs, 
                    max_new_tokens=20,  # Short output for sentiment
                    num_return_sequences=1,
                    temperature=0.1,  # Low temperature for deterministic output
                    do_sample=False   # We want deterministic output for sentiment
                )
            
            sentiment = self._llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            # Extract just the sentiment word (in case model outputs more text)
            for valid_sentiment in ["positive", "negative", "neutral"]:
                if valid_sentiment in sentiment:
                    return valid_sentiment
                    
            # Fallback to neutral if no valid sentiment found
            return "neutral"
            
        except Exception as e:
            self.logger.log_error("Sentiment Analysis", str(e))
            return "neutral"

    def _generate_response_strategy(self, objection_data: Dict[str, Any]) -> str:
        """Generate a response strategy for an objection"""
        try:
            # Initialize model if not already done
            if not self._initialize_gemma_model():
                return "I apologize, but I'm having trouble generating a response at the moment. Please try again later."
                
            prompt = f"""
            Generate a response strategy for this customer objection:
            
            Customer Details:
            {json.dumps(objection_data['customer_details'], indent=2)}
            
            Objection Text:
            {objection_data['objection_text']}
            
            Previous Objections:
            {json.dumps(objection_data['previous_objections'], indent=2)}
            
            Consider:
            1. Customer's industry and size
            2. Previous objections and concerns
            3. Current sales stage
            4. Available products and solutions
            
            Provide a natural, conversational response that addresses the objection while maintaining a professional tone.
            """
            
            # Tokenize input with proper padding
            inputs = self._llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Move inputs to device
            if hasattr(self, '_llm_device') and self._llm_device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate response with proper parameters
            with torch.no_grad():
                outputs = self._llm_model.generate(
                    **inputs,
                    max_new_token=300,  # Longer context for better response
                    num_return_sequences=1,
                    temperature=0.7,  # Add some creativity
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True
                )
            
            response = self._llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response by removing the prompt
            if prompt in response:
                response = response[response.find(prompt) + len(prompt):].strip()
            
            return response
            
        except Exception as e:
            import traceback
            self.logger.log_error("Response Strategy Generation", str(e))
            self.logger.log_error("Traceback", traceback.format_exc())
            return "I apologize, but I'm having trouble generating a response at the moment. Please try again later."

agent = SalesConversionAgent(crm_data_path="Raw_data\çrm_interactions.csv", use_llm=True)
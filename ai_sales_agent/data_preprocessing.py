import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_crm_data(filepath):
    """Preprocess the CRM interaction data"""
    # Load the data
    df = pd.read_csv(filepath)
    
    # Convert date and time columns to datetime
    df['Interaction_DateTime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time_of_Interaction'], 
        errors='coerce'
    )
    
    # Create a simplified customer ID using Document_No
    df['Customer_ID'] = df['Document_No'].astype(str)
    
    # Create a clean description by combining fields
    df['Clean_Description'] = df['Description'].fillna('')
    # Add work description if available
    mask = df['TRIM(WorkDescription)'].notna()
    df.loc[mask, 'Clean_Description'] += ': ' + df.loc[mask, 'TRIM(WorkDescription)']
    
    # Map evaluation to sentiment
    sentiment_map = {
        'Success': 'positive',
        'Successful': 'positive',
        'Good': 'positive',
        'Excellent': 'positive',
        'Average': 'neutral',
        'Neutral': 'neutral',
        'Fail': 'negative',
        'Poor': 'negative',
        'Rejected': 'negative'
    }
    df['Sentiment'] = df['Evaluation'].map(lambda x: sentiment_map.get(x, 'neutral'))
    
    # Create an aggregated view per customer
    customers = df.groupby('Customer_ID').agg(
        company_name=('Contact_Company_Name', lambda x: x.mode()[0] if not x.mode().empty else None),
        contact_name=('Contact_Name', lambda x: x.mode()[0] if not x.mode().empty else None),
        interaction_count=('Entry_No', 'count'),
        last_interaction=('Interaction_DateTime', 'max'),
        avg_duration=('Duration_Min', 'mean'),
        success_rate=('Evaluation', lambda x: (x == 'Success').mean()),
        sentiment_positive=('Sentiment', lambda x: (x == 'positive').mean()),
        sentiment_negative=('Sentiment', lambda x: (x == 'negative').mean()),
        total_cost=('Cost_LCY', 'sum'),
        has_campaign=('Campaign_Response', 'any')
    ).reset_index()
    
    # Extract customer interactions for detailed analysis
    customer_interactions = {}
    for customer_id in customers['Customer_ID'].unique():
        customer_data = df[df['Customer_ID'] == customer_id]
        interactions = []
        
        for _, row in customer_data.iterrows():
            interactions.append({
                'date': row['Interaction_DateTime'].strftime('%Y-%m-%d') if pd.notna(row['Interaction_DateTime']) else None,
                'type': row['Document_Type'],
                'sentiment': row['Sentiment'],
                'description': row['Clean_Description'],
                'duration': row['Duration_Min'],
                'agent': row['User_ID'],
                'has_attachment': row['Attachment']
            })
        
        customer_interactions[customer_id] = interactions
    
    return customers, customer_interactions
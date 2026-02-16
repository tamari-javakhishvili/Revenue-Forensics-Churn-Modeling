import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

def load_config():
    """
    Loads simulation parameters from a configuration file.

    Checks for the existence of config.json in the project root. If the file 
    is missing, it returns default values to ensure the simulation can run.

    Returns:
        dict: A dictionary containing 'n_customers' and 'collapse_factor'.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"n_customers": 5000, "collapse_factor": 0.2}

def generate_ecom_data():
    """
    Generates a synthetic e-commerce dataset with embedded market collapse logic.

    Simulation Details:
    - Customer Behavior: Purchases follow a Gamma distribution for realistic spending.
    - Market Anomaly (Germany): 
        1. Reduced transaction window (500 days vs 730) to simulate early churn.
        2. Structural AOV Failure: Transactions in 2025 are reduced by a collapse_factor.
    - Global Stability: Other markets follow a full 2-year cycle without collapse.

    Returns:
        pd.DataFrame: A dataset containing ['InvoiceNo', 'CustomerID', 'Country', 
                      'TotalAmount', 'InvoiceDate'].
    """
    config = load_config()
    np.random.seed(42)
    data = []
    start_date = datetime(2024, 1, 1)
    countries = ['United Kingdom', 'Germany', 'France', 'Spain']
    
    # Generate range of unique customer IDs based on config
    customer_ids = np.arange(10001, 10001 + config['n_customers'])
    
    for cid in customer_ids:
        country = np.random.choice(countries)
        
        # Simulate churn patterns for Germany (shorter observation window)
        max_days = 500 if country == 'Germany' else 730
        num_purchases = np.random.randint(5, 15)
        
        for _ in range(num_purchases):
            random_days = np.random.randint(0, max_days)
            invoice_date = start_date + timedelta(days=random_days)
            
            # Use Gamma distribution for realistic price skewing
            
            amount = np.random.gamma(shape=3, scale=100)
            
            # Apply simulated market collapse for Germany in 2025
            if country == 'Germany' and invoice_date.year == 2025:
                amount *= config.get('collapse_factor', 0.2)
                
            data.append({
                'InvoiceNo': np.random.randint(500000, 700000),
                'CustomerID': cid,
                'Country': country,
                'TotalAmount': round(amount, 2),
                'InvoiceDate': invoice_date
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    """
    Main execution entry point. 
    Generates the dataset and saves it as a CSV file in the data/ directory.
    """
    df = generate_ecom_data()
    

    #Calculate total revenue
    global_total = df['TotalAmount'].sum()
    
    # Calculate revenue from Germany
    germany_total = df[df['Country'] == 'Germany']['TotalAmount'].sum()
    
    print(f"✅ Simulation complete. Total rows: {len(df)}")
    print(f"🌍 Global Revenue (All countries): €{global_total:,.2f}")
    print(f"🇩🇪 Germany Revenue Only: €{germany_total:,.2f}")
    
    # Create data directory if it doesn't exist  
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    
    
    # Save data
    save_path = os.path.join(data_dir, 'ecommerce_data.csv')
    df.to_csv(save_path, index=False)
    print(f"✅ Simulation complete. Data generated: {len(df)} rows.")

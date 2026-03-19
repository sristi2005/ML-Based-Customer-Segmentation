import pandas as pd
import numpy as np

def generate_data(num_samples=200):
    np.random.seed(42)
    
    # Generate random data
    customer_ids = range(1, num_samples + 1)
    ages = np.random.randint(18, 70, size=num_samples)
    annual_incomes = np.random.randint(15, 140, size=num_samples)  # in thousands
    spending_scores = np.random.randint(1, 100, size=num_samples)
    genders = np.random.choice(['Male', 'Female'], size=num_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': genders,
        'Age': ages,
        'Annual Income (k$)': annual_incomes,
        'Spending Score (1-100)': spending_scores
    })
    
    # Save to CSV
    df.to_csv('customers.csv', index=False)
    print("Synthetic dataset generated and saved to 'customers.csv'.")

if __name__ == "__main__":
    generate_data()

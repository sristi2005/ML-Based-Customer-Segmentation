import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_save_model():
    try:
        df = pd.read_csv('customers.csv')
    except FileNotFoundError:
        print("Dataset not found. Please run 'data_generator.py' first.")
        return

    # Select features for clustering: Annual Income and Spending Score
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # We will use 5 clusters (a common optimal number for this type of data)
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)

    # Save the model and scaler
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Model and scaler have been trained and saved successfully.")

if __name__ == "__main__":
    train_and_save_model()

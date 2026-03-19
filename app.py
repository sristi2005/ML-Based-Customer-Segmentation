import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation App", page_icon="🛍️", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛍️ ML-Based Customer Segmentation</div>', unsafe_allow_html=True)
st.write("Welcome to the Customer Segmentation predictor! Enter the customer's Annual Income and Spending Score below to see which segment they belong to based on our K-Means clustering model.")

# Load model, scaler and data
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        data = pd.read_csv('customers.csv')
        return model, scaler, data
    except Exception as e:
        return None, None, None

model, scaler, df = load_assets()

if model is None:
    st.error("Model, scaler, or dataset not found. Please ensure you have run the data generator and training scripts first.")
    st.stop()

# Segment naming logic (approximate for typical 5-cluster KMeans on this data)
def get_segment_name(cluster):
    names = {
        0: "Standard (Average Income, Average Spend)",
        1: "Target (High Income, High Spend)",
        2: "Careful (High Income, Low Spend)",
        3: "Careless (Low Income, High Spend)",
        4: "Sensible (Low Income, Low Spend)"
    }
    return names.get(cluster, f"Cluster {cluster}")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Input Customer Data")
    
    with st.form("prediction_form"):
        annual_income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, help="Customer's annual income in thousands of dollars.")
        spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, help="Score assigned by the mall based on customer behavior and spending nature.")
        
        submitted = st.form_submit_button("Predict Segment")

    if submitted:
        # Predict
        input_data = pd.DataFrame([[annual_income, spending_score]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
        input_scaled = scaler.transform(input_data)
        cluster = model.predict(input_scaled)[0]
        
        st.success(f"### Predicted Segment:\n**{get_segment_name(cluster)}**")
        st.session_state['prediction'] = (annual_income, spending_score, cluster)

with col2:
    st.subheader("📊 Customer Segments Visualization")
    
    # Predict clusters for the entire dataset to visualize consistently
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    df['Cluster'] = model.predict(scaler.transform(X))
    
    # Visualization using Seaborn and Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define a custom palette
    palette = sns.color_palette("Set2", n_colors=len(df['Cluster'].unique()))
    
    sns.scatterplot(
        data=df, 
        x='Annual Income (k$)', 
        y='Spending Score (1-100)', 
        hue='Cluster', 
        palette=palette, 
        s=100, 
        alpha=0.7, 
        edgecolor='w',
        ax=ax
    )
    
    # Plot the predicted point if it exists
    if 'prediction' in st.session_state:
        inc, spnd, clst = st.session_state['prediction']
        ax.scatter(inc, spnd, color='red', s=400, marker='*', edgecolor='black', linewidths=1.5, label='Predicted Customer', zorder=5)
        
    ax.set_title('Customer Segments clusters', fontsize=14, weight='bold')
    ax.set_xlabel('Annual Income (k$)', fontsize=12)
    ax.set_ylabel('Spending Score (1-100)', fontsize=12)
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    # Replace cluster numbers with names in legend
    new_labels = []
    for label in labels:
        if label.isdigit():
            new_labels.append(get_segment_name(int(label)))
        else:
            new_labels.append(label)
    ax.legend(handles, new_labels, title="Segments", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    st.pyplot(fig)

with st.expander("View Raw Dataset"):
    st.dataframe(df.drop(columns=['Cluster'], errors='ignore'), use_container_width=True)
    st.caption(f"Dataset contains {len(df)} records.")

# ML-Based Customer Segmentation

This project implements a Machine Learning Customer Segmentation application using K-Means clustering. Customers are segmented based on their Annual Income and Spending Score. The project uses Python, scikit-learn for modeling, and Streamlit for a beautiful, responsive web-based user interface.

## Quick Start

1. **Install Dependencies**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data**
   Generate the synthetic dataset (`customers.csv`):
   ```bash
   python data_generator.py
   ```

3. **Train Model**
   Train the K-Means clustering model and save the model artifacts (`kmeans_model.pkl` and `scaler.pkl`):
   ```bash
   python train_model.py
   ```

4. **Run the Application**
   Launch the Streamlit web UI using the module command (recommended for Windows):
   ```bash
   python -m streamlit run app.py
   ```
   *(Alternatively, if your PATH is configured, you can just run `streamlit run app.py`)*
   This will open the application in your default web browser where you can interact with the segmentation model and visualize the customer clusters.

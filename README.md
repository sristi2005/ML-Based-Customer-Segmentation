# ML-Based Customer Segmentation

An interactive Machine Learning project that segments customers into distinct groups using K-Means clustering. The application allows users to input customer details and instantly predict their segment through a user-friendly Streamlit web interface.

---

## 🚀 Features

- Customer segmentation using K-Means clustering
- Synthetic dataset generation for training
- Data preprocessing with StandardScaler
- Interactive Streamlit web application
- Real-time predictions based on user input
- Data visualization using Seaborn & Matplotlib
- Model and scaler persistence using Joblib

---

## 🧠 Tech Stack

- Python
- scikit-learn
- pandas, numpy
- Streamlit
- matplotlib, seaborn
- joblib

---

## 📂 Project Structure

```

ML-Customer-Segmentation/
│── data_generator.py      # Generates synthetic customer dataset
│── train_model.py         # Trains K-Means model and saves artifacts
│── app.py                 # Streamlit web application
│── customers.csv          # Generated dataset
│── kmeans_model.pkl       # Saved trained model
│── scaler.pkl             # Saved scaler
│── requirements.txt       # Project dependencies
│── README.md              # Documentation

````

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ml-customer-segmentation.git
cd ml-customer-segmentation
````

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 How to Run

### Step 1: Generate Dataset

```bash
python data_generator.py
```

### Step 2: Train Model

```bash
python train_model.py
```

### Step 3: Run Web App

```bash
streamlit run app.py
```

---

## 📈 How It Works

1. **Data Generation**

   * Creates synthetic customer data with:

     * Age
     * Annual Income
     * Spending Score

2. **Model Training**

   * Data is scaled using StandardScaler
   * K-Means clustering is applied (k=5)
   * Model is saved using Joblib

3. **Prediction**

   * User inputs are scaled
   * Model predicts customer segment in real-time

---

## 📊 Sample Use Case

Businesses can use this model to:

* Identify high-value customers
* Target marketing campaigns
* Improve customer retention strategies

---

## ✅ Verification

* Dataset successfully generated (`customers.csv`)
* Model trained and saved (`kmeans_model.pkl`, `scaler.pkl`)
* Streamlit app runs without errors and produces correct predictions

---

## 🔮 Future Improvements

* Use real-world datasets (e.g., Mall Customers Dataset)
* Add cluster labeling (e.g., Premium, Budget, Regular)
* Deploy on cloud (Streamlit Cloud / AWS)
* Add more features (Gender, Location, Purchase History)

## ⭐ If you like this project

Give it a star ⭐ and feel free to contribute!



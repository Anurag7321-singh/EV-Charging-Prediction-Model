## 📌 EV-Charging-Prediction-Model

A machine learning and forecasting project to predict Electric Vehicle (EV) adoption trends using real-world data and build an interactive dashboard.

---

### 🚀 **Project Overview**

With the rising number of EVs, power grids and city planners require smarter infrastructure decisions.
This project forecasts EV adoption and visualizes insights using **data preprocessing, machine learning, and a Streamlit dashboard**.

---

### 🛠 **Technologies Used**

* **Python**
* **Jupyter Notebook (`Complete.ipynb`)** for model building & experimentation
* **Pandas, NumPy** for data processing
* **Scikit-learn** for machine learning
* **Matplotlib, Seaborn** for visualization
* **Streamlit (`app.py`)** for interactive dashboard
* **Joblib** for model saving/loading
* **Git & GitHub** for version control

---

### 📊 **Key Features**

* ✅ Data Preprocessing & Feature Engineering
* ✅ EV Adoption Forecast using Random Forest Regression
* ✅ 3-Year Forecast Visualization for each county
* ✅ Compare EV growth for multiple counties
* ✅ Interactive Streamlit Dashboard
* ✅ Forecast Growth Summary & Insights

---

### 📂 **Project Structure**

```
EV-Charging-Prediction-Model/
│
├── Complete.ipynb                 # Jupyter Notebook (model training & EDA)
├── app.py                         # Streamlit dashboard
├── forecasting_ev_model.pkl       # Trained machine learning model
├── preprocessed_ev_data.csv       # Cleaned dataset
├── ev_car_factory.jpg             # Dashboard background image
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

### 📈 **Model Performance**

* **MAE:** \~0.01
* **RMSE:** \~8.06
* **R² Score:** 1.00
* 3-Year EV growth insights for top counties

---

### 🎨 **Streamlit Dashboard**

The dashboard includes:

* Cumulative EV adoption graph (historical + 3-year forecast)
* Comparison of EV adoption growth for up to 3 counties
* Forecast insights with growth percentage
* Attractive, modern UI with animated effects

---

### 🔧 **How to Run**

```bash
# Clone repository
git clone https://github.com/Anurag7321-singh/EV-Charging-Prediction-Model.git

# Navigate into project
cd EV-Charging-Prediction-Model

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

### 📌 **Dataset**

Data sourced from:

* Washington State Government EV datasets
* Preprocessed using Python and Pandas

---

### 👨‍💻 **Developer**

**Anurag Pratap Singh**
🌐 [GitHub Repository](https://github.com/Anurag7321-singh/EV-Charging-Prediction-Model)


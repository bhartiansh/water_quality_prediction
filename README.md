# Water Quality Analysis and Prediction

This project aims to analyze and predict the quality of water using real-world datasets with key physicochemical and biological indicators. It combines data preprocessing, threshold-based classification, and machine learning modeling for intelligent, proactive water quality management.

---

## 📊 Dataset
The dataset includes parameters such as:

- Ammonia Nitrogen (mg/L)
- Dissolved Oxygen (% saturation & mg/L)
- Faecal Coliforms (cfu/100mL)
- Nitrate, Nitrite, Total Nitrogen
- Orthophosphate Phosphorus, Total Phosphorus
- Temperature, pH, Suspended Solids
- Chlorophyll-a, Total Kjeldahl Nitrogen

With thresholds applied to assess water quality.

---

## 🔍 Project Workflow

1. **Data Preprocessing**
   - Handle missing values
   - Normalize numerical features
   - Encode categorical variables

2. **Exploratory Data Analysis (EDA)**
   - Visualize distributions and correlations
   - Identify potential outliers and anomalies

3. **Threshold Classification**
   - Label samples as 'Good' or 'Poor' based on provided water quality thresholds

4. **Model Building**
   - Train ML models (e.g., Random Forest, SVM, XGBoost)
   - Evaluate using Accuracy, Precision, Recall, F1 Score

5. **Deployment Options**
   - Run locally using Python
   - Execute in Google Colab (if GPU/remote access required)

---

## Folder Structure

```bash
water_quality_prediction/
├── data/                 # Raw and processed datasets
├── notebooks/            # Jupyter notebooks (EDA, preprocessing, training)
├── src/                  # Source Python scripts (preprocessing, modeling)
├── results/              # Outputs, visualizations, model metrics
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── run.py                # Entry point to execute pipeline
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/water_quality_prediction.git
cd water_quality_prediction
```

### 2. Set Up Environment
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Run
```bash
python run.py
```

Or run notebooks from the `notebooks/` folder.

---

## References
- WHO Guidelines for Drinking-water Quality
- US EPA Water Quality Standards
- Hong Kong Environmental Protection Department (HK EPD)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib & Seaborn](https://seaborn.pydata.org/)

---

## Author
Ansh (B.Tech AI & ML)  
Feel free to raise issues or contribute to improvements!
# Water Quality Analysis and Prediction

This project aims to analyze and predict the quality of water using real-world datasets with key physicochemical and biological indicators. It combines data preprocessing, threshold-based classification, and machine learning modeling for intelligent, proactive water quality management.

---

## 📊 Dataset

Dataset is taken from : https://cd.epic.epd.gov.hk/EPICRIVER/marine/?lang=en

The dataset includes parameters such as:

Water quality parameters

-------------------------------------
5-day Biochemical Oxygen Demand (mg/L)
Silica (mg/L)
Chlorophyll-a μg/L
pH
Temperature (C)
E. coli (cfu/100mL)
Faecal Coliforms (cfu/100mL)
Total Phosphorus (mg/L)
Orthophosphate Phosphorus (mg/L)
Volatile Suspended Solids (mg/L)
Phaeo-pigments (μg/L)
Suspended Solids (mg/L)
Dissolved Oxygen Saturation (%)
Dissolved Oxygen (mg/L)
Salinity (psu)
Secchi Disc Depth (M)
Nitrite Nitrogen (mg/L)
Nitrate Nitrogen (mg/L)
Ammonia Nitrogen (mg/L)
Total Nitrogen (mg/L)
Total Kjeldahl Nitrogen (mg/L)
Total Inorganic Nitrogen (mg/L)
Turbidity (NTU)
Unionized Ammonia (mg/L)

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
   - Train ML models (Random Forest, SVM, XGBoost, Logistic regression, Stocastic Gradient Decient)
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

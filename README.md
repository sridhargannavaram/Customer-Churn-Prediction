# Customer Churn Prediction using Machine Learning  

This project applies data analysis, visualization, and machine learning to predict whether a customer will leave a telecom service.  

## Overview  

Customer churn is a major issue for businesses, especially in the telecom industry. This project aims to:  
- Analyze customer data to understand churn behavior  
- Use Machine Learning models to predict churn  
- Provide business insights to reduce churn rates  

## Dataset Details  

- Dataset Name: Telco Customer Churn  
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- Rows: 7,043  
- Columns: 21  

### Key Features in the Dataset:
- `tenure` – How long the customer has been with the company  
- `MonthlyCharges` – Amount customer pays per month  
- `Contract` – Type of contract (Month-to-month, One year, Two years)  
- `InternetService` – Type of internet service  
- `PaymentMethod` – How the customer pays  
- `Churn` – (Target Variable: 1 = Churn, 0 = No Churn)  

## Tools & Technologies  

- Programming Language: Python  
- Libraries Used:
  - `pandas`, `numpy` – Data processing  
  - `matplotlib`, `seaborn` – Data visualization  
  - `scikit-learn` – Machine learning models  

## Exploratory Data Analysis (EDA)  

- Churn Rate Analysis: Understand how many customers leave  
- Feature Correlation: Find patterns between customer behavior and churn  
- Visualization Techniques:  
  - Boxplots – Compare `MonthlyCharges` across churned and non-churned customers  
  - Heatmaps – Show relationships between numerical features  

## Model Performance  

We trained two machine learning models to predict churn:  
1. Logistic Regression 
2. Random Forest Classifier

| Model                | Accuracy  | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression | 82%       | 0.86      | 0.90   | 0.88     |
| Random Forest       | 80%       | 0.68      | 0.58   | 0.62     |

Best Model: Logistic Regression (82% Accuracy)  

## Key Findings  

- Customers with high `MonthlyCharges` are more likely to churn  
- Short-term contract customers have a higher churn rate  
- Electronic check payment method has a higher churn probability  
- Longer tenure customers tend to stay with the company  

---

## How to Run the Project  

### 1. Clone the Repository
```
git clone https://github.com/yourusername/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install Dependencies
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Jupyter Notebook
Open `Customer_Churn_Prediction.ipynb` and run all cells.

## Future Improvements  
To improve accuracy, we can:  
- Use Deep Learning (Neural Networks) for better predictions  
- Implement SMOTE to balance the dataset  
- Deploy the model using Flask / Streamlit  





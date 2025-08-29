### Health Insurance Charges Prediction

#### Overview

This project focuses on predicting health insurance charges for individuals based on their demographic and lifestyle information. By analyzing a dataset containing features like age, BMI, number of children, and smoking status, we build a machine learning model to estimate medical costs.

---

#### Project Goals

* Perform **Exploratory Data Analysis (EDA)** to understand the distribution of key features and their relationships with insurance charges.
* Clean and preprocess the data, handling duplicates and encoding categorical variables.
* Train and evaluate multiple regression models, including Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, and SVR.
* Select the best-performing model based on evaluation metrics like R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).
* Create a predictive function to estimate insurance charges for new data points.

---

#### Key Features

The dataset includes the following features:

* **age:** The age of the primary beneficiary.
* **sex:** The gender of the beneficiary (male or female).
* **bmi:** Body Mass Index, an indicator of body fatness.
* **children:** The number of children covered by the insurance.
* **smoker:** Whether the beneficiary is a smoker (yes/no).
* **region:** The beneficiary's residential area in the US (e.g., northeast, southeast).
* **charges:** The individual medical costs billed by health insurance.

---

#### Models Evaluated

The following models were trained and compared:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree Regressor
* **Random Forest Regressor** (The best-performing model)
* **SVR** (Support Vector Regressor)
* XGBoost Regressor

---

#### Model Performance Metrics

The table below shows the performance of the models on three key metrics: **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R-squared ($R^2$)**.

| Model | MSE | MAE | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | 3.85E+07 | 4334.04 | 0.7853 |
| **Ridge Regression** | 3.86E+07 | 4346.20 | 0.7846 |
| **Lasso Regression** | 3.85E+07 | 4334.09 | 0.7853 |
| **Decision Tree** | 4.09E+07 | 2901.15 | 0.7718 |
| **Random Forest** | **2.28E+07** | **2606.45** | **0.8729** |
| **XGBoost** | 2.25E+07 | 2675.45 | 0.8745 |
| **SVR** | 1.96E+08 | 8402.81 | -0.0942 |

**Analysis of Results:**

* **Random Forest** and **XGBoost** are the top-performing models with the lowest MSE and MAE and the highest $R^2$ scores.
* The **SVR** model performed poorly, indicated by a negative $R^2$ score, which suggests its predictions are worse than simply using the mean of the data.
* Linear models showed similar performance to each other but were outperformed by the tree-based models, which better captured the data's complexity.

---

#### Predicted vs. Actual Plots

The following plots visualize the predicted charges against the actual charges for each model. The goal is for the predicted values to align closely with the red dashed line, which represents the ideal scenario where `y_predicted` equals `y_actual`. A tight cluster of points around this line indicates a high-performing model.

**Random Forest Regressor: Predicted vs Actual**



This plot for the **Random Forest Regressor** shows that the data points are tightly clustered around the ideal line, especially for lower charge values. This visualization further supports its high $R^2$ score and confirms its strong predictive performance.

---

#### Code Structure

* `health_insurance_predection.py`: The main Python script containing all the data processing, model training, and evaluation code.
* `medical_insurance.csv`: The dataset used for the project.

---

#### How to Run the Code

1.  Ensure you have all the required libraries installed: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, and `xgboost`.
2.  Download the `medical_insurance.csv` file and place it in the same directory as the script.
3.  Run the `health_insurance_predection.py` file.
check the project :- https://colab.research.google.com/drive/12e2xHhMI_6Q7HNWvZ8VychcWUUEWbhtZ#scrollTo=mnJo8SV6K3g5

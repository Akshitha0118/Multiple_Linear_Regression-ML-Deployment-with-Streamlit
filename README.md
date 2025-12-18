# Multiple_Linear_Regression-ML-Deployment-with-Streamlit


# streamlit live app link :https://multiplelinearregression-ml-deployment-with-app-a8vyz5zuerwhmd.streamlit.app/


# 📈 Investment Profit Prediction using Multiple Linear Regression

This project demonstrates how Multiple Linear Regression can be applied to an investment dataset to analyze the impact of different investment attributes on profit and to identify statistically significant features using OLS (Ordinary Least Squares) regression.

# 🚀 Project Overview

The objective of this project is to:

* Build a Linear Regression model to predict profit based on multiple investment factors

* Encode categorical variables for model compatibility

* Perform train–test splitting

* Evaluate model performance using bias and variance

* Apply backward elimination using p-values from OLS to select the most significant features

# 🛠️ Technologies & Libraries Used

* Python

* Pandas – data handling

* NumPy – numerical operations

* Matplotlib – visualization support

* Scikit-learn – machine learning model

* Statsmodels – statistical analysis (OLS)

# 📂 Dataset Description

The dataset (Investment.csv) contains multiple investment-related attributes such as:

* Different investment channels

* Promotional and research spending

* State information (categorical)

* Profit (target variable)

# ⚙️ Workflow & Methodology

* Data Loading & Preprocessing

* Loaded dataset using Pandas

* Separated independent variables (X) and dependent variable (y)

* Converted categorical data using LabelEncoder

* Model Training

* Split data into training and testing sets (75% / 25%)

* Trained a Multiple Linear Regression model

* Extracted slope (coefficients) and intercept

* Statistical Analysis

* Applied OLS regression

* Added constant term manually

* Performed backward elimination by removing features with p-value > 0.05

* Model Evaluation

* Bias Score: Training accuracy

* Variance Score: Testing accuracy

# 📊 Key Insights

* OLS regression helps identify statistically significant investment factors

* Features with high p-values contribute less to profit prediction

* The refined model improves interpretability and reliability

* Useful for data-driven investment decision-making

# 📌 Results

* Coefficients and intercept obtained from Linear Regression

* Final model retains only significant predictors

* Clear distinction between training and testing performance

# 🔮 Future Enhancements

* Add data visualization for feature impact

* Deploy the model using Streamlit or Flask

* Automate feature selection

# 📎 How to Run
pip install pandas numpy matplotlib scikit-learn statsmodels
python app.py

# 🙌 Acknowledgements

This project was developed as part of hands-on learning in Machine Learning and Statistical Modeling, focusing on practical implementation and interpretability.

# 🚀 Data-backed recommendation: 
 Based on the regression model predictions, higher investment in the Digital Marketing attribute shows a strong positive impact on future profit, making it the most effective investment driver in the dataset.

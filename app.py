import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import os

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="MLR Dashboard", layout="wide")

# ===============================
# Load CSS
# ===============================
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===============================
# Header
# ===============================
st.markdown('<div class="title">📊 Multiple Linear Regression</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Glass UI • Sklearn • Statsmodels</div>', unsafe_allow_html=True)

# ===============================
# Load Dataset (Safe Path)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(BASE_DIR, "Investment.csv"))

# ===============================
# Sidebar Navigation
# ===============================
section = st.sidebar.radio(
    "Navigate",
    ["Dataset", "Train Model", "Predictions", "OLS Summary"]
)

# ===============================
# Data Preprocessing
# ===============================
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

labelencoder = LabelEncoder()
X[:, -1] = labelencoder.fit_transform(X[:, -1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

bias = regressor.score(X_train, y_train)
variance = regressor.score(X_test, y_test)

# ===============================
# Dataset Section
# ===============================
if section == "Dataset":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📄 Dataset Preview")
    st.dataframe(dataset.head())
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Train Model Section
# ===============================
elif section == "Train Model":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("⚙️ Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(
        f"<div class='metric-box'><div class='metric-value'>{bias:.3f}</div>"
        "<div class='metric-label'>Bias (Train R²)</div></div>",
        unsafe_allow_html=True
    )

    col2.markdown(
        f"<div class='metric-box'><div class='metric-value'>{variance:.3f}</div>"
        "<div class='metric-label'>Variance (Test R²)</div></div>",
        unsafe_allow_html=True
    )

    col3.markdown(
        f"<div class='metric-box'><div class='metric-value'>{regressor.intercept_:.2f}</div>"
        "<div class='metric-label'>Intercept</div></div>",
        unsafe_allow_html=True
    )

    col4.markdown(
        f"<div class='metric-box'><div class='metric-value'>{len(regressor.coef_)}</div>"
        "<div class='metric-label'>Features</div></div>",
        unsafe_allow_html=True
    )

    st.markdown("### 📐 Coefficients")
    st.write(regressor.coef_)

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Predictions Section
# ===============================
elif section == "Predictions":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🔮 Actual vs Predicted")

    pred_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    st.dataframe(pred_df.head(15))
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# OLS Summary Section
# ===============================
elif section == "OLS Summary":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📑 OLS Regression Output")

    X_ols = sm.add_constant(pd.DataFrame(X).astype(float))
    y_ols = pd.Series(y).astype(float)

    ols_model = sm.OLS(y_ols, X_ols).fit()
    st.text(ols_model.summary())

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Footer
# ===============================
st.markdown('<div class="footer">Built for ML Portfolio • Streamlit UI</div>', unsafe_allow_html=True)

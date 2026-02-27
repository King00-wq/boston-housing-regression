import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Boston Housing Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# =========================================================
# Hero Section
# =========================================================
st.title("🏡 Boston Housing Price Prediction Dashboard")
st.subheader("Regression-based property valuation using ML")
st.divider()
st.caption(
    "This dashboard presents a regression-based valuation model built on the "
    "Boston Housing dataset. It enables structured feature input and analytical "
    "interpretation of predicted property values."
)

# =========================================================
# Load Data
# =========================================================
df = pd.read_csv("boston.csv")
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# =========================================================
# Train Models (Base Reference Structure Preserved)
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linear_model = LinearRegression()
linear_model.fit(X_scaled, y)

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X, y)

# =========================================================
# Tabs Layout
# =========================================================
tab1, tab2 = st.tabs(["Prediction Dashboard", "Model Insights"])

# =========================================================
# TAB 1 – Prediction Dashboard
# =========================================================
with tab1:

    left_col, right_col = st.columns([1, 1])

    # -----------------------------------------------------
    # Input Section
    # -----------------------------------------------------
    with left_col:
        st.header("Input Property Features")

        st.subheader("Primary Drivers")

        rm = st.slider(
            "RM – Average Number of Rooms",
            min_value=3.0,
            max_value=9.0,
            value=6.0,
            help="Average number of rooms per dwelling."
        )

        lstat = st.slider(
            "LSTAT – % Lower Status Population",
            min_value=1.0,
            max_value=40.0,
            value=12.0,
            help="Percentage of lower socio-economic population."
        )

        ptratio = st.slider(
            "PTRATIO – Pupil-Teacher Ratio",
            min_value=12.0,
            max_value=25.0,
            value=18.0,
            help="Pupil-teacher ratio by town."
        )

        crim = st.slider(
            "CRIM – Crime Rate",
            min_value=0.0,
            max_value=90.0,
            value=3.0,
            help="Per capita crime rate by town."
        )

        st.subheader("Secondary Variables")

        tax = st.number_input(
            "TAX – Property Tax Rate",
            min_value=100,
            max_value=800,
            value=300,
            help="Full-value property tax rate per $10,000."
        )

        nox = st.number_input(
            "NOX – Nitric Oxide Concentration",
            min_value=0.3,
            max_value=1.0,
            value=0.5,
            help="Air pollution concentration level."
        )

        indus = st.number_input(
            "INDUS – Non-Retail Business Acres",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            help="Proportion of non-retail business acres."
        )

        with st.expander("All Dataset Features"):
            zn = st.number_input("ZN – Residential Land Zoned (%)", 0.0, 100.0, 0.0)
            chas = st.number_input("CHAS – Near Charles River (0/1)", 0, 1, 0)
            age = st.number_input("AGE – Older Housing Units (%)", 0.0, 100.0, 60.0)
            dis = st.number_input("DIS – Distance to Employment Centers", 1.0, 12.0, 4.0)
            rad = st.number_input("RAD – Highway Accessibility Index", 1, 24, 5)
            b = st.number_input("B – Demographic Feature", 0.0, 400.0, 350.0)

        predict_button = st.button("Predict House Price")

    # -----------------------------------------------------
    # Prediction Logic
    # -----------------------------------------------------
    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    if predict_button:

        with st.spinner("Running regression model..."):
            progress = st.progress(0)
            for i in range(0, 101, 25):
                time.sleep(0.1)
                progress.progress(i)

            input_array = np.array([[
                crim, zn, indus, chas, nox, rm, age,
                dis, rad, tax, ptratio, b, lstat
            ]])

            linear_pred = linear_model.predict(
                scaler.transform(input_array)
            )[0]

            st.session_state.prediction = linear_pred
            st.balloons()

    # -----------------------------------------------------
    # Output Section
    # -----------------------------------------------------
    with right_col:
        st.header("Prediction Output")

        if st.session_state.prediction is not None:

            st.success("Prediction Complete")

            predicted_value = st.session_state.prediction
            predicted_usd = predicted_value * 1000
            market_average = 22500
            adjusted_impact = predicted_usd * 1.05

            m1, m2, m3 = st.columns(3)

            m1.metric("Predicted Price", f"${predicted_usd:,.2f}")

            market_label = (
                "Above Market"
                if predicted_usd > market_average
                else "Below Market"
            )

            m2.metric("Market Comparison", market_label)

            investment_score = min(max(int((predicted_value / 50) * 100), 0), 100)
            m3.metric("Investment Score", f"{investment_score}/100")

            st.subheader("Price Comparison Overview")

            chart_df = pd.DataFrame({
                "Category": ["Predicted", "Market Average", "Adjusted Impact"],
                "Value": [predicted_usd, market_average, adjusted_impact]
            }).set_index("Category")

            st.bar_chart(chart_df)

            if predicted_value < 20:
                st.success("Affordable Segment")
            elif 20 <= predicted_value <= 35:
                st.warning("Mid-Range Market")
            else:
                st.error("Luxury Market")

# =========================================================
# TAB 2 – Model Insights
# =========================================================
with tab2:

    st.header("Model Insights & Academic Notes")

    st.info(
        "Regression is a supervised learning method that models relationships "
        "between independent variables (housing attributes) and a continuous "
        "dependent variable (property price)."
    )

    st.write(
        "Linear Regression is appropriate for structured tabular datasets "
        "such as Boston Housing because it captures linear associations between "
        "features like number of rooms and socio-economic indicators."
    )

    st.write(
        "Key drivers influencing property valuation typically include:"
    )

    feature_summary = pd.DataFrame({
        "Feature": ["RM (Rooms)", "LSTAT (%)", "CRIM", "PTRATIO"],
        "Influence on Price": [
            "Strong Positive",
            "Strong Negative",
            "Negative",
            "Moderate Negative"
        ]
    })

    st.table(feature_summary)

    st.write(
        "Higher room counts generally increase valuation, while higher crime "
        "rates and socio-economic disadvantage reduce property price. "
        "These relationships align with urban economic principles."
    )
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Performance Predictor", page_icon="💼", layout="centered")

@st.cache_data
def train_manual_model():
    with open('dataconfig.txt', 'r') as f:
        data_path = f.read().strip()

    df = pd.read_csv(data_path)
    numerical_df = df.select_dtypes(include=[np.number])

    raw_X = numerical_df.iloc[:-1, :-1].values
    y = numerical_df.iloc[:-1, -1].values

    N = len(df)

    X = np.hstack((np.ones((N - 1, 1)), raw_X.reshape(N - 1, -1)))

    X_transpose = X.T
    beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

    return beta

beta = train_manual_model()

st.title("🎓 Student Performance Predictor")
st.write("This app uses a custom-built Statistical Decision Theory regression model to predict a student's Performance Index.")
st.markdown("---")

st.subheader("Enter Student Details")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

papers_practiced = st.number_input(
    "Practice Papers Completed",
    min_value=0,
    max_value=20,
    value=5,
    step=1
)

previous_marks = st.number_input(
    "Previous Marks (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

sleeping_hours = st.number_input(
    "Sleeping Hours per Day",
    min_value=0.0,
    max_value=12.0,
    value=7.0,
    step=0.5
)

if st.button("Predict Performance", type="primary"):

    predicted_performance = (
        beta[0]
        + beta[1] * study_hours
        + beta[2] * papers_practiced
        + beta[3] * previous_marks
        + beta[4] * sleeping_hours
    )

    st.markdown("### 📊 Prediction Result")

    st.success(
        f"Predicted Performance Index: **{predicted_performance:.2f}**"
    )

    with st.expander("See the underlying math (Decision Rule)"):
        st.latex(
            r"f(x)=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_3+\beta_4x_4"
        )

        st.write(f"Study Hours = {study_hours}")
        st.write(f"Practice Papers = {papers_practiced}")
        st.write(f"Previous Marks = {previous_marks}")
        st.write(f"Sleeping Hours = {sleeping_hours}")

        st.write(
            f"f(x) = {beta[0]:.2f} + ({beta[1]:.2f} × {study_hours}) + "
            f"({beta[2]:.2f} × {papers_practiced}) + "
            f"({beta[3]:.2f} × {previous_marks}) + "
            f"({beta[4]:.2f} × {sleeping_hours})"
        )

        st.write(f"**Predicted Performance Index:** {predicted_performance:.2f}")
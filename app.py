import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Interactive BMI Prediction System",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown("""
<h1 style='text-align:center; color:#1F618D;'>
📊 Interactive BMI Prediction System
</h1>
<p style='text-align:center;'>
Linear Regression Model to Predict BMI using Age
</p>
""", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# Load Dataset Automatically
# --------------------------------------------------
try:
    data = pd.read_csv("bmi.csv")
except:
    st.error("bmi.csv file not found. Make sure it is in the same folder as app.py.")
    st.stop()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(data.head(), use_container_width=True)

# --------------------------------------------------
# Model Training
# --------------------------------------------------
if "age" in data.columns and "bmi" in data.columns:

    X = data[["age"]]
    y = data["bmi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.success("✅ Model Trained Successfully")

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("📈 R² Score", round(r2, 4))
    col2.metric("📉 MSE", round(mse, 4))
    col3.metric("📊 Total Records", len(data))

    st.divider()

    # --------------------------------------------------
    # Regression Plot
    # --------------------------------------------------
    st.subheader("📈 Age vs BMI Regression Plot")

    fig, ax = plt.subplots()
    sns.regplot(x="age", y="bmi", data=data, ax=ax)
    ax.set_title("Regression Line")
    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------
    # Prediction Section
    # --------------------------------------------------
    st.subheader("🔮 Predict BMI")

    age_input = st.slider("Select Age", 1, 100, 25)

    if st.button("Predict BMI"):
        prediction = model.predict([[age_input]])

        st.markdown(f"""
        <div style="
            padding:20px;
            border-radius:10px;
            background-color:#D4EFDF;
            font-size:22px;
            text-align:center;">
            Predicted BMI for Age <b>{age_input}</b> is 
            <b>{prediction[0]:.2f}</b>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("Dataset must contain 'age' and 'bmi' columns.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.markdown(
    "<center>Developed using Streamlit | Machine Learning Project</center>",
    unsafe_allow_html=True
)
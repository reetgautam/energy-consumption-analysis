import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Consumption App", page_icon="⚡", layout="wide")

st.title("⚡ Energy Consumption Prediction App")
st.sidebar.title("Navigation")

options = ["Introduction", "Data Loading", "Data Exploration", "Visualization", "Machine Learning"]
choice = st.sidebar.radio("Go to:", options)


if choice == "Introduction":
     st.markdown("""
    ## 👋 Welcome!

    This project is a **Machine Learning-based Energy Consumption Prediction System** 
    built using **Python, Scikit-learn, and Streamlit**.

    It predicts building energy usage based on environmental and operational factors.

    ---

    ### 🎯 Objective
    To estimate energy consumption using features like:
    - 🌡 Temperature
    - 💧 Humidity
    - 🏢 Square Footage
    - 👥 Occupancy
    - ❄ HVAC Usage
    - 💡 Lighting Usage
    - 🔋 Renewable Energy
    

    ---

    ### 🧠 Machine Learning Model
    - Algorithm Used: **Linear Regression**
    - Type: Supervised Learning
    - Output: Continuous value (Energy Consumption)
    - Evaluation Metrics: MAE

    ---

    ### 📂 Application Pages
    - 📤 Upload Dataset
    - 🔍 Explore Dataset
    - 📊 Visualize Dataset
    - 🤖 Predict Energy Consumption

    ---
    This system helps in **energy planning, optimization, and sustainability analysis**.
    """)



if choice == "Data Loading":
    st.subheader("📂 Upload Dataset")
    d_upload = st.file_uploader("Upload your cleaned CSV file", type=["csv"])

    if d_upload:
        data = pd.read_csv(d_upload)
        st.success("File uploaded successfully!")
        st.dataframe(data)





if choice == "Data Exploration":
    st.subheader("🔎 Data Exploration")

    d_upload = st.file_uploader("Upload your cleaned CSV file", type=["csv"])

    if d_upload:
        data = pd.read_csv(d_upload)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dataset Structure")
            st.write(f"**Rows:** {data.shape[0]}")
            st.write(f"**Columns:** {data.shape[1]}")
            st.write("**Column Names:**", list(data.columns))

        with col2:
            st.subheader("Statistical Summary")
            st.write(data.describe())




if choice == "Visualization":
    st.subheader("📊 Visualization")

    d_upload = st.file_uploader("Upload your cleaned CSV file", type=["csv"])

    if d_upload:
        data = pd.read_csv(d_upload)

        tab1, tab2, tab3 = st.tabs(["Histogram", "Scatter Plot", "Box Plot"])

        num_cols = data.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = data.select_dtypes(include=["object", "category"]).columns


        with tab1:
            st.write("### 2D Histogram / Distribution")

            x_col = st.selectbox("Select X axis column", num_cols, index=0, key="hist_x")
            y_col = st.selectbox("Select Y axis column", num_cols, index=1, key="hist_y")

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data=data, x=x_col, y=y_col, bins=20, pmax=0.9, cmap="coolwarm", ax=ax)
            ax.set_title(f"2D Histogram of {x_col} vs {y_col}")
            plt.tight_layout()
            st.pyplot(fig)


        with tab2:
            st.write("### Scatter Plot Between Two Columns")

            x_col = st.selectbox("Select X axis column", num_cols, index=0, key="scatter_x")
            y_col = st.selectbox("Select Y axis column", num_cols, index=1, key="scatter_y")

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=data, x=x_col, y=y_col, color='green', ax=ax)
            ax.set_title(f"{y_col} vs {x_col}")
            plt.tight_layout()
            st.pyplot(fig)


        with tab3:
            st.write("### Box Plot of Selected Columns")

            x_col = st.selectbox("Select column for X axis (categorical)", cat_cols if len(cat_cols) > 0 else num_cols, key="box_x")
            y_col = st.selectbox("Select column for Y axis (numeric)", num_cols, key="box_y")

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=data, x=x_col, y=y_col, palette="Set2", ax=ax)
            ax.set_title(f"Box Plot of {y_col} grouped by {x_col}")
            plt.tight_layout()
            st.pyplot(fig)






if choice == "Machine Learning":
    st.subheader("🤖 Predict Energy Consumption")

    with open("energy_consumption.pkl", "rb") as file:
        my_model = pickle.load(file)

    st.write("### Enter Input Features")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("Temperature", value=25.0)
        humidity = st.number_input("Humidity", value=50.0)
        sqft = st.number_input("Square Footage", value=1500.0)
        occupancy = st.number_input("Occupancy", value=5)

    with col2:
        hvac = st.selectbox("HVAC Usage", ["On", "Off"])
        lighting = st.selectbox("Lighting Usage", ["On", "Off"])
        holiday = st.selectbox("Holiday", ["Yes", "No"])
        day = st.selectbox(
            "Day Of Week", 
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        )

    renewable = st.number_input("Renewable Energy", value=10.0)
    hour = st.slider("Hour of Day", 0, 23, 12)
    month = st.slider("Month", 1, 12, 1)

    hvac_encoded = 1 if hvac == "On" else 0
    lighting_encoded = 1 if lighting == "On" else 0
    holiday_encoded = 1 if holiday == "Yes" else 0

    day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }
    day_encoded = day_map[day]

    weekday = day_encoded 
    is_weekend = 1 if day_encoded in [5, 6] else 0

    input_data = [[
        temperature,
        humidity,
        sqft,
        occupancy,
        hvac_encoded,
        lighting_encoded,
        renewable,
        day_encoded,
        holiday_encoded,
        hour,
        weekday,
        month,
        is_weekend
    ]]


    prediction = my_model.predict(input_data)
    prediction_value = float(prediction[0])
    st.success(f"✅ Predicted Energy Consumption: **{prediction_value:.2f}**")









#python -m streamlit run stremlit.py

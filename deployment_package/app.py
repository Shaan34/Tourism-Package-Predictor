import streamlit as st
import pandas as pd
import joblib
import os

# --- Configuration ---
# Since we upload the folder structure, files are local
MODEL_FILENAME = "xgb_holiday_model.joblib"
COLUMNS_FILENAME = "model_columns.joblib"

# --- 1. Load Model & Columns ---
@st.cache_resource
def load_assets():
    # Try loading locally first (Best for Docker/Spaces)
    if os.path.exists(MODEL_FILENAME) and os.path.exists(COLUMNS_FILENAME):
        model = joblib.load(MODEL_FILENAME)
        model_cols = joblib.load(COLUMNS_FILENAME)
        return model, model_cols
    else:
        st.error("Model or Columns file not found. Ensure 'xgb_holiday_model.joblib' and 'model_columns.joblib' are in the same folder.")
        return None, None

model, model_columns = load_assets()

# --- 2. UI Layout ---
st.title("Holiday Package Prediction App")
st.markdown("Enter customer details below to see the purchase probability.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Monthly Income", min_value=1000, value=20000)
    pitch_duration = st.number_input("Duration of Pitch (min)", min_value=5, value=15)
    trips = st.number_input("Number of Trips", min_value=0, value=2)
    children = st.number_input("Children Visiting", min_value=0, max_value=5, value=1)

with col2:
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    gender = st.selectbox("Gender", ["Female", "Male"])
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    marital = st.selectbox("Marital Status", ["Married", "Unmarried", "Divorced"])

# --- 3. Robust Preprocessing ---
def preprocess_input(age, income, pitch_duration, trips, children, city_tier, gender, product, marital):
    # 1. Create Dictionary with Raw Inputs
    # Note: We must match the column names expected by the training script before encoding
    data = {
        'Age': [age],
        'DurationOfPitch': [pitch_duration],
        'NumberOfTrips': [trips],
        'MonthlyIncome': [income],
        'CityTier': [city_tier],
        'NumberOfChildrenVisiting': [children], 
        
        # Label Encoded Features (Manual Mapping matches Training)
        'Gender': [1 if gender == 'Male' else 0],
        'ProductPitched': [{'Basic':0, 'Standard':1, 'Deluxe':2, 'Super Deluxe':3, 'King':4}[product]],
        
        # Categorical Features (passed as string for get_dummies)
        'MaritalStatus': [marital]
        # Add 'Occupation' or 'TypeofContact' here if they were in your training set
    }
    
    # 2. Convert to DataFrame
    df = pd.DataFrame(data)
    
    # 3. One-Hot Encoding
    # This simulates what happened during training
    df = pd.get_dummies(df)

    # 4. Align with Training Columns (The Critical Fix)
    # This ensures the input DF has EXACTLY the same columns as the model expects.
    # It adds missing columns (filling with 0) and removes extras.
    if model_columns:
        df = df.reindex(columns=model_columns, fill_value=0)
    
    return df

if st.button("Predict Purchase"):
    if model is not None and model_columns is not None:
        try:
            # Process Input
            input_df = preprocess_input(age, income, pitch_duration, trips, children, city_tier, gender, product, marital)
            
            # Predict
            prediction = model.predict(input_df)
            prob = model.predict_proba(input_df)[0][1]

            st.subheader("Results")
            if prediction[0] == 1:
                st.success(f"Prediction: **Likely to Buy** (Probability: {prob:.2%})")
            else:
                st.warning(f"Prediction: **Unlikely to Buy** (Probability: {prob:.2%})")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model not loaded. Please check the files.")
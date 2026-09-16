# %% [markdown]
# ## All-in-One (Python Web App) Framework using STREAMLIT

# %%
#v Load libraries for the process
import joblib
import pandas as pd
import streamlit as st
import pickle

# %%
# Load the trained model
model_insurance = joblib.load('lasso_insurance_model.pkl')
scaler_insurance = joblib.load('scaler.pkl')


# %%
# Set Streamlit layout & Page Design
st.set_page_config(page_title= 'Insurance Claim Predictor', layout= 'wide')

# ---- Title and Description -----
st.title('Insurance Claim Predictor')
st.caption('Predict the future of claims and protect your bottom line with AI-driven foresight.Our insurance predictive model analyzes complex data patterns in real-time to forecast claim amount, detect high-risk anomalies,\
            and estimate potential payouts before they happen. Seamlessly transition from reactive processing to proactive risk management—all from a single, intuitive dashboard.')

# --- Side Bar description (Generic)----
# Creates a header, then a slider below it
# In sidebar, slider only accepst numerical values while selectbox/ radio is for strings or characte values
st.sidebar.header('Medical  & Personal Information')
age = st.sidebar.slider('Age', min_value=18, max_value=100, value=30) #  The initial default value when the app loads.
gender = st.sidebar.selectbox('Gender', ['male', 'female'])
bmi = st.sidebar.slider('BMI', min_value = 1, max_value = 150, value = 50)
bloodpressure = st.sidebar.slider('Blood Pre ssure', min_value = 40, max_value = 250, value = 70)
diabetic = st.sidebar.selectbox('Diabetic',['Yes', 'No'])
region = st.sidebar.selectbox('Region of Residence (US)',['northeast', 'northwest', 'southeast', 'southwest'])
smoker = st.sidebar.selectbox('Smoker', ['Yes','No'])
children = st.sidebar.slider("Number of Children", min_value=0, max_value=10, value=0, step=1)
# ---- Define pages (generic sections) ----
prediction_page = st.Page(
    "prediction_page/section_1.py", title="Insurance Prediction Page"
)  # rename file to use underscore if possible
pg = st.navigation([prediction_page], position="top") # Collects page objects into a menu, placing it in the sidebar (default) or top header (position="top").
pg.run() # Executes and displays the UI code for whichever page the user currently selects. 



# %%
# Sidebar: just with a logo
with st.sidebar:
    st.image('https://static.investindia.gov.in/s3fs-public/2019-05/Insurance1.jpg', use_container_width= True)
    # use_container_width=True parameter forces the image to automatically resize and stretch to fill the width of its parent container


# %%

if st.button("Predict Claim Payout"):
    # 1. Prepare raw numerical data (3 columns which will be scaled)
    raw_numeric = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'bloodpressure': bloodpressure
    }])

    # 2. Scale ONLY the numeric columns
    scaled_numeric = scaler_insurance.transform(raw_numeric)

    # Convert back to a DataFrame with original names
    scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=['age', 'bmi', 'bloodpressure'])

    # 3. Prepare non-scaled binary/encoded columns
    encoded_cols = pd.DataFrame([{
        'gender': 0 if gender == "male" else 1,
        'diabetic': 0 if diabetic == "yes" else 1,
        'children': children,
        'smoker': 0 if smoker == "yes" else 1,
        'region_northwest': 1 if region == "northwest" else 0,
        'region_southeast': 1 if region == "southeast" else 0,
        'region_southwest': 1 if region == "southwest" else 0
    }])

    # 4. Concatenate scaled numeric and unscaled categorical columns
    full_features = pd.concat([scaled_numeric_df, encoded_cols], axis=1)

    # 5. Reorder columns to match the exact training set column order.
    # ``x_train`` is not available in this deployed app, so use the feature
    # names stored by scikit-learn when the model was fitted.
    feature_order = getattr(model_insurance, 'feature_names_in_', full_features.columns)
    full_features = full_features.reindex(columns=feature_order, fill_value=0)

    # 6. Predict
    prediction = model_insurance.predict(full_features)[0]
    st.success(f"Estimated Claim Amount: ${max(0.0, prediction):,.2f}")


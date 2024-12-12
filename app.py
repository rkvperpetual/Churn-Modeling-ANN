import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.title("Churn Prediction")

# Load the models and encoders
try:
    model = load_model('churn_model.h5')
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))
    onehot_encoder_geography = pickle.load(open('onehot_encoder_geography.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading models or encoders: {e}")

# User Input Form
customer_data = {
    'CreditScore': st.number_input('Credit Score', min_value=0),
    'Geography': st.selectbox('Geography', ['France', 'Spain', 'Germany']),
    'Gender': st.selectbox('Gender', ['Male', 'Female']),
    'Age': st.slider('Age',0,100),
    'Tenure': st.slider('Tenure', 0, 10),
    'Balance': st.number_input('Balance', min_value=0.0, format="%.2f"),  # Float input for Balance
    'NumOfProducts': st.number_input('Number of Products', min_value=1),
    'HasCrCard': st.selectbox('Has Credit Card', ['Yes', 'No']),
    'IsActiveMember': st.selectbox('Active Member', ['Yes', 'No']),
    'EstimatedSalary': st.number_input('Estimated Salary', min_value=0.0, format="%.2f")  # Float input for Estimated Salary
}

if st.button('Predict'):
    try:
        # Create a DataFrame from the user input
        customer_df = pd.DataFrame(customer_data, index=[0])

        # Encode Gender
        customer_df['Gender'] = label_encoder_gender.transform(customer_df['Gender'])

        # Encode Geography
        geo_encoded = onehot_encoder_geography.transform([[customer_df['Geography'][0]]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

        # Drop Geography and concatenate encoded columns
        customer_df = pd.concat([customer_df.drop('Geography', axis=1), geo_encoded_df], axis=1)

        # Encode HasCrCard and IsActiveMember (convert 'Yes'/'No' to binary)
        customer_df['HasCrCard'] = customer_df['HasCrCard'].apply(lambda x: 1 if x == 'Yes' else 0)
        customer_df['IsActiveMember'] = customer_df['IsActiveMember'].apply(lambda x: 1 if x == 'Yes' else 0)

        # Scale the data
        customer_data_scaled = scaler.transform(customer_df)

        # Make the prediction
        prediction = model.predict(customer_data_scaled)
        probability = prediction[0][0]

        # Display the prediction result and probability
        if probability > 0.5:
            st.success(f"The customer is NOT likely to churn. (Probability: {probability:.2f})")
        else:
            st.error(f"The customer is likely to churn. (Probability: {probability:.2f})")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

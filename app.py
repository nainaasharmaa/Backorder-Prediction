import streamlit as st
import pandas as pd
import pickle

# Load the trained logistic regression model from the pickle file
model_filename = 'backorder_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title('Backorder Prediction')

# Input fields for features
national_inv = st.number_input('National Inventory', value=0)
in_transit_qty = st.number_input('In Transit Quantity', value=0)
forecast_3_month = st.number_input('forecast_3_month', value=0)
forecast_6_month = st.number_input('forecast_6_month', value=0)
forecast_9_month = st.number_input('forecast_9_month', value=0)
sales_1_month = st.number_input('sales_1_month', value=0)
sales_3_month = st.number_input('sales_3_month', value=0)
sales_6_month = st.number_input('sales_6_month', value=0)
sales_9_month = st.number_input('sales_9_month', value=0)
min_bank = st.number_input('min_bank', value=0)

# Add more input fields for other features...

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'national_inv': [national_inv],
    'in_transit_qty': [in_transit_qty],
    'forecast_3_month': [forecast_3_month],
    'forecast_6_month': [forecast_6_month],
    'forecast_9_month': [forecast_9_month],
    'sales_1_month': [sales_1_month],
    'sales_3_month': [sales_3_month],
    'sales_6_month': [sales_6_month],
    'sales_9_month': [sales_9_month],
    'min_bank': [min_bank]
})

# Preprocess the input data
# input_data = preprocess_input(input_data)

# Predictions
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write('The product is likely to go on backorder.')
    else:
        st.write('The product is not likely to go on backorder.')

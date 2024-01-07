# Import modules
import streamlit as st
import requests
import datetime

# Title of the interface
st.title("Demand Forecasting")

st.markdown('Created by: Keembo')
st.markdown("---")

st.subheader("Please type the value and click the predict button to predicted demand!")

# User inputs
with st.form(key="month and year to predict"):
    year = st.number_input("Year", min_value=2017, value=datetime.datetime.now().year)
    month = st.number_input("Month", min_value=1, max_value=12,value=datetime.datetime.now().month)

    # Submit button
    submit_button = st.form_submit_button(label="Predict")
    
    if submit_button:
        data = {
            'order_year': year,
            'order_month': month
        }
        #True then predict
        with st.spinner("Predicting..."):
            
            #Send request
            response = requests.post("http://api:8000/predict", json=data)
            result = response.json()
            
            # if success
            if result['code'] == 200:
                st.success("Prediction success!,{}".format(result['prediction']))
                st.snow()
            else:
                st.error("Prediction failed!")
                st.warning("Try Again!")
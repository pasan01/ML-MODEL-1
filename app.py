# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle

# Set up the Streamlit page
st.title("Employee Performance Prediction App")
st.write("This app uses a Random Forest model to predict employee performance based on various factors such as over time, number of workers, SMV, idle time, idle men, tenure, and department.")

# Load the saved model
model_filename = 'employee_performance_model.pkl'  # Ensure this file is in your working directory
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Define the maximum Over_time value from the training dataset
max_over_time = 15120  # Set this to the actual maximum Over_time value from your dataset

# Set predefined ranges and categories based on the original data structure
over_time_range = (0, max_over_time)  # Set the range for 'Over_time' based on your data
no_of_workers_range = (2, 89)  # Minimum and maximum values for 'No_of_workers'
smv_range = (2.9, 54.56)       # Minimum and maximum values for 'SMV'
idle_time_range = (0, 270)     # Minimum and maximum values for 'Idle_time'
idle_men_range = (0, 45)       # Minimum and maximum values for 'Idle_men'
tenure_options = [1, 2, 3, 4]  # Unique values for 'Tenure'
department_options = ['Legal and Compliance', 'Sales and Marketing', 'Product Management', 'Research and Development', 'Quality Assurance']  # Add departments from your dataset

# Sidebar input widgets for prediction
st.sidebar.header("Input Features for New Prediction")
over_time = st.sidebar.slider('Over Time (minutes)', over_time_range[0], over_time_range[1], 960)  # Include Over_time input field
no_of_workers = st.sidebar.slider('Number of Workers', no_of_workers_range[0], no_of_workers_range[1], 34)
smv = st.sidebar.slider('SMV', smv_range[0], smv_range[1], 15.15)
idle_time = st.sidebar.slider('Idle Time', idle_time_range[0], idle_time_range[1], 0)
idle_men = st.sidebar.slider('Idle Men', idle_men_range[0], idle_men_range[1], 0)
tenure = st.sidebar.selectbox('Tenure', tenure_options)
department = st.sidebar.selectbox('Department', department_options)  # Include Department selection

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'Over_time': [over_time],  # Include Over_time in the input data
    'No_of_workers': [no_of_workers],
    'SMV': [smv],
    'Idle_time': [idle_time],
    'Idle_men': [idle_men],
    'Tenure': [tenure],
    'Department': [department]  # Include Department in the data for model prediction
})

# Display the input data excluding the department column if necessary
st.subheader("Input Data for Prediction")
st.write(input_data.drop(columns=['Department']))

# Make prediction using the loaded model
performance_percentage_prediction = loaded_model.predict(input_data)[0]

# Display the prediction results
st.subheader("Predicted Performance Percentage")
st.write(f"Based on the input features, the predicted performance percentage is: **{performance_percentage_prediction:.2f}%**")

# Optional: Provide some explanation about the prediction process
st.info("The prediction is made using a Random Forest model that takes into account the provided features. Adjust the inputs on the left sidebar to see how they impact the performance percentage.")

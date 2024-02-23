import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
# Load the saved model
loaded_model = pickle.load(open("model_5.pkl", "rb"))

# Function to preprocess input data
def preprocess_input_data(input_data):
    # Perform the same preprocessing steps as done during training
    # Assuming input_data is a DataFrame with the same columns as the training data
    input_data["Vehicle_Type"] = Le.fit_transform(input_data["Vehicle_Type"])
    input_data["Location_Category"] = Le.fit_transform(input_data["Location_Category"])
    input_data["Customer_Loyalty_Status"] = Le.fit_transform(input_data["Customer_Loyalty_Status"])
    input_data["Time_of_Booking"] = Le.fit_transform(input_data["Time_of_Booking"])
    return input_data

# Function to make predictions
def predict_cost(input_data):
    # Preprocess the input data
    preprocessed_data = preprocess_input_data(input_data)
    # Make predictions
    predictions = loaded_model.predict(preprocessed_data)
    return predictions

# Define the Streamlit app
def main():
    st.title("Historical Cost of Ride Prediction")
    st.write("Enter the details below to predict the historical cost of a ride.")
    # Create input fields for user input
    col1, col2 = st.columns(2)
    col3, col4=st.columns(2)
    col5, col6=st.columns(2)
    col7,col8,col9=st.columns(3)
    
    with col1:
        number_of_riders = st.number_input("Number of Riders", min_value=1)
    with col2:
        number_of_drivers = st.number_input("Number of Drivers", min_value=1)  # Example choices
    with col3:
        location_category = st.selectbox("Location Category", ["Urban", "Suburban", "Rural"])  # Example choices
    with col4:
        customer_loyalty_status = st.selectbox("Customer Loyalty Status", ["Silver", "Regular", "Gold"])  # Example choices
    with col5:
        number_of_past_rides=st.number_input("Enter numbe of past rides",min_value=0)
    with col6:
        average_rating=st.number_input("rating of previous rating",min_value=0.0)
    with col7:
        time_of_booking = st.selectbox("Time of Booking", ["Night","Evening","Afternoon","Morning"])  # Example choices
    with col8:
        vehicle_type = st.selectbox("Vehicle Type", ["premium","Economy"])
    with col9:
        travel_time_min = st.number_input("Travel Time (in minutes)", min_value=0)
    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        "Number_of_Riders": [number_of_riders],
        "Number_of_Drivers": [number_of_drivers],
        "Location_Category": [location_category],
        "Customer_Loyalty_Status": [customer_loyalty_status],
        "Number_of_Past_Rides": [number_of_past_rides],
        "Average_Ratings" : [average_rating],
        "Time_of_Booking": [time_of_booking],
        "Vehicle_Type": [vehicle_type],
        "Expected_Ride_Duration": [travel_time_min]
        
    })
    
    # When the user clicks the predict button
    if st.button("Predict"):
        # Make predictions
        predicted_cost = predict_cost(input_data)
        st.success(f"Predicted Cost of Ride: {predicted_cost}")

if __name__ == "__main__":
    main()

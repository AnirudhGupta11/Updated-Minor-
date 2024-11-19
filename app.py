import streamlit as st
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Load the trained model
with open("fraud_detection_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Title and instructions
st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's likely to be fraudulent.")

# Input fields for user
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
old_balance = st.number_input("Old Balance (Origin Account)", min_value=0.0, step=0.01)
new_balance = st.number_input("New Balance (Origin Account)", min_value=0.0, step=0.01)

# Map transaction type to numerical value as done in training
type_map = {'PAYMENT': 1, 'TRANSFER': 4, 'CASH_OUT': 2, 'DEBIT': 5, 'CASH_IN': 3}
transaction_type_value = type_map[transaction_type]

# Predict button
if st.button("Predict Fraud"):
    # Prepare input data in the format expected by the model
    input_data = np.array([[transaction_type_value, amount, old_balance, new_balance]])
    
    # Make prediction
    prediction = model.predict(input_data)

    # Prerequisite Condition
    if (amount + new_balance <= old_balance):
        st.success(f"Transaction approved! Remaining balance: {old_balance - amount}")
        if prediction[0] == 'Fraud':
            st.error("This transaction is predicted to be FRAUDULENT!")
        else:
            st.success("This transaction is predicted to be LEGITIMATE.")
    else:
        st.error("Error: Transaction can't be possible. Kindly add correct details.")

# Run Streamlit using the command in the terminal: streamlit run app.py

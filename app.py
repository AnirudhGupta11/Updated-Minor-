import streamlit as st
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

    if amount>0 and old_balance>=0 and new_balance>=0:
        if (amount + new_balance <= old_balance):
            st.success("The transaction has been approved.")
            if prediction[0] == 'Fraud':
                st.error("This transaction is predicted to be FRAUDULENT!")
            else:
                st.success("This transaction is predicted to be LEGITIMATE.")
        else:
            st.error("Transaction can't be proceeded due to incorrect entries.")
    else:
        st.error("Please enter correct details for the transaction to get completed.")

# Section for Visualizations
st.header("Visualizations")

# Visualization 1: Decision Tree (ID3)
if st.checkbox("Show Decision Tree"):
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model, feature_names=["Transaction Type", "Amount", "Old Balance", "New Balance"], filled=True, ax=ax)
    st.pyplot(fig)

# Visualization 2: Confusion Matrix with Simulated Test Data
if st.checkbox("Show Confusion Matrix"):
    st.subheader("Confusion Matrix")
    try:
        # Simulated test data
        X_test = np.array([
            [1, 1000.0, 5000.0, 4000.0],  # PAYMENT
            [2, 2000.0, 3000.0, 1000.0],  # CASH_OUT
            [4, 500.0, 2000.0, 1500.0],   # TRANSFER
            [3, 1000.0, 1500.0, 500.0],   # CASH_IN
            [5, 200.0, 1000.0, 800.0]     # DEBIT
        ])
        y_test = ["Legitimate", "Fraud", "Legitimate", "Fraud", "Legitimate"]  # Simulated true labels

        # Generate predictions
        y_pred = model.predict(X_test)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Confusion matrix cannot be displayed: {e}")

# Visualization 3: PCA
if st.checkbox("Show PCA"):
    st.subheader("PCA Visualization")
    try:
        # Simulated data for PCA
        X = np.random.rand(100, 4)  # Simulate 100 data points with 4 features
        y = np.random.choice(["Legitimate", "Fraud"], 100)  # Simulated labels
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=[1 if label == "Fraud" else 0 for label in y], cmap="viridis", s=50)
        ax.set_title("PCA Visualization")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        fig.colorbar(scatter, ax=ax, label="Class")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"PCA visualization cannot be displayed: {e}")

# Instructions to run the app
st.write("To run the app, use the command in your terminal:")
st.code("streamlit run app.py")

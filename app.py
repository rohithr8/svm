import streamlit as st
import pickle
import pandas as pd

pd.set_option("styler.render.max_elements", 8829017)
# Function to load the pipeline (cached to run only once)
@st.cache_resource
def load_pipeline():
    """Loads the pickled SVM pipeline."""
    with open('svm_fraud_detector.pkl', 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline


# Load the trained pipeline
pipeline = load_pipeline()

# --- Streamlit App Interface ---

st.title("💳 Credit Card Fraud Detector (CSV Upload)")
st.write(
    "Upload a CSV file with transaction data to check for fraud. "
    "The model will predict whether each transaction is legitimate or fraudulent."
)

# Instructions and CSV format requirements
with st.expander("ℹ️ How to format your CSV file"):
    st.write(
        "Your CSV file must contain **29 columns of numerical data**. The first row should contain the headers."
    )
    st.markdown("**Required Headers:** `V1, V2, V3, ..., V28, Amount`")
    st.write("Below is an example of what a **single row** in your CSV should look like:")
    st.code("""
-1.3598,-0.0727,2.5363,1.3781,-0.3383,0.4623,0.2395,0.0986,0.3637,0.0907,-0.5516,-0.6178,-0.9913,-0.3111,1.4681,-0.4704,0.2079,0.0257,0.4039,0.2514,-0.0183,0.2778,-0.1104,0.0669,0.1285,-0.1891,0.1335,-0.021,149.62
    """, language='text')
    # The warning message has been removed from here.

# Step 1: File Uploader
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv"
)

# Step 2: Process the file and make predictions
if uploaded_file is not None:
    try:
        # Read the uploaded data
        data = pd.read_csv(uploaded_file)

        # --- NEW: Automatically drop 'Time' and 'Class' columns if they exist ---
        columns_to_drop = ['Time', 'Class']
        # Check which of the columns to drop are actually in the DataFrame
        existing_cols_to_drop = [col for col in columns_to_drop if col in data.columns]

        if existing_cols_to_drop:
            data.drop(columns=existing_cols_to_drop, inplace=True)
            st.info(
                f"Note: The following columns were found and automatically removed: **{', '.join(existing_cols_to_drop)}**")

        st.write("Data Preview (after cleaning):")
        st.dataframe(data.head())

        # Define expected columns for validation
        expected_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']

        # Validate that the necessary 29 columns are present after cleaning
        if not all(col in data.columns for col in expected_cols):
            st.error(
                f"CSV file is missing one or more required feature columns. "
                f"Please ensure the file contains all 29 feature columns: V1 through V28, and Amount."
            )
        else:
            # Reorder columns to match model's training order
            input_data = data[expected_cols]

            # Make predictions and get scores
            predictions = pipeline.predict(input_data)
            decision_scores = pipeline.decision_function(input_data)

            # Create a results DataFrame
            results_df = data.copy()
            results_df['Prediction'] = ['Fraudulent' if p == 1 else 'Legitimate' for p in predictions]
            results_df['Confidence Score'] = decision_scores

            # Display results
            st.subheader("Prediction Results")
            st.write("Each transaction has been classified as 'Legitimate' or 'Fraudulent'.")


            # Style the DataFrame to highlight fraudulent transactions
            def highlight_fraud(s):
                return ['background-color: #ffc7ce'] * len(s) if s.Prediction == 'Fraudulent' else [''] * len(s)


            st.dataframe(results_df.style.apply(highlight_fraud, axis=1), height=500)

            # Show a summary
            fraud_count = (predictions == 1).sum()
            st.success(
                f"**Summary:** Found **{fraud_count}** potentially fraudulent transaction(s) out of **{len(data)}** total."
            )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
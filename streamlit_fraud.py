# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from feature_config import FEATURE_TYPES  # Ensure this file exists with correct mappings

# Configure Logging (Optional)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_objects(imputer_path, encoder_path, scaler_path, model_path):
    """
    Load the saved imputer, encoder, scaler, and model.
    """
    try:
        # Load Imputer
        imputer = joblib.load(imputer_path)
        logging.info(f"Loaded imputer from {imputer_path}.")
        
        # Load Encoder if it exists
        if encoder_path.exists():
            encoder = joblib.load(encoder_path)
            logging.info(f"Loaded OneHotEncoder from {encoder_path}.")
        else:
            encoder = None
            logging.info("OneHotEncoder not found. Proceeding without encoder.")
        
        # Load Scaler
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from {scaler_path}.")
        
        # Load Model
        model = joblib.load(model_path)
        logging.info(f"Loaded model from {model_path}.")
        
        return imputer, encoder, scaler, model
    
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        st.error(f"File not found: {e}")
        return None, None, None, None
    except Exception as e:
        logging.error(f"An error occurred while loading objects: {e}")
        st.error(f"An error occurred while loading objects: {e}")
        return None, None, None, None

def preprocess_input(user_input, imputer, encoder, scaler):
    """
    Preprocess the user input data.
    """
    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])
    logging.info(f"User input DataFrame:\n{input_df}")
    
    # Handle missing values
    input_numeric = input_df.select_dtypes(include=["float64", "int64"])
    input_non_numeric = input_df.select_dtypes(exclude=["float64", "int64"])
    
    input_numeric_imputed = pd.DataFrame(imputer.transform(input_numeric), columns=input_numeric.columns)
    logging.info("Missing values imputed for numerical features.")
    
    # Encode categorical variables if encoder exists
    if not input_non_numeric.empty and encoder is not None:
        input_encoded = pd.DataFrame(encoder.transform(input_non_numeric))
        input_encoded.columns = encoder.get_feature_names_out(input_non_numeric.columns)
        logging.info("Categorical features encoded using One-Hot Encoding.")
    elif not input_non_numeric.empty and encoder is None:
        logging.warning("Categorical features present but encoder was not saved. Categorical features will be ignored.")
        input_encoded = pd.DataFrame()
    else:
        input_encoded = pd.DataFrame()
        logging.info("No non-numeric features to encode.")
    
    # Combine numerical and encoded categorical features
    input_processed = pd.concat([input_numeric_imputed, input_encoded], axis=1)
    logging.info(f"Combined processed features:\n{input_processed}")
    
    # Scale features
    input_scaled = pd.DataFrame(scaler.transform(input_processed), columns=input_processed.columns)
    logging.info("Features scaled using Min-Max Scaler.")
    
    return input_scaled

def predict(model, preprocessed_data, model_features):
    """
    Predict the class of the input data.
    """
    # Handle missing columns
    missing_cols = set(model_features) - set(preprocessed_data.columns)
    for col in missing_cols:
        preprocessed_data[col] = 0
        logging.info(f"Added missing column '{col}' with default value 0.")
    
    # Reorder columns to match training data
    preprocessed_data = preprocessed_data[model_features]
    logging.info("Reordered preprocessed data columns to match model's expected features.")
    
    # Predict
    prediction = model.predict(preprocessed_data)[0]
    prediction_proba = model.predict_proba(preprocessed_data)[0, 1]
    
    return prediction, prediction_proba

def main():
    st.title("📈 Fraud Detection Prediction")
    st.write("Enter the required feature values below to predict whether the data is **FRAUDULENT** or **NON-FRAUDULENT**.")
    
    # Define paths to saved objects
    imputer_path = Path("imputer.pkl")
    encoder_path = Path("onehot_encoder.pkl")
    scaler_path = Path("scaler.pkl")
    model_path = Path("best_xgb_model_pygad.pkl")
    
    # Load preprocessors and model
    imputer, encoder, scaler, model = load_objects(imputer_path, encoder_path, scaler_path, model_path)
    
    if not all([imputer, scaler, model]):
        st.error("Failed to load necessary components. Please check the logs for more details.")
        return
    
    # Initialize a dictionary to store user inputs
    user_input = {}
    
    st.header("🔍 Input Features")
    
    # Iterate over features and create input widgets based on feature type
    for feature, f_type in FEATURE_TYPES.items():
        if f_type == "numerical":
            # Use number_input for numerical features
            user_input[feature] = st.text_input(
                f"{feature} (Numerical)",
                value="",
                help="Enter a numerical value using a dot `.` as the decimal separator (e.g., 12.34)."
            )
        elif f_type == "categorical":
            # You can define the possible categories or allow free text
            # For demonstration, let's assume categories are predefined
            # You might need to extract categories from the encoder if available
            if encoder is not None:
                # Extract categories from encoder
                categories = encoder.categories_[FEATURE_TYPES.keys().index(feature)]
                user_input[feature] = st.selectbox(
                    f"{feature} (Categorical)",
                    options=categories,
                    help="Select the appropriate category."
                )
            else:
                # If encoder is None, allow free text input
                user_input[feature] = st.text_input(
                    f"{feature} (Categorical)",
                    value="",
                    help="Enter the category as used during training."
                )
        else:
            # Handle other types if any
            user_input[feature] = st.text_input(f"{feature}", value="")
    
    # Button to trigger prediction
    if st.button("🔮 Predict"):
        try:
            # Validate and preprocess inputs
            processed_input = {}
            for feature, value in user_input.items():
                if FEATURE_TYPES[feature] == "numerical":
                    if value == "":
                        st.error(f"Please enter a value for {feature}.")
                        raise ValueError(f"Missing input for {feature}.")
                    # Replace comma with dot and convert to float
                    corrected_value = value.replace(',', '.')
                    processed_input[feature] = float(corrected_value)
                elif FEATURE_TYPES[feature] == "categorical":
                    if value == "":
                        st.error(f"Please select a category for {feature}.")
                        raise ValueError(f"Missing input for {feature}.")
                    processed_input[feature] = value
                else:
                    # Handle other types if necessary
                    processed_input[feature] = value
            
            # Preprocess input
            preprocessed_data = preprocess_input(processed_input, imputer, encoder, scaler)
            
            # Retrieve feature names from the model
            try:
                model_features = model.get_booster().feature_names
            except AttributeError:
                # For scikit-learn API
                model_features = model.feature_names_in_
            
            # Make prediction
            prediction, prediction_proba = predict(model, preprocessed_data, model_features)
            
            # Display result
            if prediction == 1:
                st.success(f"The input data is classified as **FRAUDULENT** with a probability of {prediction_proba:.2f}.")
            else:
                st.info(f"The input data is classified as **NON-FRAUDULENT** with a probability of {1 - prediction_proba:.2f}.")
        
        except ValueError as ve:
            st.error(f"Input Error: {ve}")
        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

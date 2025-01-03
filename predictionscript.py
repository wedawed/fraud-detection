import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def load_objects(imputer_path, encoder_path, scaler_path, model_path):
    """
    Load the saved imputer, encoder, scaler, and model.
    
    Parameters:
    - imputer_path (Path): Path to the saved imputer.
    - encoder_path (Path): Path to the saved OneHotEncoder.
    - scaler_path (Path): Path to the saved scaler.
    - model_path (Path): Path to the saved trained model.
    
    Returns:
    - tuple: imputer, encoder, scaler, model
    """
    try:
        imputer = joblib.load(imputer_path)
        logging.info(f"Loaded imputer from {imputer_path}.")
        
        encoder = joblib.load(encoder_path)
        logging.info(f"Loaded OneHotEncoder from {encoder_path}.")
        
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from {scaler_path}.")
        
        model = joblib.load(model_path)
        logging.info(f"Loaded model from {model_path}.")
        
        return imputer, encoder, scaler, model
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading objects: {e}")
        raise

def get_user_input(feature_names):
    """
    Prompt the user to input values for each feature.
    
    Parameters:
    - feature_names (list): List of feature names required for prediction.
    
    Returns:
    - dict: Dictionary of feature values.
    """
    user_data = {}
    print("Please enter the following feature values:")
    for feature in feature_names:
        value = input(f"{feature}: ")
        # Attempt to convert to float or keep as string based on expected type
        try:
            # Assuming numerical features are float or int
            if '.' in value or 'e' in value.lower():
                user_data[feature] = float(value)
            else:
                user_data[feature] = int(value)
        except ValueError:
            # If conversion fails, treat as string (categorical)
            user_data[feature] = value
    return user_data

def preprocess_input(user_input, imputer, encoder, scaler):
    """
    Preprocess the user input data.
    
    Parameters:
    - user_input (dict): Dictionary of user-input feature values.
    - imputer: Loaded imputer object.
    - encoder: Loaded OneHotEncoder object.
    - scaler: Loaded scaler object.
    
    Returns:
    - pandas.DataFrame: Preprocessed data ready for prediction.
    """
    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])
    logging.info(f"User input DataFrame:\n{input_df}")
    
    # Handle missing values
    input_numeric = input_df.select_dtypes(include=["float64", "int64"])
    input_non_numeric = input_df.select_dtypes(exclude=["float64", "int64"])
    
    input_numeric_imputed = pd.DataFrame(imputer.transform(input_numeric), columns=input_numeric.columns)
    logging.info("Missing values imputed for numerical features.")
    
    # Encode categorical variables
    if not input_non_numeric.empty:
        input_encoded = pd.DataFrame(encoder.transform(input_non_numeric))
        input_encoded.columns = encoder.get_feature_names_out(input_non_numeric.columns)
        logging.info("Categorical features encoded using One-Hot Encoding.")
    else:
        input_encoded = pd.DataFrame()
    
    # Combine numerical and encoded categorical features
    input_processed = pd.concat([input_numeric_imputed, input_encoded], axis=1)
    logging.info(f"Combined processed features:\n{input_processed}")
    
    # Scale features
    input_scaled = pd.DataFrame(scaler.transform(input_processed), columns=input_processed.columns)
    logging.info("Features scaled using Min-Max Scaler.")
    
    return input_scaled

def predict(model, preprocessed_data):
    """
    Predict the class of the input data.
    
    Parameters:
    - model: Loaded trained model.
    - preprocessed_data (pandas.DataFrame): Preprocessed input data.
    
    Returns:
    - int: Predicted class (1 for fraud, 0 for non-fraud).
    """
    prediction = model.predict(preprocessed_data)
    prediction_proba = model.predict_proba(preprocessed_data)[:, 1]
    return prediction[0], prediction_proba[0]

def main_prediction():
    """
    Main function to execute the prediction workflow.
    """
    # Define paths to saved objects
    imputer_path = Path("imputer.pkl")
    encoder_path = Path("onehot_encoder.pkl")
    scaler_path = Path("scaler.pkl")
    model_path = Path("best_xgb_model_pygad.pkl")
    
    # Load preprocessors and model
    imputer, encoder, scaler, model = load_objects(imputer_path, encoder_path, scaler_path, model_path)
    
    # Define feature names as per training data
    # Replace with your actual feature names used during training
    feature_names = [
        "feature1", "feature2", "feature3",  # Numerical features
        "categorical_feature1", "categorical_feature2"  # Categorical features
        # Add all necessary features
    ]
    
    # Get user input
    user_input = get_user_input(feature_names)
    
    # Preprocess input
    preprocessed_data = preprocess_input(user_input, imputer, encoder, scaler)
    
    # Ensure that the preprocessed data has the same columns as the training data
    # If there are missing columns due to one-hot encoding, add them with zeros
    model_features = model.get_booster().feature_names
    missing_cols = set(model_features) - set(preprocessed_data.columns)
    for col in missing_cols:
        preprocessed_data[col] = 0
        logging.info(f"Added missing column '{col}' with default value 0.")
    
    # Reorder columns to match training data
    preprocessed_data = preprocessed_data[model_features]
    logging.info(f"Reordered preprocessed data columns to match model's expected features.")
    
    # Predict
    prediction, prediction_proba = predict(model, preprocessed_data)
    
    # Display result
    if prediction == 1:
        print(f"The input data is classified as FRAUDULENT with a probability of {prediction_proba:.2f}.")
    else:
        print(f"The input data is classified as NON-FRAUDULENT with a probability of {1 - prediction_proba:.2f}.")

if __name__ == "__main__":
    main_prediction()

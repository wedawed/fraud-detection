import joblib
import pandas as pd
import logging
from pathlib import Path

# Configure Logging
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
    """
    user_data = {}
    print("Please enter the following feature values:")
    for feature in feature_names:
        value = input(f"{feature}: ")
        # Attempt to convert to float or keep as string based on expected type
        try:
            # Adjust the conversion based on your feature types
            user_data[feature] = float(value)
        except ValueError:
            # If conversion fails, treat as string (categorical)
            user_data[feature] = value
    return user_data

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

    # Encode categorical variables
    if not input_non_numeric.empty and encoder is not None:
        input_encoded = pd.DataFrame(encoder.transform(input_non_numeric))
        input_encoded.columns = encoder.get_feature_names_out(input_non_numeric.columns)
        logging.info("Categorical features encoded using One-Hot Encoding.")
    else:
        input_encoded = pd.DataFrame()
        if input_non_numeric.empty:
            logging.info("No non-numeric features to encode.")
        else:
            logging.warning("Encoder is None, but non-numeric features are present.")

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

def main_inference():
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
        "feature1", "feature2", "feature3",
        "categorical_feature1", "categorical_feature2"
        # Add all necessary features
    ]

    # Get user input
    user_input = get_user_input(feature_names)

    # Preprocess input
    preprocessed_data = preprocess_input(user_input, imputer, encoder, scaler)

    # Retrieve feature names from the model
    try:
        model_features = model.get_booster().feature_names
    except AttributeError:
        # For scikit-learn API
        model_features = model.feature_names_in_

    # Predict
    prediction, prediction_proba = predict(model, preprocessed_data, model_features)

    # Display result
    if prediction == 1:
        print(f"The input data is classified as FRAUDULENT with a probability of {prediction_proba:.2f}.")
    else:
        print(f"The input data is classified as NON-FRAUDULENT with a probability of {1 - prediction_proba:.2f}.")

if __name__ == "__main__":
    main_inference()

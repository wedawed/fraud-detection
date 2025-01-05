# Import Necessary Libraries
import os
import json
import logging
from pathlib import Path
from collections import Counter
from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb
from xgboost import XGBClassifier

import pygad

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fraud_detection.log"),
        logging.StreamHandler()
    ]
)

def load_fraudulent_emitens(file_path):
    """
    Load the list of fraudulent emitens from a JSON file.

    Parameters:
    - file_path (str or Path): Path to the JSON file containing fraudulent emitens.

    Returns:
    - list: List of fraudulent emiten identifiers.
    """
    try:
        with open(file_path, 'r') as f:
            fraudulent_emitens = json.load(f)
        logging.info(f"Loaded {len(fraudulent_emitens)} fraudulent emitens.")
        return fraudulent_emitens
    except FileNotFoundError:
        logging.error(f"Fraudulent emitens file not found at {file_path}.")
        raise
    except json.JSONDecodeError:
        logging.error("Error decoding the fraudulent emitens JSON file.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading fraudulent emitens: {e}")
        raise

def load_data(file_path):
    """
    Load the dataset from an Excel file.

    Parameters:
    - file_path (str or Path): Path to the Excel file containing the dataset.

    Returns:
    - pandas.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_excel(file_path)
        logging.info(f"Data loaded successfully from {file_path}. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        logging.error(f"Data file not found at {file_path}.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        raise

def preprocess_data(df, fraudulent_emitens):
    """
    Preprocess the dataset by handling missing values, encoding categorical variables,
    feature engineering, and labeling.

    Parameters:
    - df (pandas.DataFrame): The raw dataset.
    - fraudulent_emitens (list): List of fraudulent emiten identifiers.

    Returns:
    - pandas.DataFrame: Preprocessed dataset ready for modeling.
    - SimpleImputer: Fitted imputer object for numerical features.
    - OneHotEncoder: Fitted encoder object for categorical features (or None if no categorical features).
    """
    logging.info("Starting data preprocessing...")

    # Feature Engineering: Create 'emiten' first
    if 'tahun' in df.columns and 'kode' in df.columns:
        df['tahun'] = df['tahun'].astype(int)
        df['kode'] = df['kode'].astype(str)
        df['emiten'] = df['kode'] + " " + df['tahun'].astype(str)
        logging.info("Feature engineering completed: 'emiten' column created.")
    else:
        logging.error("'tahun' or 'kode' columns not found for feature engineering.")
        raise KeyError("'tahun' or 'kode' columns missing in the raw data.")

    # Label Encoding: Create target variable
    if 'emiten' in df.columns:
        df['emiten_label'] = df['emiten'].apply(lambda x: 1 if x in fraudulent_emitens else 0)
        logging.info("Label encoding completed: 'emiten_label' column created.")
    else:
        logging.error("'emiten' column not found after preprocessing.")
        raise KeyError("'emiten' column missing in the processed data.")

    # Drop Unnecessary Columns (including 'kode', 'tahun', 'emiten')
    columns_to_drop = [
        'nama', 'kode', 'tahun', "mata_uang", "kurs", "tanggal_listing",
        "sektor", "tahun_listing", 'total_pendapatan','laba_sebelum_pajak',
        'laba_bersih_tahun_berjalan', 'laba_bersih_tahun_berjalan_idr',
        'aset_t-1', 'close', 'adj_price', 'share_outstanding', 'market_cap',
        'cce', 'dar','laba_kotor', 'laba_operasional', 'aset_lancar',
        'liabilitas_jangka_pendek','ato', 'cash_ratio', 'current_ratio',
        'gpm', 'aset', 'ekuitas', 'liabilitas',
        'emiten'  # Drop the original 'emiten' string column
    ]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_processed = df.drop(columns=existing_columns_to_drop)
    logging.info(f"Dropped {len(existing_columns_to_drop)} unnecessary columns.")

    # Separate features and target
    if 'emiten_label' not in df_processed.columns:
        logging.error("'emiten_label' column not found after preprocessing.")
        raise KeyError("'emiten_label' column missing in the processed data.")

    X = df_processed.drop(columns=['emiten_label'])
    y = df_processed['emiten_label']

    # Separate numeric and non-numeric columns
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    non_numeric_cols = X.select_dtypes(exclude=["float64", "int64"]).columns.tolist()
    logging.info(f"Numeric columns: {len(numeric_cols)}")
    logging.info(f"Non-numeric columns: {len(non_numeric_cols)}")

    # Handle missing values for numeric columns
    imputer = SimpleImputer(strategy="median")
    df_numeric = pd.DataFrame(imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)
    logging.info("Missing values imputed for numeric columns using median strategy.")

    # Handle non-numeric columns
    df_non_numeric = X[non_numeric_cols].copy()
    if not df_non_numeric.empty:
        # Updated parameter: 'sparse_output' instead of 'sparse'
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_non_numeric = pd.DataFrame(encoder.fit_transform(df_non_numeric))
        encoded_non_numeric.columns = encoder.get_feature_names_out(non_numeric_cols)
        df_numeric = pd.concat([df_numeric, encoded_non_numeric], axis=1)
        logging.info("Non-numeric columns encoded using One-Hot Encoding.")
    else:
        logging.info("No non-numeric columns to encode.")
        encoder = None  # If no categorical features, set encoder to None

    # Combine features
    df_final = df_numeric.copy()

    # Add target
    df_final['emiten_label'] = y

    # Verify class distribution
    class_counts = df_final['emiten_label'].value_counts()
    logging.info(f"Class Distribution after encoding:\n{class_counts}")

    # Additional Debugging Logs
    logging.info(f"Columns after label encoding and before returning: {df_final.columns.tolist()}")

    logging.info("Data preprocessing completed.")

    # Return the processed DataFrame and the fitted preprocessors
    return df_final, imputer, encoder


def split_data(df, test_size=0.3, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - df (pandas.DataFrame): The preprocessed dataset.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=['emiten_label'])
    y = df['emiten_label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info(f"Train-Test split: Train={X_train.shape}, Test={X_test.shape}")
    logging.info(f"Training Labels Distribution: {Counter(y_train)}")
    logging.info(f"Test Labels Distribution: {Counter(y_test)}")
    return X_train, X_test, y_train, y_test

def build_resampling_pipeline():
    """
    Build a pipeline for handling class imbalance using SMOTE and Random Under Sampling.

    Returns:
    - imblearn.pipeline.Pipeline: Resampling pipeline.
    """
    resampling_pipeline = ImbPipeline(steps=[
        ('over', SMOTE(sampling_strategy=0.2, random_state=42)),
        ('under', RandomUnderSampler(sampling_strategy=0.5, random_state=42))
    ])
    logging.info("Resampling pipeline created with SMOTE and Random Under Sampling.")
    return resampling_pipeline

def scale_features(X_train, X_test):
    """
    Scale features using Min-Max Scaling.

    Parameters:
    - X_train (pandas.DataFrame): Training features.
    - X_test (pandas.DataFrame): Testing features.

    Returns:
    - scaler (MinMaxScaler): Fitted scaler object.
    - X_train_scaled (pandas.DataFrame): Scaled training data.
    - X_test_scaled (pandas.DataFrame): Scaled testing data.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    logging.info("Feature scaling completed using Min-Max Scaler.")
    return scaler, X_train_scaled, X_test_scaled

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate the trained model on the test set and display performance metrics and plots.

    Parameters:
    - model: Trained machine learning model.
    - X_test (pandas.DataFrame): Scaled test features.
    - y_test (pandas.Series): True labels for the test set.
    - model_name (str): Name of the model for display purposes.
    """
    logging.info(f"Evaluating model: {model_name}")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Log Metrics
    logging.info(f"=== {model_name} Evaluation ===")
    logging.info(f"Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"Precision: {precision * 100:.2f}%")
    logging.info(f"Recall (Sensitivity): {recall * 100:.2f}%")
    logging.info(f"F1-Score: {f1 * 100:.2f}%")
    logging.info(f"ROC-AUC: {roc_auc:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    logging.info("Confusion Matrix:")
    logging.info(f"\n{cm}")

    # Classification Report
    report = classification_report(y_test, y_pred, zero_division=0)
    logging.info(f"Classification Report:\n{report}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random chance
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

    # Plot Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    plt.figure(figsize=(6, 4))
    plt.plot(recall_vals, precision_vals, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='lower left')
    plt.show()

def plot_feature_importance(model, feature_names, model_name="Model", top_n=20, normalize=False):
    """
    Plot the top_n feature importances from the trained model.

    Parameters:
    - model: Trained machine learning model.
    - feature_names (list): List of feature names.
    - model_name (str): Name of the model for display purposes.
    - top_n (int): Number of top features to display.
    - normalize (bool): Whether to normalize the feature importances.
    """
    importances = model.feature_importances_
    
    if normalize:
        importances = importances / importances.sum()
        logging.info("Feature importances normalized.")

    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top_features = feature_importances.head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
    plt.title(f'Feature Importance - {model_name}')
    plt.xlabel('Normalized Importance Score' if normalize else 'Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def shap_feature_importance(model, X, model_name="Model", sample_size=100):
    """
    Plot SHAP feature importances for the trained model.

    Parameters:
    - model: Trained machine learning model.
    - X (pandas.DataFrame): Feature DataFrame used for SHAP analysis.
    - model_name (str): Name of the model for display purposes.
    - sample_size (int): Number of samples to use for SHAP analysis.
    """
    try:
        logging.info(f"Generating SHAP values for {model_name}...")
        explainer = shap.Explainer(model)
        sample_X = X.sample(n=min(sample_size, len(X)), random_state=42)
        shap_values = explainer(sample_X)
        
        # SHAP Summary Plot
        shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.tight_layout()
        plt.show()
        logging.info("SHAP feature importance plot generated.")
    except Exception as e:
        logging.error(f"An error occurred during SHAP feature importance plotting: {e}")

def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness function for the Genetic Algorithm to evaluate XGBoost hyperparameters.

    Parameters:
    - ga_instance: Instance of the pygad.GA class.
    - solution (list): List of hyperparameter values.
    - solution_idx (int): Index of the solution.

    Returns:
    - float: Fitness score (mean F1-Score from cross-validation).
    """
    try:
        # Decode the solution to hyperparameters
        hyperparams = {
            'learning_rate': solution[0],
            'n_estimators': int(solution[1]),
            'max_depth': int(solution[2]),
            'min_child_weight': int(solution[3]),
            'subsample': solution[4],
            'colsample_bytree': solution[5],
            'gamma': solution[6],
            'reg_alpha': solution[7],
            'reg_lambda': solution[8],
            'scale_pos_weight': int(solution[9])
        }

        # Initialize the XGBoost classifier with the given hyperparameters
        model = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            **hyperparams
        )

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='f1')

        # The fitness function is the mean F1-Score
        return scores.mean()
    except Exception as e:
        logging.error(f"Error in fitness function: {e}")
        return 0  # Assign a minimal fitness score in case of error
    
def save_objects(imputer, encoder, scaler, model, paths):
    """
    Save the imputer, encoder, scaler, and model to disk.

    Parameters:
    - imputer: Fitted imputer object.
    - encoder: Fitted OneHotEncoder object.
    - scaler: Fitted scaler object.
    - model: Trained machine learning model.
    - paths (dict): Dictionary with keys 'imputer', 'encoder', 'scaler', 'model' and their respective paths.
    """
    try:
        joblib.dump(imputer, paths['imputer'])
        logging.info(f"Imputer saved to {paths['imputer']}.")

        if encoder is not None:
            joblib.dump(encoder, paths['encoder'])
            logging.info(f"OneHotEncoder saved to {paths['encoder']}.")
        else:
            logging.info("No OneHotEncoder to save.")

        joblib.dump(scaler, paths['scaler'])
        logging.info(f"Scaler saved to {paths['scaler']}.")

        joblib.dump(model, paths['model'])
        logging.info(f"Trained model saved to {paths['model']}.")
    except Exception as e:
        logging.error(f"Error saving objects: {e}")
        raise


def perform_hyperparameter_tuning(X, y):
    """
    Perform hyperparameter tuning using a Genetic Algorithm.

    Parameters:
    - X (pandas.DataFrame): Resampled and scaled training features.
    - y (pandas.Series): Resampled training labels.

    Returns:
    - dict: Best hyperparameters found by the GA.
    """
    logging.info("Starting hyperparameter tuning using Genetic Algorithm...")

    # Define gene space for each hyperparameter
    gene_space = [
        {'low': 0.01, 'high': 0.2},     # learning_rate (float)
        {'low': 100, 'high': 500, 'step': 1},      # n_estimators (int)
        {'low': 3, 'high': 8, 'step': 1},          # max_depth (int)
        {'low': 1, 'high': 5, 'step': 1},          # min_child_weight (int)
        {'low': 0.6, 'high': 1.0},      # subsample (float)
        {'low': 0.6, 'high': 1.0},      # colsample_bytree (float)
        {'low': 0, 'high': 0.4},        # gamma (float)
        {'low': 0, 'high': 1.0},        # reg_alpha (float)
        {'low': 0, 'high': 1.0},        # reg_lambda (float)
        {'low': 1, 'high': 5, 'step':1}           # scale_pos_weight (int)
    ]

    # Initialize the GA instance
    ga_instance = pygad.GA(
        num_generations=50,              # Number of generations
        num_parents_mating=10,           # Number of parents to mate
        fitness_func=fitness_func,        # Fitness function
        sol_per_pop=20,                   # Population size
        num_genes=10,                     # Number of hyperparameters
        gene_space=gene_space,            # Gene space
        gene_type=[float, int, int, int, float, float, float, float, float, int],
        mutation_percent_genes=10,        # Percentage of genes to mutate
        crossover_type="uniform",         # Crossover type
        mutation_type="random",           # Mutation type
        random_mutation_min_val=-0.1,     # Min mutation value
        random_mutation_max_val=0.1,      # Max mutation value
        on_generation=lambda ga: logging.info(
            f"Generation {ga.generations_completed}: Best Fitness = {ga.best_solution()[1]:.4f}"
        ),
        parallel_processing=cpu_count(),  # Enable parallel processing
        suppress_warnings=True,           # Suppress warnings
        stop_criteria=["saturate_10"]     # Stop if no improvement over 10 generations
    )

    # Run the GA
    ga_instance.run()

    # Retrieve the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    logging.info(f"Best Solution: {solution}")
    logging.info(f"Best Solution Fitness (F1-Score): {solution_fitness:.4f}")

    # Decode the best solution into hyperparameters
    best_hyperparams = {
        'learning_rate': solution[0],
        'n_estimators': int(solution[1]),
        'max_depth': int(solution[2]),
        'min_child_weight': int(solution[3]),
        'subsample': solution[4],
        'colsample_bytree': solution[5],
        'gamma': solution[6],
        'reg_alpha': solution[7],
        'reg_lambda': solution[8],
        'scale_pos_weight': int(solution[9])
    }

    logging.info("Best Hyperparameters from GA:")
    for param, value in best_hyperparams.items():
        logging.info(f"{param}: {value}")

    return best_hyperparams

def main():
    """
    Main function to execute the fraud detection pipeline.
    """
    # Paths Configuration
    data_path = Path("C:/Users/User/OneDrive - UNIVERSITAS INDONESIA/Perkuliahan/Semester 9/Tesis/Data Processing/Data/fs_data_clean.xlsx")
    fraudulent_emitens_path = Path("fraudulent_emitens.json")  # Ensure this JSON file exists with the list

    # Load Fraudulent Emitens
    fraudulent_emitens = load_fraudulent_emitens(fraudulent_emitens_path)

    # Load Data
    fs_data = load_data(data_path)

    # Preprocess Data
    df_processed, imputer, encoder = preprocess_data(fs_data, fraudulent_emitens)

    # Split Data
    X_train, X_test, y_train, y_test = split_data(df_processed, test_size=0.3, random_state=42)

    # Scale Features using the scale_features function
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Handle Class Imbalance
    resampling_pipeline = build_resampling_pipeline()
    X_resampled, y_resampled = resampling_pipeline.fit_resample(X_train_scaled, y_train)
    logging.info(f"After Resampling: {Counter(y_resampled)}")

    # Hyperparameter Tuning with GA
    best_hyperparams = perform_hyperparameter_tuning(X_resampled, y_resampled)

    # Initialize and Train the Best Model
    best_model = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        **best_hyperparams
    )
    best_model.fit(X_resampled, y_resampled)
    logging.info("Best XGBoost model trained with optimized hyperparameters.")

    # Save Preprocessors and Model
    paths = {
        'imputer': Path("imputer.pkl"),
        'encoder': Path("onehot_encoder.pkl"),
        'scaler': Path("scaler.pkl"),
        'model': Path("best_xgb_model_pygad.pkl")
    }
    save_objects(imputer, encoder, scaler, best_model, paths)

    # Evaluate the Model
    evaluate_model(best_model, X_test_scaled, y_test, model_name="GA Optimized XGBClassifier")

    # Save the Trained Model
    model_save_path = Path("best_xgb_model_pygad.pkl")
    joblib.dump(best_model, model_save_path)
    logging.info(f"Trained model saved to {model_save_path}.")

    # Feature Importance
    plot_feature_importance(
        model=best_model,
        feature_names=X_train.columns.tolist(),
        model_name="GA Optimized XGBClassifier",
        top_n=20
    )

    # SHAP Feature Importance
    shap_feature_importance(
        model=best_model,
        X=X_test_scaled,
        model_name="GA Optimized XGBClassifier",
        sample_size=100
    )

if __name__ == "__main__":
    main()
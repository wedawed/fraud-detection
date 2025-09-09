# Import Necessary Libraries
import os
import json
import logging
from pathlib import Path
from collections import Counter
from multiprocessing import cpu_count
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
)

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
        logging.FileHandler("model_comparison.log"),
        logging.StreamHandler()
    ]
)

# Import functions from your existing module
# You'll need to make sure these functions are available, either by importing from your original script
# or by copying them here
# For now, we'll re-implement the essential functions to ensure this script can run independently

def load_fraudulent_emitens(file_path):
    """
    Load the list of fraudulent emitens from a JSON file.
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

    logging.info("Data preprocessing completed.")

    # Return the processed DataFrame and the fitted preprocessors
    return df_final, imputer, encoder

def split_data(df, test_size=0.3, random_state=42):
    """
    Split the dataset into training and testing sets.
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
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    logging.info("Feature scaling completed using Min-Max Scaler.")
    return scaler, X_train_scaled, X_test_scaled

def perform_grid_search_xgboost(X, y):
    """
    Perform hyperparameter tuning using Grid Search for XGBoost.
    """
    logging.info("Starting hyperparameter tuning using Grid Search for XGBoost...")
    
    # Define parameter grid - using a smaller grid for demonstration
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.5],
        'reg_lambda': [0.5, 1.0],
        'scale_pos_weight': [1, 3]
    }
    
    # Initialize XGBoost classifier
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    # Initialize Grid Search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        verbose=1,
        n_jobs=-1
    )
    
    # Start timing
    start_time = time.time()
    
    # Fit Grid Search
    grid_search.fit(X, y)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Get best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logging.info(f"Grid Search completed in {elapsed_time:.2f} seconds.")
    logging.info(f"Best F1-Score: {best_score:.4f}")
    logging.info("Best Parameters:")
    for param, value in best_params.items():
        logging.info(f"{param}: {value}")
    
    return best_params, grid_search.best_estimator_

def perform_grid_search_rf(X, y):
    """
    Perform hyperparameter tuning using Grid Search for Random Forest.
    """
    logging.info("Starting hyperparameter tuning using Grid Search for Random Forest...")
    
    # Define parameter grid - using a smaller grid for demonstration
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Initialize Random Forest classifier
    rf_model = RandomForestClassifier(random_state=42)
    
    # Initialize Grid Search
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        verbose=1,
        n_jobs=-1
    )
    
    # Start timing
    start_time = time.time()
    
    # Fit Grid Search
    grid_search.fit(X, y)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Get best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logging.info(f"Grid Search completed in {elapsed_time:.2f} seconds.")
    logging.info(f"Best F1-Score: {best_score:.4f}")
    logging.info("Best Parameters:")
    for param, value in best_params.items():
        logging.info(f"{param}: {value}")
    
    return best_params, grid_search.best_estimator_

def perform_ga_rf(X, y):
    """
    Perform hyperparameter tuning using Genetic Algorithm for Random Forest.
    """
    logging.info("Starting hyperparameter tuning using Genetic Algorithm for Random Forest...")

    # Define gene space for each hyperparameter
    gene_space = [
        {'low': 100, 'high': 500, 'step': 10},         # n_estimators (int)
        {'low': 5, 'high': 30, 'step': 1},             # max_depth (int)
        {'low': 2, 'high': 10, 'step': 1},             # min_samples_split (int)
        {'low': 1, 'high': 5, 'step': 1},              # min_samples_leaf (int)
        [0.5, 0.7, 'sqrt', 'log2', None],              # max_features (mixed)
        [True, False],                                  # bootstrap (bool)
        ['balanced', 'balanced_subsample', None]        # class_weight (categorical)
    ]

    def fitness_func(ga_instance, solution, solution_idx):
        """
        Fitness function for the Genetic Algorithm to evaluate RF hyperparameters.
        """
        try:
            # Decode the solution to hyperparameters
            hyperparams = {
                'n_estimators': int(solution[0]),
                'max_depth': int(solution[1]) if solution[1] < 30 else None,
                'min_samples_split': int(solution[2]),
                'min_samples_leaf': int(solution[3]),
                'max_features': solution[4],
                'bootstrap': solution[5],
                'class_weight': solution[6]
            }

            # Initialize the Random Forest classifier with the given hyperparameters
            model = RandomForestClassifier(random_state=42, **hyperparams)

            # Perform cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

            # The fitness function is the mean F1-Score
            return scores.mean()
        except Exception as e:
            logging.error(f"Error in fitness function: {e}")
            return 0  # Assign a minimal fitness score in case of error

    # Initialize the GA instance
    ga_instance = pygad.GA(
        num_generations=30,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=7,
        gene_space=gene_space,
        gene_type=[int, int, int, int, object, bool, object],
        mutation_percent_genes=10,
        crossover_type="uniform",
        mutation_type="random",
        on_generation=lambda ga: logging.info(
            f"Generation {ga.generations_completed}: Best Fitness = {ga.best_solution()[1]:.4f}"
        ),
        parallel_processing=cpu_count(),
        suppress_warnings=True,
        stop_criteria=["saturate_10"]
    )

    # Start timing
    start_time = time.time()
    
    # Run the GA
    ga_instance.run()
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Retrieve the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    logging.info(f"GA completed in {elapsed_time:.2f} seconds.")
    logging.info(f"Best Solution: {solution}")
    logging.info(f"Best Solution Fitness (F1-Score): {solution_fitness:.4f}")

    # Decode the best solution into hyperparameters
    best_hyperparams = {
        'n_estimators': int(solution[0]),
        'max_depth': int(solution[1]) if solution[1] < 30 else None,
        'min_samples_split': int(solution[2]),
        'min_samples_leaf': int(solution[3]),
        'max_features': solution[4],
        'bootstrap': solution[5],
        'class_weight': solution[6]
    }

    logging.info("Best Hyperparameters from GA:")
    for param, value in best_hyperparams.items():
        logging.info(f"{param}: {value}")

    # Create and train the best model
    best_model = RandomForestClassifier(random_state=42, **best_hyperparams)
    best_model.fit(X, y)

    return best_hyperparams, best_model

def compare_models(models_dict, X_test, y_test):
    """
    Compare different models using various metrics.

    Parameters:
    - models_dict (dict): Dictionary of model names and their corresponding trained models.
    - X_test (pandas.DataFrame): Test features.
    - y_test (pandas.Series): Test labels.

    Returns:
    - pandas.DataFrame: Comparison results.
    """
    logging.info("Comparing models...")
    
    results = []
    
    for model_name, model in models_dict.items():
        logging.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Calculate PR-AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['PR-AUC'] = auc(recall_vals, precision_vals)
        
        # Add to results
        results.append(metrics)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by F1-Score
    results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    return results_df

def plot_model_comparison(results_df):
    """
    Plot model comparison results.

    Parameters:
    - results_df (pandas.DataFrame): Comparison results.
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    
    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        
        # Add values on top of bars
        for j, v in enumerate(results_df[metric]):
            ax.text(j, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

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
        {'low': 0.01, 'high': 0.2},                     # learning_rate (float)
        {'low': 100, 'high': 500, 'step': 1},           # n_estimators (int)
        {'low': 3, 'high': 8, 'step': 1},               # max_depth (int)
        {'low': 1, 'high': 5, 'step': 1},               # min_child_weight (int)
        {'low': 0.6, 'high': 1.0},                      # subsample (float)
        {'low': 0.6, 'high': 1.0},                      # colsample_bytree (float)
        {'low': 0, 'high': 0.4},                        # gamma (float)
        {'low': 0, 'high': 1.0},                        # reg_alpha (float)
        {'low': 0, 'high': 1.0},                        # reg_lambda (float)
        {'low': 1, 'high': 5, 'step': 1}                # scale_pos_weight (int)
    ]

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
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

            # The fitness function is the mean F1-Score
            return scores.mean()
        except Exception as e:
            logging.error(f"Error in fitness function: {e}")
            return 0  # Assign a minimal fitness score in case of error

    # Initialize the GA instance
    ga_instance = pygad.GA(
        num_generations=50,               # Number of generations
        num_parents_mating=10,            # Number of parents to mate
        fitness_func=fitness_func,        # Fitness function
        sol_per_pop=20,                    # Population size
        num_genes=10,                      # Number of hyperparameters
        gene_space=gene_space,             # Gene space
        gene_type=[float, int, int, int, float, float, float, float, float, int],
        mutation_percent_genes=10,         # Percentage of genes to mutate
        crossover_type="uniform",          # Crossover type
        mutation_type="random",            # Mutation type
        random_mutation_min_val=-0.1,      # Min mutation value
        random_mutation_max_val=0.1,       # Max mutation value
        on_generation=lambda ga: logging.info(
            f"Generation {ga.generations_completed}: Best Fitness = {ga.best_solution()[1]:.4f}"
        ),
        parallel_processing=cpu_count(),   # Enable parallel processing
        suppress_warnings=True,            # Suppress warnings
        stop_criteria=["saturate_10"]      # Stop if no improvement over 10 generations
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

def run_comparison():
    """
    Main function to run the model comparison.
    """
    # Paths Configuration
    data_path = Path("C:/Users/User/OneDrive - UNIVERSITAS INDONESIA/Perkuliahan/Semester 9/Tesis/Data Processing/Data/fs_data_clean.xlsx")
    fraudulent_emitens_path = Path("fraudulent_emitens.json")
    
    # Load Fraudulent Emitens
    fraudulent_emitens = load_fraudulent_emitens(fraudulent_emitens_path)
    
    # Load Data
    fs_data = load_data(data_path)
    
    # Preprocess Data
    df_processed, imputer, encoder = preprocess_data(fs_data, fraudulent_emitens)
    
    # Split Data
    X_train, X_test, y_train, y_test = split_data(df_processed, test_size=0.3, random_state=42)
    
    # Scale Features
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Handle Class Imbalance
    resampling_pipeline = build_resampling_pipeline()
    X_resampled, y_resampled = resampling_pipeline.fit_resample(X_train_scaled, y_train)
    logging.info(f"After Resampling: {Counter(y_resampled)}")
    
    # Load the existing XGBoost model with GA
    try:
        existing_xgb_ga_model = joblib.load("best_xgb_model_pygad.pkl")
        logging.info("Loaded existing XGBoost model with GA hyperparameters.")
    except FileNotFoundError:
        logging.error("Existing XGBoost model not found. Retraining...")
        # Perform GA hyperparameter tuning for XGBoost
        best_xgb_ga_params = perform_hyperparameter_tuning(X_resampled, y_resampled)
        existing_xgb_ga_model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, 
                                           eval_metric='logloss', random_state=42, **best_xgb_ga_params)
        existing_xgb_ga_model.fit(X_resampled, y_resampled)
    
    # Train XGBoost with Grid Search
    best_xgb_gs_params, xgb_gs_model = perform_grid_search_xgboost(X_resampled, y_resampled)
    
    # Train Random Forest with Grid Search
    best_rf_
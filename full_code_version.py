#!/usr/bin/env python3
"""
Survival Classification ML Pipeline
==================================

A comprehensive machine learning pipeline for binary classification using survival data.
This pipeline includes data preprocessing, feature engineering, model training with 
hyperparameter tuning, and comprehensive evaluation with visualizations.

Author: [Your Name]
Date: July 2025
Purpose: Graduation Project - Machine Learning Classification
"""

import os
import logging
import warnings
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline.log'),
        logging.StreamHandler()
    ]
)

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configuration
CONFIG = {
    'input_file': 'zincphosph1.xlsx',
    'target_column': 'Survival (1=Survived, 0=Died) ',
    'output_dir': 'outputs',
    'test_size': 0.2,
    'cv_folds': 5,
    'n_jobs': -1,
    'random_state': RANDOM_STATE
}


class DataProcessor:
    """
    Handles data loading, preprocessing, and feature engineering.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataProcessor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from Excel file with multiple sheets.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Combined DataFrame from all sheets
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If no sheets are found or data is empty
        """
        logging.info(f"Loading data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            
            if not sheet_names:
                raise ValueError("No sheets found in the Excel file")
            
            logging.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
            
            # Load and concatenate all sheets
            dfs = []
            for sheet in sheet_names:
                df_sheet = pd.read_excel(xls, sheet_name=sheet)
                if not df_sheet.empty:
                    dfs.append(df_sheet)
                    logging.info(f"Loaded sheet '{sheet}' with shape {df_sheet.shape}")
            
            if not dfs:
                raise ValueError("All sheets are empty")
            
            # Concatenate horizontally (axis=1) as in original code
            df_full = pd.concat(dfs, axis=1)
            
            # Remove duplicate columns
            df_full = df_full.loc[:, ~df_full.columns.duplicated()]
            
            logging.info(f"Combined dataset shape: {df_full.shape}")
            return df_full
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features_target(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            Tuple of (features, target)
            
        Raises:
            KeyError: If target column is not found
        """
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataset")
        
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
        
        logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        logging.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def handle_missing_values(self, X: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            X: Input features DataFrame
            strategy: Strategy for handling missing values ('median', 'mean', 'mode', 'constant')
            
        Returns:
            DataFrame with handled missing values
        """
        logging.info(f"Handling missing values using strategy: {strategy}")
        
        missing_info = X.isnull().sum()
        if missing_info.sum() > 0:
            logging.info(f"Missing values found:\n{missing_info[missing_info > 0]}")
        
        X_filled = X.copy()
        
        for col in X_filled.columns:
            if X_filled[col].isnull().any():
                if X_filled[col].dtype == 'object':
                    # For categorical variables, use mode or constant
                    mode_val = X_filled[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'unknown'
                    X_filled[col] = X_filled[col].fillna(fill_val)
                else:
                    # For numerical variables, use specified strategy
                    if strategy == 'median':
                        fill_val = X_filled[col].median()
                    elif strategy == 'mean':
                        fill_val = X_filled[col].mean()
                    else:  # constant
                        fill_val = -999
                    X_filled[col] = X_filled[col].fillna(fill_val)
        
        return X_filled
    
    def encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        logging.info("Encoding categorical features")
        
        X_encoded = X.copy()
        categorical_columns = X_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            # Convert to string first to handle mixed types
            X_encoded[col] = X_encoded[col].astype(str)
            X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col])
        
        logging.info(f"Encoded {len(categorical_columns)} categorical columns")
        return X_encoded
    
    def remove_low_variance_features(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Remove features with low variance.
        
        Args:
            X: Input features DataFrame
            threshold: Variance threshold
            
        Returns:
            DataFrame with low variance features removed
        """
        logging.info(f"Removing low variance features (threshold: {threshold})")
        
        initial_features = X.shape[1]
        variances = X.var()
        low_variance_features = variances[variances < threshold].index
        
        X_filtered = X.drop(columns=low_variance_features)
        
        logging.info(f"Removed {len(low_variance_features)} low variance features")
        logging.info(f"Features: {initial_features} → {X_filtered.shape[1]}")
        
        return X_filtered
    
    def balance_dataset(self, X: pd.DataFrame, y: pd.Series, method: str = 'upsample') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance the dataset using upsampling or downsampling.
        
        Args:
            X: Input features DataFrame
            y: Target variable
            method: Balancing method ('upsample' or 'downsample')
            
        Returns:
            Tuple of (balanced_features, balanced_target)
        """
        logging.info(f"Balancing dataset using {method}")
        
        # Combine features and target
        df_combined = X.copy()
        df_combined['target'] = y
        
        # Separate by class
        df_majority = df_combined[df_combined['target'] == y.value_counts().index[0]]
        df_minority = df_combined[df_combined['target'] == y.value_counts().index[1]]
        
        logging.info(f"Majority class: {len(df_majority)}, Minority class: {len(df_minority)}")
        
        if method == 'upsample':
            # Upsample minority class
            df_minority_resampled = resample(
                df_minority,
                replace=True,
                n_samples=len(df_majority),
                random_state=self.config['random_state']
            )
            df_balanced = pd.concat([df_majority, df_minority_resampled])
        else:  # downsample
            # Downsample majority class
            df_majority_resampled = resample(
                df_majority,
                replace=False,
                n_samples=len(df_minority),
                random_state=self.config['random_state']
            )
            df_balanced = pd.concat([df_majority_resampled, df_minority])
        
        # Shuffle the balanced dataset
        df_balanced = df_balanced.sample(frac=1, random_state=self.config['random_state']).reset_index(drop=True)
        
        X_balanced = df_balanced.drop(columns=['target'])
        y_balanced = df_balanced['target']
        
        logging.info(f"Balanced dataset shape: {X_balanced.shape}")
        logging.info(f"Balanced target distribution:\n{y_balanced.value_counts()}")
        
        return X_balanced, y_balanced


class ModelTrainer:
    """
    Handles model training, hyperparameter tuning, and evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelTrainer.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.models = {}
        self.best_models = {}
        self.results = {}
        
    def define_models(self) -> Dict[str, Any]:
        """
        Define the models to be trained.
        
        Returns:
            Dictionary of model instances
        """
        models = {
            'RandomForest': RandomForestClassifier(
                random_state=self.config['random_state'],
                n_jobs=self.config['n_jobs']
            ),
            'LogisticRegression': LogisticRegression(
                random_state=self.config['random_state'],
                max_iter=1000
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=self.config['random_state'],
                eval_metric='logloss',
                verbosity=0
            )
        }
        return models
    
    def define_hyperparameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Define hyperparameter grids for each model.
        
        Returns:
            Dictionary of hyperparameter grids
        """
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        return param_grids
    
    def train_and_tune_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train and tune multiple models using cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        logging.info("Starting model training and hyperparameter tuning")
        
        models = self.define_models()
        param_grids = self.define_hyperparameters()
        
        cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=self.config['random_state'])
        
        for name, model in models.items():
            logging.info(f"Training {name}...")
            
            # Perform randomized search for faster tuning
            random_search = RandomizedSearchCV(
                model,
                param_grids[name],
                n_iter=50,  # Limit iterations for faster execution
                cv=cv,
                scoring='f1',
                n_jobs=self.config['n_jobs'],
                random_state=self.config['random_state'],
                verbose=0
            )
            
            random_search.fit(X_train, y_train)
            
            self.best_models[name] = random_search.best_estimator_
            
            # Store results
            self.results[name] = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'cv_results': random_search.cv_results_
            }
            
            logging.info(f"{name} - Best CV Score: {random_search.best_score_:.4f}")
            logging.info(f"{name} - Best Parameters: {random_search.best_params_}")
        
        return self.best_models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        logging.info("Evaluating models on test data")
        
        evaluation_results = {}
        
        for name, model in self.best_models.items():
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            evaluation_results[name] = metrics
            
            logging.info(f"{name} Test Results:")
            for metric, value in metrics.items():
                logging.info(f"  {metric.capitalize()}: {value:.4f}")
        
        return evaluation_results


class Visualizer:
    """
    Handles all visualization tasks including plots and charts.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the Visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_feature_importance(self, model, feature_names: List[str], model_name: str, top_n: int = 20):
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to display
        """
        if not hasattr(model, 'feature_importances_'):
            logging.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.output_dir / f'feature_importance_{model_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Feature importance plot saved: {save_path}")
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str):
        """
        Plot confusion matrix for a model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Died', 'Survived'],
                   yticklabels=['Died', 'Survived'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        save_path = self.output_dir / f'confusion_matrix_{model_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Confusion matrix plot saved: {save_path}")
    
    def plot_roc_curve(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series):
        """
        Plot ROC curves for all models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
        """
        plt.figure(figsize=(10, 8))
        
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_test)
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"ROC curves plot saved: {save_path}")
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]):
        """
        Plot comparison of model performance metrics.
        
        Args:
            results: Dictionary of evaluation results
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral'])
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = self.output_dir / 'model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Model comparison plot saved: {save_path}")


class ResultsManager:
    """
    Manages saving and loading of results, models, and data.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the ResultsManager.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_processed_data(self, X_balanced: pd.DataFrame, y_balanced: pd.Series):
        """
        Save the balanced dataset.
        
        Args:
            X_balanced: Balanced features
            y_balanced: Balanced target
        """
        df_balanced = X_balanced.copy()
        df_balanced['target'] = y_balanced
        
        save_path = self.output_dir / 'balanced_dataset.csv'
        df_balanced.to_csv(save_path, index=False)
        
        logging.info(f"Balanced dataset saved: {save_path}")
    
    def save_models(self, models: Dict[str, Any]):
        """
        Save trained models.
        
        Args:
            models: Dictionary of trained models
        """
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for name, model in models.items():
            save_path = models_dir / f'{name.lower()}_model.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            
            logging.info(f"Model saved: {save_path}")
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """
        Save results to CSV file.
        
        Args:
            results: Results dictionary
            filename: Name of the output file
        """
        save_path = self.output_dir / filename
        
        if isinstance(results, dict):
            df_results = pd.DataFrame(results).T
            df_results.to_csv(save_path)
        else:
            results.to_csv(save_path)
        
        logging.info(f"Results saved: {save_path}")
    
    def save_classification_reports(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series):
        """
        Save detailed classification reports for all models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
        """
        reports_dir = self.output_dir / 'classification_reports'
        reports_dir.mkdir(exist_ok=True)
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            report_df = pd.DataFrame(report).transpose()
            save_path = reports_dir / f'{name.lower()}_classification_report.csv'
            report_df.to_csv(save_path)
            
            logging.info(f"Classification report saved: {save_path}")


def main():
    """
    Main function to run the complete ML pipeline.
    """
    logging.info("Starting Survival Classification ML Pipeline")
    
    try:
        # Initialize components
        processor = DataProcessor(CONFIG)
        trainer = ModelTrainer(CONFIG)
        visualizer = Visualizer(CONFIG['output_dir'])
        results_manager = ResultsManager(CONFIG['output_dir'])
        
        # 1. Data Loading and Preprocessing
        logging.info("=== DATA LOADING AND PREPROCESSING ===")
        df = processor.load_data(CONFIG['input_file'])
        X, y = processor.prepare_features_target(df, CONFIG['target_column'])
        
        # Handle missing values
        X = processor.handle_missing_values(X, strategy='median')
        
        # Encode categorical features
        X = processor.encode_categorical_features(X)
        
        # Remove low variance features
        X = processor.remove_low_variance_features(X)
        
        # Balance the dataset
        X_balanced, y_balanced = processor.balance_dataset(X, y, method='upsample')
        
        # Save balanced dataset
        results_manager.save_processed_data(X_balanced, y_balanced)
        
        # 2. Train-Test Split
        logging.info("=== TRAIN-TEST SPLIT ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, 
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state'],
            stratify=y_balanced
        )
        
        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Test set shape: {X_test.shape}")
        
        # 3. Feature Scaling
        logging.info("=== FEATURE SCALING ===")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # 4. Model Training and Tuning
        logging.info("=== MODEL TRAINING AND HYPERPARAMETER TUNING ===")
        best_models = trainer.train_and_tune_models(X_train_scaled, y_train)
        
        # Save trained models
        results_manager.save_models(best_models)
        
        # 5. Model Evaluation
        logging.info("=== MODEL EVALUATION ===")
        evaluation_results = trainer.evaluate_models(X_test_scaled, y_test)
        
        # Save evaluation results
        results_manager.save_results(evaluation_results, 'model_evaluation_results.csv')
        
        # 6. Visualizations
        logging.info("=== GENERATING VISUALIZATIONS ===")
        
        # Plot model comparison
        visualizer.plot_model_comparison(evaluation_results)
        
        # Plot ROC curves
        visualizer.plot_roc_curve(best_models, X_test_scaled, y_test)
        
        # Plot confusion matrices and feature importance for each model
        for name, model in best_models.items():
            y_pred = model.predict(X_test_scaled)
            visualizer.plot_confusion_matrix(y_test, y_pred, name)
            
            # Feature importance (only for tree-based models)
            if hasattr(model, 'feature_importances_'):
                visualizer.plot_feature_importance(model, X_train.columns.tolist(), name)
        
        # 7. Save Classification Reports
        logging.info("=== SAVING CLASSIFICATION REPORTS ===")
        results_manager.save_classification_reports(best_models, X_test_scaled, y_test)
        
        # 8. Final Summary
        logging.info("=== PIPELINE SUMMARY ===")
        best_model_name = max(evaluation_results.items(), key=lambda x: x[1]['f1'])[0]
        best_f1_score = evaluation_results[best_model_name]['f1']
        
        logging.info(f"Best performing model: {best_model_name}")
        logging.info(f"Best F1 Score: {best_f1_score:.4f}")
        
        # Print final results table
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-"*60)
        
        for model_name, metrics in evaluation_results.items():
            print(f"{model_name:<15} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
        
        print("="*60)
        print(f"Best Model: {best_model_name} (F1-Score: {best_f1_score:.4f})")
        print(f"All results saved in: {CONFIG['output_dir']}")
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
"""
Model training utilities for fake news detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os


class ModelTrainer:
    """
    Model training class for fake news detection
    """
    
    def __init__(self, models_path: str = "models/"):
        """
        Initialize model trainer
        
        Args:
            models_path (str): Path to save models
        """
        self.models_path = models_path
        os.makedirs(models_path, exist_ok=True)
        
        # Define models to train
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': MultinomialNB(),
            'neural_network': MLPClassifier(random_state=42, max_iter=500)
        }
        
        # Hyperparameter grids for tuning
        self.param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'naive_bayes': {
                'alpha': [0.1, 1, 10]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'learning_rate': ['constant', 'adaptive']
            }
        }
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   tune_hyperparameters: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train a specific model
        
        Args:
            model_name (str): Name of model to train
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            tune_hyperparameters (bool): Whether to tune hyperparameters
            cv_folds (int): Number of CV folds
            
        Returns:
            Dict: Training results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if tune_hyperparameters and model_name in self.param_grids:
            print(f"Tuning hyperparameters for {model_name}...")
            grid_search = GridSearchCV(
                model, 
                self.param_grids[model_name], 
                cv=cv_folds, 
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
        else:
            print(f"Training {model_name} with default parameters...")
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = None
            best_score = None
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds)
        
        # Save model
        model_filename = f"{self.models_path}/{model_name}_model.pkl"
        joblib.dump(best_model, model_filename)
        
        results = {
            'model_name': model_name,
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_path': model_filename
        }
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        tune_hyperparameters: bool = True, cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Train all models
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            tune_hyperparameters (bool): Whether to tune hyperparameters
            cv_folds (int): Number of CV folds
            
        Returns:
            Dict: Results for all models
        """
        results = {}
        
        for model_name in self.models.keys():
            print(f"\nTraining {model_name}...")
            try:
                results[model_name] = self.train_model(
                    model_name, X_train, y_train, tune_hyperparameters, cv_folds
                )
                print(f"{model_name} completed successfully")
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def compare_models(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare model performance
        
        Args:
            results (Dict): Training results
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, result in results.items():
            if 'error' not in result:
                comparison_data.append({
                    'Model': model_name,
                    'CV_Mean': result['cv_mean'],
                    'CV_Std': result['cv_std'],
                    'Best_Score': result['best_score']
                })
        
        return pd.DataFrame(comparison_data).sort_values('CV_Mean', ascending=False)
    
    def load_model(self, model_name: str):
        """
        Load a trained model
        
        Args:
            model_name (str): Name of model to load
            
        Returns:
            Trained model
        """
        model_filename = f"{self.models_path}/{model_name}_model.pkl"
        if os.path.exists(model_filename):
            return joblib.load(model_filename)
        else:
            raise FileNotFoundError(f"Model {model_name} not found at {model_filename}")
    
    def get_feature_importance(self, model_name: str, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance for tree-based models
        
        Args:
            model_name (str): Name of model
            feature_names (List[str]): Names of features
            
        Returns:
            pd.DataFrame: Feature importance
        """
        model = self.load_model(model_name)
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            raise ValueError(f"Model {model_name} does not support feature importance")

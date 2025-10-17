"""
Model evaluation utilities for fake news detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import learning_curve
import os


class ModelEvaluator:
    """
    Model evaluation class for fake news detection
    """
    
    def __init__(self, results_path: str = "results/"):
        """
        Initialize model evaluator
        
        Args:
            results_path (str): Path to save results
        """
        self.results_path = results_path
        os.makedirs(results_path, exist_ok=True)
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            model_name (str): Name of model
            
        Returns:
            Dict: Evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate AUC if probabilities are available
        auc = None
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return results
    
    def evaluate_all_models(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models
        
        Args:
            models (Dict): Dictionary of trained models
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict: Evaluation results for all models
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            try:
                results[model_name] = self.evaluate_model(model, X_test, y_test, model_name)
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def compare_models(self, evaluation_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare model evaluation results
        
        Args:
            evaluation_results (Dict): Evaluation results
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            if 'error' not in results:
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1_Score': results['f1_score'],
                    'AUC': results['auc']
                })
        
        return pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, 
                            class_names: List[str] = None, save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            cm (np.ndarray): Confusion matrix
            model_name (str): Name of model
            class_names (List[str]): Names of classes
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(f"{self.results_path}/{save_path}")
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      model_name: str, save_path: str = None):
        """
        Plot ROC curve
        
        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of model
            save_path (str): Path to save plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        
        if save_path:
            plt.savefig(f"{self.results_path}/{save_path}")
        plt.show()
    
    def plot_learning_curve(self, model, X: np.ndarray, y: np.ndarray,
                          model_name: str, save_path: str = None):
        """
        Plot learning curve
        
        Args:
            model: Model to evaluate
            X (np.ndarray): Features
            y (np.ndarray): Labels
            model_name (str): Name of model
            save_path (str): Path to save plot
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(f"{self.results_path}/{save_path}")
        plt.show()
    
    def save_results(self, results: Dict[str, Any], filename: str = "evaluation_results.json"):
        """
        Save evaluation results
        
        Args:
            results (Dict): Evaluation results
            filename (str): Filename to save
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, result in results.items():
            if 'error' not in result:
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        serializable_result[key] = value.tolist()
                    else:
                        serializable_result[key] = value
                serializable_results[model_name] = serializable_result
            else:
                serializable_results[model_name] = result
        
        with open(f"{self.results_path}/{filename}", 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {self.results_path}/{filename}")

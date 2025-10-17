"""
Prediction utilities for fake news detection
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any
import joblib
import os


class FakeNewsPredictor:
    """
    Fake news prediction class
    """
    
    def __init__(self, model_path: str = "models/", model_name: str = "best_model"):
        """
        Initialize predictor
        
        Args:
            model_path (str): Path to saved models
            model_name (str): Name of model to use
        """
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        
    def load_model(self, model_name: str = None):
        """
        Load trained model and associated components
        
        Args:
            model_name (str): Name of model to load
        """
        if model_name:
            self.model_name = model_name
        
        model_file = f"{self.model_path}/{self.model_name}_model.pkl"
        vectorizer_file = f"{self.model_path}/{self.model_name}_vectorizer.pkl"
        encoder_file = f"{self.model_path}/{self.model_name}_encoder.pkl"
        
        if os.path.exists(model_file):
            self.model = joblib.load(model_file)
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        if os.path.exists(vectorizer_file):
            self.vectorizer = joblib.load(vectorizer_file)
        
        if os.path.exists(encoder_file):
            self.label_encoder = joblib.load(encoder_file)
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict fake news for a single text
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess text if vectorizer is available
        if self.vectorizer is not None:
            text_vectorized = self.vectorizer.transform([text])
        else:
            text_vectorized = np.array([text]).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(text_vectorized)[0]
        probability = None
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_vectorized)[0]
            probability = float(max(probabilities))
        
        # Decode label if encoder is available
        if self.label_encoder is not None:
            prediction = self.label_encoder.inverse_transform([prediction])[0]
        
        result = {
            'text': text,
            'prediction': prediction,
            'confidence': probability,
            'is_fake': prediction == 'fake' or prediction == 1
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict fake news for multiple texts
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Predict fake news for dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            pd.DataFrame: Dataframe with predictions
        """
        results = self.predict_batch(df[text_column].tolist())
        
        # Add predictions to dataframe
        df_result = df.copy()
        df_result['prediction'] = [r['prediction'] for r in results]
        df_result['confidence'] = [r['confidence'] for r in results]
        df_result['is_fake'] = [r['is_fake'] for r in results]
        
        return df_result
    
    def get_prediction_confidence(self, text: str) -> float:
        """
        Get prediction confidence for a text
        
        Args:
            text (str): Input text
            
        Returns:
            float: Confidence score
        """
        result = self.predict_single(text)
        return result['confidence'] if result['confidence'] is not None else 0.0
    
    def classify_news_article(self, title: str = "", content: str = "") -> Dict[str, Any]:
        """
        Classify a news article with title and content
        
        Args:
            title (str): Article title
            content (str): Article content
            
        Returns:
            Dict: Classification results
        """
        # Combine title and content
        full_text = f"{title} {content}".strip()
        
        result = self.predict_single(full_text)
        
        # Add additional information
        result['title'] = title
        result['content'] = content
        result['word_count'] = len(full_text.split())
        result['char_count'] = len(full_text)
        
        return result
    
    def save_predictions(self, predictions: List[Dict[str, Any]], filename: str = "predictions.csv"):
        """
        Save predictions to CSV
        
        Args:
            predictions (List[Dict]): List of predictions
            filename (str): Output filename
        """
        df = pd.DataFrame(predictions)
        df.to_csv(f"{self.model_path}/{filename}", index=False)
        print(f"Predictions saved to {self.model_path}/{filename}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict: Model information
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        info = {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "has_vectorizer": self.vectorizer is not None,
            "has_encoder": self.label_encoder is not None
        }
        
        if hasattr(self.model, 'feature_importances_'):
            info["has_feature_importance"] = True
        else:
            info["has_feature_importance"] = False
        
        return info

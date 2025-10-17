"""
Tests for model training and evaluation functionality
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.predictor import FakeNewsPredictor


class TestModelTrainer:
    """Test cases for ModelTrainer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.trainer = ModelTrainer(models_path="test_models/")
        
        # Create sample data
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=42
        )
        self.X_train = X[:80]
        self.y_train = y[:80]
        self.X_test = X[80:]
        self.y_test = y[80:]
    
    def test_train_model(self):
        """Test single model training"""
        result = self.trainer.train_model(
            'logistic_regression', self.X_train, self.y_train, tune_hyperparameters=False
        )
        
        # Should return results dictionary
        assert isinstance(result, dict)
        assert 'model_name' in result
        assert 'model' in result
        assert 'cv_scores' in result
        
        # Model should be trained
        assert result['model'] is not None
    
    def test_train_all_models(self):
        """Test training all models"""
        results = self.trainer.train_all_models(
            self.X_train, self.y_train, tune_hyperparameters=False
        )
        
        # Should return results for all models
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Each result should have required keys
        for model_name, result in results.items():
            if 'error' not in result:
                assert 'model' in result
                assert 'cv_scores' in result
    
    def test_compare_models(self):
        """Test model comparison"""
        # Train a few models first
        results = {}
        for model_name in ['logistic_regression', 'random_forest']:
            results[model_name] = self.trainer.train_model(
                model_name, self.X_train, self.y_train, tune_hyperparameters=False
            )
        
        comparison_df = self.trainer.compare_models(results)
        
        # Should return dataframe
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) > 0
        assert 'Model' in comparison_df.columns
        assert 'CV_Mean' in comparison_df.columns


class TestModelEvaluator:
    """Test cases for ModelEvaluator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.evaluator = ModelEvaluator(results_path="test_results/")
        
        # Create sample data
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=42
        )
        self.X_test = X[:50]
        self.y_test = y[:50]
        
        # Create a simple model
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X[50:], y[50:])
    
    def test_evaluate_model(self):
        """Test single model evaluation"""
        results = self.evaluator.evaluate_model(
            self.model, self.X_test, self.y_test, "test_model"
        )
        
        # Should return results dictionary
        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'confusion_matrix' in results
        
        # Metrics should be numeric
        assert isinstance(results['accuracy'], (int, float))
        assert 0 <= results['accuracy'] <= 1
    
    def test_compare_models(self):
        """Test model comparison"""
        # Create multiple models
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        models = {
            'logistic': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        
        # Train models
        for model in models.values():
            model.fit(self.X_test, self.y_test)
        
        # Evaluate models
        evaluation_results = self.evaluator.evaluate_all_models(
            models, self.X_test, self.y_test
        )
        
        # Compare models
        comparison_df = self.evaluator.compare_models(evaluation_results)
        
        # Should return dataframe
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) > 0
        assert 'Model' in comparison_df.columns
        assert 'Accuracy' in comparison_df.columns


class TestFakeNewsPredictor:
    """Test cases for FakeNewsPredictor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.predictor = FakeNewsPredictor(model_path="test_models/", model_name="test_model")
        
        # Create sample data
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=42
        )
        
        # Train a simple model
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X, y)
        
        # Mock the model loading
        self.predictor.model = self.model
    
    def test_predict_single(self):
        """Test single prediction"""
        # Create sample text vector (mock)
        sample_text = "This is a sample news article"
        
        # Mock vectorizer
        class MockVectorizer:
            def transform(self, texts):
                return np.random.random((len(texts), 20))
        
        self.predictor.vectorizer = MockVectorizer()
        
        result = self.predictor.predict_single(sample_text)
        
        # Should return dictionary
        assert isinstance(result, dict)
        assert 'text' in result
        assert 'prediction' in result
        assert 'confidence' in result
    
    def test_predict_batch(self):
        """Test batch prediction"""
        sample_texts = [
            "This is fake news",
            "This is real news",
            "Another news article"
        ]
        
        # Mock vectorizer
        class MockVectorizer:
            def transform(self, texts):
                return np.random.random((len(texts), 20))
        
        self.predictor.vectorizer = MockVectorizer()
        
        results = self.predictor.predict_batch(sample_texts)
        
        # Should return list of results
        assert isinstance(results, list)
        assert len(results) == len(sample_texts)
        
        # Each result should be dictionary
        for result in results:
            assert isinstance(result, dict)
            assert 'text' in result
            assert 'prediction' in result
    
    def test_get_model_info(self):
        """Test model info retrieval"""
        info = self.predictor.get_model_info()
        
        # Should return dictionary
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'model_type' in info

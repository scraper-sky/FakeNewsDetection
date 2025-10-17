#!/usr/bin/env python3
"""
Main script for fake news detection ML model
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.loader import DataLoader
from data.preprocess import TextPreprocessor, FeatureExtractor
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator
from models.predictor import FakeNewsPredictor
from utils.config import Config
from utils.helpers import setup_logging, print_data_summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fake News Detection ML Model')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], 
                       default='train', help='Mode to run')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--model', type=str, help='Model name to use')
    parser.add_argument('--text', type=str, help='Text to predict (for predict mode)')
    parser.add_argument('--config', type=str, default='config/config.json', 
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    if os.path.exists(args.config):
        config = Config.load_config(args.config)
    else:
        config = Config()
        config.create_directories()
    
    logger.info(f"Running in {args.mode} mode")
    
    if args.mode == 'train':
        train_model(args, config, logger)
    elif args.mode == 'evaluate':
        evaluate_model(args, config, logger)
    elif args.mode == 'predict':
        predict_text(args, config, logger)


def train_model(args, config, logger):
    """Train the model"""
    logger.info("Starting model training...")
    
    # Load data
    if args.data:
        loader = DataLoader(config.get_path('raw_data'))
        df = loader.load_csv(args.data)
    else:
        # Create sample data for demonstration
        logger.warning("No data file provided, creating sample data")
        df = create_sample_data()
    
    logger.info(f"Loaded data with shape: {df.shape}")
    print_data_summary(df, "Dataset Summary")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df, 'text')
    
    # Extract features
    logger.info("Extracting features...")
    feature_extractor = FeatureExtractor(max_features=config.max_features)
    
    # Combine text and title if available
    if 'title' in df_processed.columns:
        combined_text = df_processed['title'] + ' ' + df_processed['text']
    else:
        combined_text = df_processed['text']
    
    # Extract TF-IDF features
    X = feature_extractor.extract_tfidf_features(combined_text.tolist())
    y = df_processed['label'].values
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    
    # Train models
    logger.info("Training models...")
    trainer = ModelTrainer(config.get_path('models'))
    results = trainer.train_all_models(
        X_train, y_train, tune_hyperparameters=True, cv_folds=config.cv_folds
    )
    
    # Compare models
    comparison_df = trainer.compare_models(results)
    logger.info("Model comparison:")
    print(comparison_df)
    
    # Evaluate best model
    best_model_name = comparison_df.iloc[0]['Model']
    logger.info(f"Best model: {best_model_name}")
    
    # Load best model and evaluate
    best_model = trainer.load_model(best_model_name)
    evaluator = ModelEvaluator(config.get_path('results'))
    evaluation_results = evaluator.evaluate_model(best_model, X_test, y_test, best_model_name)
    
    logger.info(f"Best model accuracy: {evaluation_results['accuracy']:.3f}")
    logger.info("Training completed!")


def evaluate_model(args, config, logger):
    """Evaluate the model"""
    logger.info("Starting model evaluation...")
    
    if not args.model:
        logger.error("Model name required for evaluation")
        return
    
    # Load test data
    if args.data:
        loader = DataLoader(config.get_path('raw_data'))
        df = loader.load_csv(args.data)
    else:
        logger.error("Data file required for evaluation")
        return
    
    # Preprocess data
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df, 'text')
    
    # Extract features
    feature_extractor = FeatureExtractor(max_features=config.max_features)
    X = feature_extractor.extract_tfidf_features(df_processed['text'].tolist())
    y = df_processed['label'].values
    
    # Load model
    trainer = ModelTrainer(config.get_path('models'))
    model = trainer.load_model(args.model)
    
    # Evaluate
    evaluator = ModelEvaluator(config.get_path('results'))
    results = evaluator.evaluate_model(model, X, y, args.model)
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Accuracy: {results['accuracy']:.3f}")
    logger.info(f"Precision: {results['precision']:.3f}")
    logger.info(f"Recall: {results['recall']:.3f}")
    logger.info(f"F1-Score: {results['f1_score']:.3f}")


def predict_text(args, config, logger):
    """Predict fake news for given text"""
    logger.info("Starting prediction...")
    
    if not args.text:
        logger.error("Text required for prediction")
        return
    
    if not args.model:
        logger.error("Model name required for prediction")
        return
    
    # Initialize predictor
    predictor = FakeNewsPredictor(config.get_path('models'), args.model)
    
    try:
        # Load model
        predictor.load_model()
        
        # Make prediction
        result = predictor.predict_single(args.text)
        
        logger.info("Prediction Results:")
        logger.info(f"Text: {result['text']}")
        logger.info(f"Prediction: {result['prediction']}")
        logger.info(f"Confidence: {result['confidence']:.3f}")
        logger.info(f"Is Fake: {result['is_fake']}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")


def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'title': [f'Sample news title {i}' for i in range(n_samples)],
        'text': [f'This is sample news content number {i}. ' * 10 for i in range(n_samples)],
        'label': np.random.choice(['fake', 'real'], n_samples, p=[0.4, 0.6]),
        'subject': np.random.choice(['politics', 'sports', 'technology', 'health'], n_samples)
    }
    
    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    main()

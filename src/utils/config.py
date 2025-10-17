"""
Configuration management for fake news detection
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration class for fake news detection project
    """
    
    # Project paths
    project_root: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path: str = "data"
    models_path: str = "models"
    results_path: str = "results"
    notebooks_path: str = "notebooks"
    
    # Data paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    external_data_path: str = "data/external"
    
    # Model configuration
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Text preprocessing
    max_features: int = 10000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    
    # Model parameters
    models_to_train: list = None
    
    # Evaluation
    evaluation_metrics: list = None
    
    def __post_init__(self):
        """Initialize default values after object creation"""
        if self.models_to_train is None:
            self.models_to_train = [
                'logistic_regression',
                'random_forest',
                'gradient_boosting',
                'svm',
                'naive_bayes',
                'neural_network'
            ]
        
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                'accuracy',
                'precision',
                'recall',
                'f1_score',
                'auc'
            ]
    
    def get_path(self, path_type: str) -> str:
        """
        Get full path for a given path type
        
        Args:
            path_type (str): Type of path
            
        Returns:
            str: Full path
        """
        path_mapping = {
            'project_root': self.project_root,
            'data': os.path.join(self.project_root, self.data_path),
            'models': os.path.join(self.project_root, self.models_path),
            'results': os.path.join(self.project_root, self.results_path),
            'notebooks': os.path.join(self.project_root, self.notebooks_path),
            'raw_data': os.path.join(self.project_root, self.raw_data_path),
            'processed_data': os.path.join(self.project_root, self.processed_data_path),
            'external_data': os.path.join(self.project_root, self.external_data_path)
        }
        
        return path_mapping.get(path_type, self.project_root)
    
    def create_directories(self) -> None:
        """Create all necessary directories"""
        directories = [
            self.get_path('data'),
            self.get_path('models'),
            self.get_path('results'),
            self.get_path('notebooks'),
            self.get_path('raw_data'),
            self.get_path('processed_data'),
            self.get_path('external_data')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary
        
        Returns:
            Dict: Configuration as dictionary
        """
        return {
            'project_root': self.project_root,
            'data_path': self.data_path,
            'models_path': self.models_path,
            'results_path': self.results_path,
            'notebooks_path': self.notebooks_path,
            'raw_data_path': self.raw_data_path,
            'processed_data_path': self.processed_data_path,
            'external_data_path': self.external_data_path,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'cv_folds': self.cv_folds,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'models_to_train': self.models_to_train,
            'evaluation_metrics': self.evaluation_metrics
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create config from dictionary
        
        Args:
            config_dict (Dict): Configuration dictionary
            
        Returns:
            Config: Configuration object
        """
        return cls(**config_dict)
    
    def save_config(self, filepath: str) -> None:
        """
        Save configuration to file
        
        Args:
            filepath (str): Path to save config
        """
        import json
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'Config':
        """
        Load configuration from file
        
        Args:
            filepath (str): Path to config file
            
        Returns:
            Config: Configuration object
        """
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


# Default configuration instance
default_config = Config()

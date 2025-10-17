"""
Data loading utilities for fake news detection
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os


class DataLoader:
    """
    Data loading and splitting utilities
    """
    
    def __init__(self, data_path: str = "data/raw/"):
        """
        Initialize data loader
        
        Args:
            data_path (str): Path to data directory
        """
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
    
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file
        
        Args:
            filename (str): Name of CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        filepath = os.path.join(self.data_path, filename)
        return pd.read_csv(filepath, **kwargs)
    
    def load_json(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load JSON file
        
        Args:
            filename (str): Name of JSON file
            **kwargs: Additional arguments for pd.read_json
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        filepath = os.path.join(self.data_path, filename)
        return pd.read_json(filepath, **kwargs)
    
    def load_fake_news_dataset(self, filename: str = "fake_news_dataset.csv") -> pd.DataFrame:
        """
        Load fake news dataset with standard column names
        
        Args:
            filename (str): Name of dataset file
            
        Returns:
            pd.DataFrame: Loaded and standardized dataframe
        """
        df = self.load_csv(filename)
        
        # Standardize column names
        column_mapping = {
            'title': 'title',
            'text': 'text',
            'label': 'label',
            'subject': 'subject',
            'date': 'date'
        }
        
        # Map columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, 
                    text_column: str = 'text',
                    label_column: str = 'label',
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            test_size (float): Test set size
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Remove rows with missing values
        df_clean = df.dropna(subset=[text_column, label_column])
        
        # Extract features and labels
        X = df_clean[text_column].values
        y = df_clean[label_column].values
        
        # Encode labels if they are strings
        if isinstance(y[0], str):
            y = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_class_distribution(self, df: pd.DataFrame, label_column: str = 'label') -> Dict[Any, int]:
        """
        Get class distribution
        
        Args:
            df (pd.DataFrame): Input dataframe
            label_column (str): Name of label column
            
        Returns:
            Dict: Class distribution
        """
        return df[label_column].value_counts().to_dict()
    
    def save_processed_data(self, X_train: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_test: np.ndarray,
                          filename_prefix: str = "processed_data") -> None:
        """
        Save processed data
        
        Args:
            X_train, X_test, y_train, y_test: Data arrays
            filename_prefix (str): Prefix for saved files
        """
        processed_path = "data/processed/"
        os.makedirs(processed_path, exist_ok=True)
        
        np.save(f"{processed_path}/{filename_prefix}_X_train.npy", X_train)
        np.save(f"{processed_path}/{filename_prefix}_X_test.npy", X_test)
        np.save(f"{processed_path}/{filename_prefix}_y_train.npy", y_train)
        np.save(f"{processed_path}/{filename_prefix}_y_test.npy", y_test)
    
    def load_processed_data(self, filename_prefix: str = "processed_data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load processed data
        
        Args:
            filename_prefix (str): Prefix for saved files
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        processed_path = "data/processed/"
        
        X_train = np.load(f"{processed_path}/{filename_prefix}_X_train.npy")
        X_test = np.load(f"{processed_path}/{filename_prefix}_X_test.npy")
        y_train = np.load(f"{processed_path}/{filename_prefix}_y_train.npy")
        y_test = np.load(f"{processed_path}/{filename_prefix}_y_test.npy")
        
        return X_train, X_test, y_train, y_test

"""
Helper utility functions for fake news detection
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import logging


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level (str): Logging level
        log_file (str): Log file path
        
    Returns:
        logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file
    
    Args:
        data (Dict): Data to save
        filepath (str): File path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file
    
    Args:
        filepath (str): File path
        
    Returns:
        Dict: Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object to pickle file
    
    Args:
        obj (Any): Object to save
        filepath (str): File path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file
    
    Args:
        filepath (str): File path
        
    Returns:
        Any: Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_directory_structure(base_path: str, directories: List[str]) -> None:
    """
    Create directory structure
    
    Args:
        base_path (str): Base directory path
        directories (List[str]): List of directories to create
    """
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)


def get_file_info(filepath: str) -> Dict[str, Any]:
    """
    Get file information
    
    Args:
        filepath (str): File path
        
    Returns:
        Dict: File information
    """
    if not os.path.exists(filepath):
        return {"exists": False}
    
    stat = os.stat(filepath)
    return {
        "exists": True,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "extension": os.path.splitext(filepath)[1]
    }


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate dataframe has required columns
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (List[str]): Required columns
        
    Returns:
        bool: True if valid
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def print_data_summary(df: pd.DataFrame, title: str = "Data Summary") -> None:
    """
    Print data summary
    
    Args:
        df (pd.DataFrame): Dataframe to summarize
        title (str): Summary title
    """
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def calculate_text_statistics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate text statistics
    
    Args:
        texts (List[str]): List of texts
        
    Returns:
        Dict: Text statistics
    """
    if not texts:
        return {}
    
    word_counts = [len(text.split()) for text in texts if text]
    char_counts = [len(text) for text in texts if text]
    
    return {
        "avg_word_count": np.mean(word_counts),
        "std_word_count": np.std(word_counts),
        "avg_char_count": np.mean(char_counts),
        "std_char_count": np.std(char_counts),
        "min_word_count": np.min(word_counts),
        "max_word_count": np.max(word_counts),
        "total_texts": len(texts),
        "non_empty_texts": len([t for t in texts if t and t.strip()])
    }


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    else:
        return f"{seconds/3600:.2f} hours"


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are installed
    
    Returns:
        Dict: Dependency status
    """
    dependencies = [
        'pandas', 'numpy', 'scikit-learn', 'nltk', 'matplotlib', 
        'seaborn', 'plotly', 'wordcloud', 'textblob'
    ]
    
    status = {}
    for dep in dependencies:
        try:
            __import__(dep)
            status[dep] = True
        except ImportError:
            status[dep] = False
    
    return status


def get_system_info() -> Dict[str, Any]:
    """
    Get system information
    
    Returns:
        Dict: System information
    """
    import platform
    import sys
    
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "machine": platform.machine()
    }

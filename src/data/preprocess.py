"""
Text preprocessing utilities for fake news detection
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TextPreprocessor:
    """
    Text preprocessing class for fake news detection
    """
    
    def __init__(self, language='english'):
        """
        Initialize the preprocessor
        
        Args:
            language (str): Language for stopwords
        """
        self.language = language
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(language))
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_stem(self, text: str) -> List[str]:
        """
        Tokenize and stem text
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of stemmed tokens
        """
        if not text:
            return []
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        stemmed_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return stemmed_tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and stem
        tokens = self.tokenize_and_stem(cleaned_text)
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Preprocess text column in dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            pd.DataFrame: Dataframe with preprocessed text
        """
        df_processed = df.copy()
        df_processed[f'{text_column}_processed'] = df_processed[text_column].apply(
            self.preprocess_text
        )
        return df_processed


class FeatureExtractor:
    """
    Feature extraction for fake news detection
    """
    
    def __init__(self, max_features: int = 10000):
        """
        Initialize feature extractor
        
        Args:
            max_features (int): Maximum number of features
        """
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
    
    def extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract TF-IDF features
        
        Args:
            texts (List[str]): List of preprocessed texts
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        return self.tfidf_vectorizer.fit_transform(texts).toarray()
    
    def extract_count_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract count features
        
        Args:
            texts (List[str]): List of preprocessed texts
            
        Returns:
            np.ndarray: Count feature matrix
        """
        return self.count_vectorizer.fit_transform(texts).toarray()
    
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract linguistic features
        
        Args:
            texts (List[str]): List of texts
            
        Returns:
            np.ndarray: Linguistic feature matrix
        """
        features = []
        
        for text in texts:
            if pd.isna(text):
                text = ""
                
            # Basic text statistics
            word_count = len(text.split())
            char_count = len(text)
            avg_word_length = char_count / word_count if word_count > 0 else 0
            
            # Punctuation count
            punct_count = sum(1 for char in text if char in string.punctuation)
            punct_ratio = punct_count / char_count if char_count > 0 else 0
            
            # Uppercase ratio
            upper_ratio = sum(1 for char in text if char.isupper()) / char_count if char_count > 0 else 0
            
            # Digit ratio
            digit_ratio = sum(1 for char in text if char.isdigit()) / char_count if char_count > 0 else 0
            
            features.append([
                word_count,
                char_count,
                avg_word_length,
                punct_ratio,
                upper_ratio,
                digit_ratio
            ])
        
        return np.array(features)

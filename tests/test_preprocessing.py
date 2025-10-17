"""
Tests for text preprocessing functionality
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import TextPreprocessor, FeatureExtractor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = TextPreprocessor()
        self.sample_text = "This is a sample text with URLs https://example.com and email@test.com"
        self.sample_dataframe = pd.DataFrame({
            'text': [
                "This is fake news!",
                "This is real news.",
                "Another fake article here."
            ],
            'label': ['fake', 'real', 'fake']
        })
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        cleaned = self.preprocessor.clean_text(self.sample_text)
        
        # Should remove URLs and emails
        assert "https://example.com" not in cleaned
        assert "email@test.com" not in cleaned
        
        # Should be lowercase
        assert cleaned.islower()
        
        # Should not contain special characters
        assert not any(char in cleaned for char in "!@#$%^&*()")
    
    def test_tokenize_and_stem(self):
        """Test tokenization and stemming"""
        tokens = self.preprocessor.tokenize_and_stem("This is a test sentence")
        
        # Should return list of tokens
        assert isinstance(tokens, list)
        
        # Should not contain stopwords
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
    
    def test_preprocess_text(self):
        """Test complete preprocessing pipeline"""
        processed = self.preprocessor.preprocess_text(self.sample_text)
        
        # Should be string
        assert isinstance(processed, str)
        
        # Should be cleaned and processed
        assert len(processed) > 0
    
    def test_preprocess_dataframe(self):
        """Test dataframe preprocessing"""
        processed_df = self.preprocessor.preprocess_dataframe(
            self.sample_dataframe, 'text'
        )
        
        # Should have new column
        assert 'text_processed' in processed_df.columns
        
        # Should have same number of rows
        assert len(processed_df) == len(self.sample_dataframe)
        
        # Processed text should be different from original
        assert not processed_df['text'].equals(processed_df['text_processed'])


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.extractor = FeatureExtractor(max_features=1000)
        self.sample_texts = [
            "This is a sample text for testing",
            "Another sample text with different content",
            "Third sample text for feature extraction"
        ]
    
    def test_extract_tfidf_features(self):
        """Test TF-IDF feature extraction"""
        features = self.extractor.extract_tfidf_features(self.sample_texts)
        
        # Should return numpy array
        assert isinstance(features, np.ndarray)
        
        # Should have correct shape
        assert features.shape[0] == len(self.sample_texts)
        assert features.shape[1] <= self.extractor.max_features
    
    def test_extract_count_features(self):
        """Test count feature extraction"""
        features = self.extractor.extract_count_features(self.sample_texts)
        
        # Should return numpy array
        assert isinstance(features, np.ndarray)
        
        # Should have correct shape
        assert features.shape[0] == len(self.sample_texts)
        assert features.shape[1] <= self.extractor.max_features
    
    def test_extract_linguistic_features(self):
        """Test linguistic feature extraction"""
        features = self.extractor.extract_linguistic_features(self.sample_texts)
        
        # Should return numpy array
        assert isinstance(features, np.ndarray)
        
        # Should have correct shape (6 features)
        assert features.shape == (len(self.sample_texts), 6)
        
        # All features should be numeric
        assert np.all(np.isfinite(features))

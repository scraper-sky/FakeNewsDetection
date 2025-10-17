"""
Data visualization utilities for fake news detection
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from wordcloud import WordCloud
import os


class DataVisualizer:
    """
    Data visualization class for fake news detection
    """
    
    def __init__(self, style: str = 'whitegrid', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualizer
        
        Args:
            style (str): Seaborn style
            figsize (Tuple): Default figure size
        """
        self.style = style
        self.figsize = figsize
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
    
    def plot_class_distribution(self, df: pd.DataFrame, label_column: str = 'label',
                              title: str = 'Class Distribution', save_path: str = None):
        """
        Plot class distribution
        
        Args:
            df (pd.DataFrame): Dataframe with labels
            label_column (str): Name of label column
            title (str): Plot title
            save_path (str): Path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        # Count classes
        class_counts = df[label_column].value_counts()
        
        # Create pie chart
        plt.subplot(1, 2, 1)
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        plt.title(f'{title} - Pie Chart')
        
        # Create bar chart
        plt.subplot(1, 2, 2)
        class_counts.plot(kind='bar')
        plt.title(f'{title} - Bar Chart')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_text_length_distribution(self, df: pd.DataFrame, text_column: str = 'text',
                                    title: str = 'Text Length Distribution', save_path: str = None):
        """
        Plot text length distribution
        
        Args:
            df (pd.DataFrame): Dataframe with text
            text_column (str): Name of text column
            title (str): Plot title
            save_path (str): Path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        # Calculate text lengths
        text_lengths = df[text_column].str.len()
        word_counts = df[text_column].str.split().str.len()
        
        # Plot character length distribution
        plt.subplot(1, 2, 1)
        plt.hist(text_lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{title} - Character Length')
        plt.xlabel('Character Count')
        plt.ylabel('Frequency')
        
        # Plot word count distribution
        plt.subplot(1, 2, 2)
        plt.hist(word_counts, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{title} - Word Count')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_word_cloud(self, texts: List[str], title: str = 'Word Cloud',
                       max_words: int = 100, save_path: str = None):
        """
        Plot word cloud
        
        Args:
            texts (List[str]): List of texts
            title (str): Plot title
            max_words (int): Maximum number of words
            save_path (str): Path to save plot
        """
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            max_words=max_words,
            background_color='white',
            colormap='viridis'
        ).generate(combined_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                              title: str = 'Feature Importance', top_n: int = 20,
                              save_path: str = None):
        """
        Plot feature importance
        
        Args:
            feature_importance (pd.DataFrame): Feature importance dataframe
            title (str): Plot title
            top_n (int): Number of top features to show
            save_path (str): Path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'{title} - Top {top_n} Features')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results: pd.DataFrame, metric: str = 'accuracy',
                            title: str = 'Model Comparison', save_path: str = None):
        """
        Plot model comparison
        
        Args:
            results (pd.DataFrame): Model results dataframe
            metric (str): Metric to compare
            title (str): Plot title
            save_path (str): Path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        # Sort by metric
        sorted_results = results.sort_values(metric, ascending=True)
        
        # Create horizontal bar plot
        plt.barh(range(len(sorted_results)), sorted_results[metric])
        plt.yticks(range(len(sorted_results)), sorted_results['Model'])
        plt.xlabel(metric.title())
        plt.title(f'{title} - {metric.title()}')
        
        # Add value labels
        for i, v in enumerate(sorted_results[metric]):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                            title: str = 'Confusion Matrix', save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            cm (np.ndarray): Confusion matrix
            class_names (List[str]): Class names
            title (str): Plot title
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, roc_data: Dict[str, Dict[str, np.ndarray]],
                       title: str = 'ROC Curves Comparison', save_path: str = None):
        """
        Plot multiple ROC curves
        
        Args:
            roc_data (Dict): ROC data for multiple models
            title (str): Plot title
            save_path (str): Path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        for model_name, data in roc_data.items():
            fpr, tpr, auc = data['fpr'], data['tpr'], data['auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, learning_curve_data: Dict[str, Dict[str, np.ndarray]],
                            title: str = 'Learning Curves', save_path: str = None):
        """
        Plot learning curves for multiple models
        
        Args:
            learning_curve_data (Dict): Learning curve data
            title (str): Plot title
            save_path (str): Path to save plot
        """
        plt.figure(figsize=self.figsize)
        
        for model_name, data in learning_curve_data.items():
            train_sizes = data['train_sizes']
            train_scores = data['train_scores']
            val_scores = data['val_scores']
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            plt.plot(train_sizes, train_mean, 'o-', label=f'{model_name} - Training')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
            plt.plot(train_sizes, val_mean, 'o-', label=f'{model_name} - Validation')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dashboard(self, df: pd.DataFrame, text_column: str = 'text',
                        label_column: str = 'label', save_path: str = None):
        """
        Create a comprehensive data dashboard
        
        Args:
            df (pd.DataFrame): Dataframe to visualize
            text_column (str): Name of text column
            label_column (str): Name of label column
            save_path (str): Path to save dashboard
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Class distribution
        class_counts = df[label_column].value_counts()
        axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Class Distribution')
        
        # 2. Text length distribution
        text_lengths = df[text_column].str.len()
        axes[0, 1].hist(text_lengths, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Text Length Distribution')
        axes[0, 1].set_xlabel('Character Count')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Word count distribution
        word_counts = df[text_column].str.split().str.len()
        axes[0, 2].hist(word_counts, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Word Count Distribution')
        axes[0, 2].set_xlabel('Word Count')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Text length by class
        for label in df[label_column].unique():
            subset = df[df[label_column] == label]
            axes[1, 0].hist(subset[text_column].str.len(), alpha=0.7, label=label, bins=30)
        axes[1, 0].set_title('Text Length by Class')
        axes[1, 0].set_xlabel('Character Count')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 5. Word count by class
        for label in df[label_column].unique():
            subset = df[df[label_column] == label]
            axes[1, 1].hist(subset[text_column].str.split().str.len(), alpha=0.7, label=label, bins=30)
        axes[1, 1].set_title('Word Count by Class')
        axes[1, 1].set_xlabel('Word Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        # 6. Missing values
        missing_data = df.isnull().sum()
        axes[1, 2].bar(range(len(missing_data)), missing_data.values)
        axes[1, 2].set_title('Missing Values')
        axes[1, 2].set_xlabel('Columns')
        axes[1, 2].set_ylabel('Missing Count')
        axes[1, 2].set_xticks(range(len(missing_data)))
        axes[1, 2].set_xticklabels(missing_data.index, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

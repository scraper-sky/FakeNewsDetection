# Fake News Detection ML Model

A machine learning model to detect fake news using natural language processing techniques.

## Project Structure

```
├── data/                   # Data storage
│   ├── raw/               # Raw datasets
│   ├── processed/         # Processed datasets
│   └── external/          # External datasets
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── models/           # Model definitions
│   ├── utils/            # Utility functions
│   └── visualization/    # Visualization tools
├── tests/                 # Unit tests
├── config/               # Configuration files
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
└── README.md            # This file
```

## Features

- Text preprocessing and feature extraction
- Multiple ML algorithms (Logistic Regression, Random Forest, Neural Networks)
- Model evaluation and comparison
- Data visualization and analysis
- Model persistence and loading

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download datasets to `data/raw/`
4. Run preprocessing: `python src/data/preprocess.py`
5. Train models: `python src/models/train.py`
6. Evaluate models: `python src/models/evaluate.py`

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- nltk
- matplotlib
- seaborn
- jupyter

## License

MIT License


# Sentiment Analysis Project

This project focuses on building and deploying a machine learning model for sentiment analysis on movie reviews. The aim is to classify reviews into positive or negative sentiments. The repository contains all necessary code and trained models for this purpose. For a full report and understanding of each parts please read the report on this project

## Repository Structure
- `SentimentAnalysis.py`: Main script containing the model building, training, and evaluation processes.
- `my_sentiment_model.h5`: Saved trained model.
- `my_training_history.pkl`: Pickle file containing the history of model accuracy and loss during training.
- `load_train.py`: Script for loading the trained model and skeleton for future development.

## Dependencies
This project relies on several Python libraries including TensorFlow, Keras, NumPy, and Matplotlib. The required imports are as follows:
```python
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import pickle
```

## Setting Up the Environment
To run this project, follow these steps to set up a virtual environment:

1. **Install Python**: Ensure you have Python installed on your system.

2. **Create a Virtual Environment**:
    ```bash
    python -m venv sentiment-analysis-env
    ```

3. **Activate the Virtual Environment**:
    - On Windows:
        ```bash
        .\sentiment-analysis-env\Scripts\activate
        ```
    - On Unix or MacOS:
        ```bash
        source sentiment-analysis-env/bin/activate
        ```

4. **Install Required Libraries**:
    ```bash
    pip install numpy tensorflow keras matplotlib
    ```

## Running the Project
To run the sentiment analysis model:

1. **Activate the Virtual Environment** as shown in the setup steps.

2. **Run the Main Script**:
    ```bash
    python SentimentAnalysis.py
    ```

## Model Loading and Further Development
To load the existing model and work on further development:

1. Run the `load_train.py` script.
2. This script will load the trained model (`my_sentiment_model.h5`) and the training history (`my_training_history.pkl`).

## Contributing
Contributions to enhance the model's accuracy, improve the codebase, or expand the application's features are welcome. Please follow the standard fork-and-pull request workflow for contributions.



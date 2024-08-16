# Sentiment Analysis Project

This project implements a comprehensive pipeline for sentiment analysis of tweets using machine learning models. It includes data preprocessing, model training, and inference using a pre-trained model from the Hugging Face Transformers library.

## Overview

The project consists of the following main components:

1. **sentiment_analysis_baseline.py**:

This script provides a pipeline for sentiment analysis, including:
- Loading and managing GloVe embeddings. 
- Preprocessing tweet texts (handling slang, emojis, and stopwords). 
- Building and training a deep convolutional neural network (CNN) for sentiment classification. 
- Plotting training and testing accuracy over epochs. 
- Predicting sentiment for new tweets using the trained model.

2. **run.py**: 
- This script sets up a FastAPI application for sentiment analysis of tweets using the Mistral model. 
- It provides an endpoint to get the sentiment of a tweet using the specified model.

3. **mistral_inference.py**:
- This script performs sentiment analysis on tweets using a pre-trained model from the Hugging Face Transformers library.
- It reads a dataset of tweets, processes them, and classifies each tweet's sentiment as positive, neutral, or negative.
- The results are saved to a JSON lines file for further analysis.

4. **metrics_calculation.py**: 
- This script calculates and visualizes evaluation metrics for sentiment analysis.
- It computes accuracy, F1 score, and confusion matrix, and displays these metrics along with a heatmap of the confusion matrix.

## Installation

To run this project, ensure you have the following dependencies installed:

```
pip install fastapi uvicorn transformers torch nltk matplotlib seaborn jsonlines pandas scikit-learn
```

## Usage

1. **Training the Model:**
 - Run the `sentiment_analysis_baseline.py` script to train the sentiment analysis model on the Sentiment140 dataset.

```
python3 sentiment_analysis_baseline.py
```

2. **Performing Inference:**

 - Run the `mistral_inference.py` script to perform sentiment analysis on a dataset of tweets.

```
python3 mistral_inference.py
```
 - This script processes the first 1000 tweets from the Sentiment140 dataset and saves the results to output_data_mistral7b.jsonl.

3. **Calculating Metrics:**

 - Run the `metrics_calculation.py` script to evaluate the model's performance.

```
python3 metrics_calculation.py
```

 - Ensure that the output data file is available for the script to read and calculate metrics.

4. **Running the FastAPI Application**:

 - Run the `run.py` script to start the FastAPI application.

```
python3 run.py
```

Access the sentiment analysis endpoint at http://localhost:8000/get_sentiment/ using a GET request with the required parameters.
Example request:

```
curl -X GET " curl -X GET "http://localhost:8000/get_sentiment/"      -H "Content-Type: application/json"      -d '{"model_name": "mistral", "tweet": "How are you?"}'"
```

**Expected Output**

The expected output for the sentiment analysis can be in the following format:

For the FastAPI application:

```
{"tweet": "How are you?", "sentiment": "neutral"}
```

For the `mistral_inference.py` script, the output will be saved in `output_data_mistral7b.jsonl` in the same format:

```
{"tweet": "How are you?", "gt_sentiment": "neutral", "pred_sentiment": "neutral"}
```
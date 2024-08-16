"""
This script performs sentiment analysis on tweets using a pre-trained model from the Hugging Face Transformers library.
It reads a dataset of tweets, processes them, and classifies each tweet's sentiment as positive, neutral, or negative.
The results are saved to a JSON lines file for further analysis.
"""

# Import libraries
from typing import Optional

import jsonlines
import pandas as pd
from transformers import pipeline, Pipeline


def model_inference(model: Pipeline, tweet: str) -> Optional[str]:  # Defining a function for model inference
    """
    Perform sentiment analysis on a given tweet using the specified model.

    Args:
        model (Pipeline): The sentiment analysis model to use for inference.
        tweet (str): The tweet text to analyze.

    Returns:
        Optional[str]: The classified sentiment of the tweet, which can be "positive", "neutral", or "negative".
                        Returns None if the sentiment cannot be determined.
    """
    # Defining a system prompt for sentiment analysis instructions
    system_prompt = """YOU ARE AN EXPERT IN SENTIMENT ANALYSIS OF SOCIAL MEDIA CONTENT, SPECIALIZING IN TWITTER DATA. YOUR TASK IS TO ACCURATELY CATEGORIZE TWEETS INTO ONE OF THREE SENTIMENTS: POSITIVE, NEUTRAL, OR NEGATIVE. THE OUTPUT MUST BE A STRING WITH THE CLASSIFIED SENTIMENT.
    ###INSTRUCTIONS###

    - You MUST categorize each tweet into one of the following sentiments: "positive," "neutral," or "negative."
    - Follow the "Chain of thoughts" before answering.
    - After chain of thoughts, provide the output as the classified sentiment.

    ###Chain of Thoughts###

    Follow the instructions in the strict order:
    1. **Reading the Tweet:**
       1.1. Read and understand the content of the tweet.
       1.2. Identify any emotional indicators, keywords, sarcasm, or subtle hints of frustration or disappointment.

    2. **Determining Sentiment:**
       2.1. Assess the overall tone of the tweet.
       2.2. Consider context, word choice, any emoticons, punctuation, and sarcasm that may influence sentiment.
       2.3. Look for phrases or keywords that indicate frustration, disappointment, or dissatisfaction 
       (e.g., "bummer," "failed," "didn't manage").
       2.4. Classify the tweet as "positive," "neutral," or "negative."
       2.5. For ambiguous tweets, consider the implied sentiment based on context and typical usage, especially looking for subtle negative emotions.

    3. **Generating Output:**
       3.1. After chain of thoughts steps, create a string containing the classified sentiment in the format:  
        ```
        sentiment: "negative|neutral|positive"
        ```
       3.2. Ensure the string format is correct and readable.

    ###What Not To Do###

    OBEY and never do:
    - NEVER SKIP CHAIN OF THOUGHTS STEPS
    - NEVER CLASSIFY A TWEET WITHOUT FULLY UNDERSTANDING ITS CONTEXT AND CONTENT.
    - NEVER INCLUDE INACCURATE OR MISLEADING SENTIMENT CLASSIFICATIONS.
    - NEVER IGNORE EMOTICONS, SARCASM, OR PUNCTUATION THAT MAY AFFECT SENTIMENT.
    - NEVER PROVIDE OUTPUT IN ANY FORMAT OTHER THAN THE SPECIFIED STRING STRUCTURE."""

    # Preparing messages for the model with the system prompt and the tweet
    messages = [
        {"role": "system", "content": system_prompt},  # System message with instructions
        {"role": "user", "content": tweet},  # User message containing the tweet
    ]

    # Generating new messages using the model with specific parameters
    new_messages = model(messages,
                         temperature=0.2,  # Setting temperature for randomness in output
                         max_new_tokens=100,  # Maximum number of tokens to generate
                         do_sample=True)  # Enable sampling for generation

    # Extracting the generated content from the model's output
    content = new_messages[0]['generated_text'][-1]['content']

    classes = ["negative", "neutral", "positive"]  # Defining possible sentiment classes
    to_find = "sentiment: "  # String to find in the generated content
    idx = content.rfind(to_find)  # Finding the last occurrence of the sentiment string

    if idx != -1:  # If the sentiment string is found
        sentiment = content[idx + len(to_find):]  # Extracting sentiment from the content
        for c in classes:  # Iterating through defined classes
            if c in sentiment.lower():  # Checking if the class is in the sentiment
                return c  # Returning the classified sentiment
    else:
        return None  # Returning None if sentiment string is not found


if __name__ == "__main__":  # Entry point of the script
    """
    Main function to execute sentiment analysis on tweets from the sentiment140 dataset.

    Reads the dataset, processes the tweets, and writes the results to a JSON lines file.
    """
    # Reading the sentiment140 dataset from a CSV file
    sentiment140_data = pd.read_csv(
        "data/training.1600000.processed.noemoticon.csv",  # Path to the CSV file
        encoding_errors="ignore",  # Ignoring encoding errors
        header=None  # No header in the CSV file
    )

    # Renaming columns of the DataFrame for better readability
    sentiment140_data = sentiment140_data.rename(
        columns={0: 'target', 1: 'ids',
                 2: 'date', 3: 'flag',
                 4: 'user', 5: 'text'}
    )

    # Shuffling the data randomly
    shuffled_data = sentiment140_data.sample(frac=1).reset_index(drop=True)

    # Initializing the sentiment analysis model using the transformers pipeline
    model = pipeline(
        "text-generation",  # Specifying the task type
        model="mistralai/Mistral-7B-Instruct-v0.1",  # Specifying the model to use
    )

    # Iterating through the first 1000 rows of the shuffled data
    for index, row in shuffled_data[:1000].iterrows():
        tweet = row['text']  # Extracting the tweet text
        gt_sentiment = "negative" if row['target'] == 0 else "positive"  # Ground truth sentiment
        pred_sentiment = model_inference(model, tweet)  # Getting predicted sentiment from the model

        # Writing the tweet, ground truth sentiment, and predicted sentiment to a JSON lines file
        jsonlines.open("output_data_mistral7b.jsonl", "a").write({
            "tweet": tweet,  # Original tweet
            "gt_sentiment": gt_sentiment,  # Ground truth sentiment
            "pred_sentiment": pred_sentiment,  # Predicted sentiment
        })

        # Printing the tweet and its sentiments for verification
        print(f'{index}: Tweet: "{tweet}""\nPredicted sentiment: {pred_sentiment}, GT sentiment: {gt_sentiment}\n')
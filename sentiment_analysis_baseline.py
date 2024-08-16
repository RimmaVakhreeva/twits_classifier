"""
This script provides a comprehensive pipeline for sentiment analysis using tweets. It includes:
1. Loading and managing GloVe embeddings.
2. Preprocessing tweet texts (e.g., handling slang, emojis, and stopwords).
3. Building and training a deep convolutional neural network (CNN) for sentiment classification.
4. Plotting training and testing accuracy over epochs.
5. Predicting sentiment for new tweets using the trained model.
"""

# Import libraries
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix, f1_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')

EMD_DIM = 200  # Set embedding dimension size


class GloveEmbeddingsIndex:
    """
        A class to manage and retrieve GloVe word embeddings from a pre-trained GloVe model file.

        Attributes:
            GLOVE_FILE_PATH (Path): The path to the GloVe embeddings file.
            cached_embeddings (dict): Cache for storing loaded embeddings to avoid re-reading the file.
            _index (dict): An index mapping words to their line numbers in the GloVe file for quick access.
            _bos_emb (np.ndarray): Embedding for the beginning of a sentence token.
            _eos_emb (np.ndarray): Embedding for the end of a sentence token.
            unknown_emb (np.ndarray): Embedding for unknown words.
        """

    # Path to the GloVe embeddings file, formatted to include embedding dimension in filename
    GLOVE_FILE_PATH = Path("./data") / f"./glove.twitter.27B.{EMD_DIM}d.txt"

    def __init__(self):
        self.cached_embeddings = {}  # Cache for storing loaded embeddings to avoid re-reading the file
        self._index = self._build_glove_file_index()  # Build an index for fast lookup of embeddings
        # Generate embeddings for special tokens: beginning of sentence, end of sentence, and unknown words
        self._bos_emb = np.random.uniform(-0.25, 0.25, EMD_DIM)
        self._eos_emb = np.random.uniform(-0.25, 0.25, EMD_DIM)
        self.unknown_emb = np.random.uniform(-0.25, 0.25, EMD_DIM)

    def get_embedding_for_word(self, word: str) -> np.ndarray:
        """
        Retrieve the embedding for a given word.

        Args:
            word (str): The word for which to retrieve the embedding.

        Returns:
            np.ndarray: The embedding vector for the word.
        """
        word = word.lower()  # Convert word to lowercase for normalization
        # Return special embeddings for beginning and end of sentence tokens
        if word == "<bos>":  # Check if the word is the beginning of sentence token
            return self._bos_emb  # Return the BOS embedding
        elif word == "<eos>":  # Check if the word is the end of sentence token
            return self._eos_emb  # Return the EOS embedding
        # Return cached embedding if available
        elif word in self.cached_embeddings:  # Check if the word is already cached
            return self.cached_embeddings[word]  # Return the cached embedding

        # Look up word position in index and read its embedding from file
        position = self._index.get(word.lower(), None)
        if position is not None:
            with open(self.GLOVE_FILE_PATH, 'r', encoding='utf-8') as file:
                # Skip lines to the word's position
                _ = [file.readline() for _ in range(position)]
                line_split = file.readline().split()  # Split the line into word and embedding
                found_word, embedding = line_split[0], line_split[1:]
                # Ensure the found word matches the requested word
                assert found_word == word, f"Invalid logic: found_word={found_word} != request_word={word}"  # Ensure match
                # Convert string embeddings to float and cache
                embedding = np.array([float(val) for val in embedding], dtype=np.float32)  # Convert to float array
                self.cached_embeddings[word] = embedding  # Cache the embedding
                return embedding
        else:
            # Return and cache unknown word embedding if word not found
            self.cached_embeddings[word] = self.unknown_emb  # Cache unknown word embedding
            return self.unknown_emb  # Return unknown word embedding

    def _build_glove_file_index(self) -> Dict[str, int]:
        """
        Build an index of words in the GloVe file for quick lookup.

        Returns:
            dict: A dictionary mapping words to their line numbers in the GloVe file.
        """
        # Create an index mapping words to their line numbers in the GloVe file for quick access
        logging.info("Loading glove embedding index")  # Log the start of index loading
        index = {}  # Initialize an empty index
        with open(self.GLOVE_FILE_PATH, 'r', encoding='utf-8') as file:
            for position, line in enumerate(file):  # Enumerate through the file lines
                line_split = line.split()  # Split the line into word and embedding
                word = line_split[0]  # Extract the word
                # Preload embeddings into cache
                embedding = np.array([float(val) for val in line_split[1:]],
                                     dtype=np.float32)  # Convert embedding to array
                index[word] = position  # Add the word's position to the index
                self.cached_embeddings[word] = embedding  # Preload the embedding into the cache
        return index  # Return the completed index

    def vocab(self) -> List[str]:
        """
        Return a list of all words for which embeddings are available.

        Returns:
            list: A list of words in the GloVe vocabulary.
        """
        return list(self._index.keys())  # Return all words in the index as a list


class TweetItem:
    """
    A class to represent and preprocess a single tweet.

    Attributes:
        SLANG_ACRONYMS (dict): Dictionary mapping slang acronyms to their expanded forms.
        EMOJI_EMOTIONS (dict): Dictionary mapping emoji to their described emotions.
        NEGATIVE_REF_REPLACEMENTS (dict): Dictionary mapping contractions and negative references to their expanded forms.
        text (str): Original tweet text.
        _stats (dict): Statistics for analyzing the text.
        normalized_text (str): Normalized text after preprocessing.
        normalized_tokens (list): Tokenized version of the normalized text.
    """

    # Dictionary mapping slang acronyms to their expanded forms
    SLANG_ACRONYMS = {
        'idts': 'I do not think so', 'icymi': 'In Case You Missed It', 'lol': 'laugh out loud', 'brb': 'be right back',
        'btw': 'by the way', 'imo': 'in my opinion', 'imho': 'in my humble opinion', 'smh': 'shaking my head',
        'tbh': 'to be honest', 'rofl': 'rolling on the floor laughing', 'bff': 'best friends forever',
        'nvm': 'never mind', 'idk': 'I do not know', 'tbt': 'throwback Thursday', 'fomo': 'fear of missing out',
        'ftw': 'for the win', 'gg': 'good game', 'irl': 'in real life', 'jk': 'just kidding',
        'lmao': 'laughing my ass off', 'omg': 'oh my god', 'ppl': 'people', 'thx': 'thanks',
        'ttyl': 'talk to you later', 'yolo': 'you only live once'
    }
    # Dictionary mapping emoji to their described emotions
    EMOJI_EMOTIONS = {
        ':)': 'happy', ':(': 'sad', '😊': 'smiling face with smiling eyes', '😂': 'face with tears of joy',
        '😢': 'crying face',
        '😠': 'angry face', '❤️': 'red heart', '👍': 'thumbs up', '👎': 'thumbs down', '😱': 'face screaming in fear',
        '🎉': 'party popper', '💔': 'broken heart', '😍': 'smiling face with heart-eyes', '😒': 'unamused face',
        '😉': 'winking face',
        '😜': 'winking face with tongue', '🙏': 'folded hands', '😐': 'neutral face', '😴': 'sleeping face',
        '💤': 'sleeping symbol',
        '😡': 'pouting face', '🤔': 'thinking face', '🤣': 'rolling on the floor laughing', '😇': 'smiling face with halo',
        '🤗': 'hugging face', '😔': 'pensive face', '😏': 'smirking face', '🥳': 'partying face', '🤯': 'exploding head',
        '🥺': 'pleading face', '🤩': 'star-struck', '😓': 'downcast face with sweat', '😖': 'confounded face',
        '🥰': 'smiling face with hearts', '😘': 'face blowing a kiss', '😷': 'face with medical mask'
    }
    # Mapping of contractions and negative references to their expanded forms
    NEGATIVE_REF_REPLACEMENTS = {
        "won't": "will not", "can't": "cannot", "n't": " not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not", "haven't": "have not",
        "hasn't": "has not", "hadn't": "had not", "doesn't": "does not", "don't": "do not",
        "didn't": "did not", "won’t": "will not", "can’t": "cannot", "shan't": "shall not",
        "shan’t": "shall not", "shouldn't": "should not", "shouldn’t": "should not", "couldn't": "could not",
        "couldn’t": "could not", "mightn't": "might not", "mightn’t": "might not", "mustn't": "must not",
        "mustn’t": "must not"
    }

    def __init__(self, text: str):
        assert type(text) is str, f"Invalid data: {text}"
        self.text = text  # Original tweet text
        # Statistics for analyzing the text
        self._stats = {
            "emoji_num": 0,  # Number of emojis
            "hashtag_num": self.text.count("#"),  # Number of hashtags
            "negation_num": 0,  # Number of negative references
            "capitalized_num": sum(int(l.isupper()) for l in self.text),  # Number of capitalized letters
        }
        # Normalized text after preprocessing
        self.normalized_text = self._normalize(self.text)
        # Tokenized version of the normalized text
        self.normalized_tokens = self._tokenize(self.normalized_text)

    @staticmethod
    # Dummy method for POS tagging (not implemented in the snippet)
    def _tag_pos(text: str) -> Dict[str, int]:
        """
        Dummy method for POS tagging (not implemented in the snippet).
        Args:
            text (str): Input text.

        Returns:
            dict: Counts of POS tags (dummy implementation).
        """
        tagged = nltk.pos_tag(nltk.word_tokenize(text))
        return Counter([pos for _, pos in tagged]).values()

    @staticmethod
    # Calculate Pointwise Mutual Information (PMI) for a word with respect to sentiment
    def pmi(word: str, sentiment: str, tokens: List[str]) -> float:
        """
        Calculate Pointwise Mutual Information (PMI) for a word with respect to sentiment.

        Args:
            word (str): The target word.
            sentiment (str): The sentiment class (e.g., "positive" or "negative").
            tokens (list): List of tokens in the text.

        Returns:
            float: The PMI score.
        """
        word_count = tokens.count(word)  # Count occurrences of the word in the token list
        sentiment_count = 1 if sentiment in tokens else 0  # Check if the sentiment is present in the tokens
        word_sentiment_count = 1 if word in tokens and sentiment in tokens else 0  # Count joint occurrences of the word and sentiment
        total_words = len(tokens)  # Get the total number of tokens in the list

        # Calculate the probabilities for the word, sentiment, and their joint occurrence
        p_word = word_count / total_words  # Probability of the word
        p_sentiment = sentiment_count / total_words  # Probability of the sentiment
        p_word_sentiment = word_sentiment_count / total_words  # Probability of the word given the sentiment

        # Calculate the PMI score using the formula
        pmi_score = math.log2((p_word_sentiment + 1e-8) / ((p_word + 1e-8) * (p_sentiment + 1e-8)))
        return pmi_score  # Return the calculated PMI score

    @staticmethod
    # Compute sentiment polarity scores for each token
    def sentiment_polarity_scores(tokens: List[str]) -> List[float]:
        """
        Compute sentiment polarity scores for each token.

        Args:
            tokens (list): List of tokens in the text.

        Returns:
            list: List of sentiment polarity scores.
        """
        scores = []  # Initialize an empty list for scores
        for token in tokens:  # Iterate over each token
            synsets = list(swn.senti_synsets(token))  # Get sentiment synsets for the token
            if synsets:  # If there are synsets
                pos_scores = [synset.pos_score() for synset in synsets]  # Get positive scores
                neg_scores = [synset.neg_score() for synset in synsets]  # Get negative scores
                avg_pos_score = sum(pos_scores) / len(pos_scores)  # Calculate average positive score
                avg_neg_score = sum(neg_scores) / len(neg_scores)  # Calculate average negative score
                score = avg_pos_score - avg_neg_score  # Calculate sentiment score
            else:
                pos_pmi = TweetItem.pmi(token, "positive", tokens)  # Calculate PMI for positive sentiment
                neg_pmi = TweetItem.pmi(token, "negative", tokens)  # Calculate PMI for negative sentiment
                score = pos_pmi - neg_pmi  # Calculate sentiment score using PMI
            scores.append(score)  # Append the score to the list
        return scores  # Return the list of scores

    def _remove_urls(self, text: str) -> str:
        """
       Remove URLs from the text.

       Args:
           text (str): Input text.

       Returns:
           str: Text with URLs removed.
       """
        return re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'an url', text
        )  # Replace URLs with a placeholder

    def _remove_numbers(self, text: str) -> str:
        """
        Remove numbers from the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with numbers removed.
        """
        return re.sub(r'\d+', '', text)

    def _replace_negative_references(self, text: str) -> str:
        """
        Replace negative references in the text with their expanded forms.

        Args:
            text (str): Input text.

        Returns:
            str: Text with negative references replaced.
        """
        for key, value in self.NEGATIVE_REF_REPLACEMENTS.items():  # Iterate over the replacements
            self._stats["negation_num"] += text.count(key)  # Update negation count
            text = text.replace(key, value)  # Replace the contraction with its full form
        return text  # Return the modified text

    def _expand_acronyms_slang(self, text: str) -> str:
        """
        Expand acronyms and slang in the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with acronyms and slang expanded.
        """
        words = text.split()  # Split text into words
        expanded_words = [
            self.SLANG_ACRONYMS[word.lower()] if word.lower() in self.SLANG_ACRONYMS else word
            for word in words  # Expand each word if it is a slang acronym
        ]
        return ' '.join(expanded_words)  # Join the words back into a string

    def _remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with stopwords removed.
        """
        stop_words = set(stopwords.words('english'))  # Get the list of stopwords
        words = text.split()  # Split text into words
        filtered_words = [word for word in words if word.lower() not in stop_words]  # Filter out stopwords
        return ' '.join(filtered_words)  # Join the words back into a string

    def _replace_emoticons_emoji(self, text: str) -> str:
        """
        Replace emoticons and emoji in the text with words.

        Args:
            text (str): Input text.

        Returns:
            str: Text with emoticons and emoji replaced with words.
        """
        for key, value in self.EMOJI_EMOTIONS.items():  # Iterate over emoji replacements
            self._stats["emoji_num"] += text.count(key)  # Update emoji count
            text = text.replace(key, value)  # Replace emoji with corresponding word
        return text  # Return the modified text

    def _normalize(self, text: str) -> str:
        """
        Normalize the tweet text by applying preprocessing functions.

        Args:
            text (str): Input text.

        Returns:
            str: Normalized text.
        """
        text = self._remove_urls(text)  # Remove URLs
        text = self._remove_numbers(text)  # Remove numbers
        text = self._replace_negative_references(text)  # Replace negative references
        text = self._expand_acronyms_slang(text)  # Expand acronyms and slang
        text = self._remove_stopwords(text)  # Remove stopwords
        text = self._replace_emoticons_emoji(text)  # Replace emoticons and emojis
        return text  # Return the normalized text

    # Tokenize the text into words
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text into words.

        Args:
            text (str): Input text.

        Returns:
            list: List of tokens.
        """
        return ["<bos>"] + word_tokenize(text, language="english", preserve_line=False) + ["<eos>"]  # Add BOS and EOS

    def n_grams_vocab_counter(self, ngram_range: Tuple = (1, 2)) -> Counter:
        """
        Generate n-grams from the tokenized text.

        Args:
            ngram_range (tuple): Range of n-grams to generate.

        Returns:
            Counter: A counter of n-grams.
        """
        all_vocab = None  # Initialize empty variable for vocabulary
        for n in ngram_range:  # Iterate over the n-gram range
            if all_vocab is None:
                all_vocab = Counter(nltk.ngrams(self.normalized_tokens, n))  # Generate n-grams
            else:
                all_vocab = all_vocab + Counter(nltk.ngrams(self.normalized_tokens, n))  # Add n-grams to the counter
        return all_vocab  # Return the counter

    def generate_feature_vector(
            self,
            embedding_index: GloveEmbeddingsIndex,
            max_length: int,
            vocab: List[str],
            ngram_range: Tuple = (1, 2),
    ) -> np.ndarray:
        """
        Generate a feature vector for the tweet using embeddings and other features.

        Args:
            embedding_index (GloveEmbeddingsIndex): Index for GloVe embeddings.
            max_length (int): Maximum length for the tweet.
            vocab (list): List of vocabulary words.
            ngram_range (tuple): Range of n-grams to consider.

        Returns:
            np.ndarray: Feature vector for the tweet.
        """
        word_embeddings = np.stack([embedding_index.get_embedding_for_word(t) for t in self.normalized_tokens], axis=0)
        pad_size = max_length - len(self.normalized_tokens)  # Calculate padding size
        if len(word_embeddings) < max_length:  # If embeddings are shorter than max length
            padded = np.array([embedding_index.unknown_emb] * pad_size, dtype=np.float32)  # Create padding
            word_embeddings = np.concatenate([word_embeddings, padded], axis=0)  # Add padding
            sentiment_scores_vector = np.array(
                self.sentiment_polarity_scores(self.normalized_tokens) + ([-1] * pad_size),
                dtype=np.float32).reshape((-1, 1))  # Calculate sentiment scores and pad
        else:
            word_embeddings = word_embeddings[:max_length]  # Truncate embeddings to max length
            sentiment_scores_vector = np.array(self.sentiment_polarity_scores(self.normalized_tokens)[:max_length],
                                               dtype=np.float32).reshape((-1, 1))  # Truncate sentiment scores

        local_vocab_counter = dict(self.n_grams_vocab_counter(ngram_range))  # Count n-grams
        unigram_bigram_vector = np.array([local_vocab_counter.get(v, 0) for v in vocab],
                                         dtype=np.float32)  # Convert to array

        twitter_specific_vector = np.array([
            self._stats["emoji_num"], self._stats["hashtag_num"], self._stats["negation_num"],
            self._stats["capitalized_num"],  # Create a vector with Twitter-specific features
        ])

        extra_data = np.concatenate([unigram_bigram_vector, twitter_specific_vector]).reshape(
            (1, -1))  # Combine features
        extra_data = np.repeat(extra_data, (max_length,), axis=0)  # Repeat to match max length
        return np.concatenate((word_embeddings, sentiment_scores_vector, extra_data), axis=1).astype(
            np.float32)  # Combine all


def load_sentiment_140(path: Path = Path("./data") / "training.1600000.processed.noemoticon.csv") -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Load the Sentiment140 dataset from a CSV file.

    Args:
        path (Path): Path to the dataset file.

    Returns:
        tuple: A tuple containing a list of tweets and their corresponding sentiment labels.
    """

    data = pd.read_csv(path, encoding_errors="ignore",
                       header=None)  # Load the dataset from a CSV file, ignoring encoding errors
    data = data.rename(columns={0: 'target', 1: 'ids', 2: 'date', 3: 'flag', 4: 'user', 5: 'text'})  # Rename columns
    return data["text"].tolist(), data["target"].tolist()  # Return tweets and their corresponding sentiment labels


# Dictionary mapping dataset names to their respective loading functions
ds_readers = {
    "Sentiment140": load_sentiment_140  # Function to load the Sentiment140 dataset
}


class Dataset:
    """
    A class to manage the dataset, including loading, preprocessing, and splitting into training and test sets.

    Attributes:
        ngram_range (tuple): Specifies the range of n-grams to be considered.
        tweet_max_tokens_num (int): Maximum number of tokens per tweet.
        datasets (tuple): Names of datasets to load.
        top_k_for_ngram (int): Number of top n-grams to retain.
        limit (int, optional): Limit on the number of tweets to load.
        seed (int): Random seed for shuffling.
        test_split_ratio (float): Ratio for splitting data into training and test sets.
    """

    def __init__(
            self,
            ngram_range: Tuple = (1, 2),  # Specifies the range of n-grams to be considered
            tweet_max_tokens_num: int = 128,  # Maximum number of tokens per tweet
            datasets: Tuple = ("Sentiment140",),  # Names of datasets to load
            top_k_for_ngram: int = 512,  # Number of top n-grams to retain
            limit: Optional[int] = None,  # Limit on the number of tweets to load
            seed: int = 42,  # Random seed for shuffling
            test_split_ratio: float = 0.2  # Ratio for splitting data into training and test sets
    ):
        self.ngram_range = ngram_range  # Set the n-gram range
        self.tweet_max_tokens_num = tweet_max_tokens_num  # Set the max tokens per tweet
        self.is_train = True  # Flag to toggle between training and evaluation modes
        self.x, self.y = [], []  # Lists to store tweet texts and their labels
        self.random = random.Random(seed)  # Random number generator
        self.emb_index = GloveEmbeddingsIndex()  # Instance of GloveEmbeddingsIndex for accessing embeddings

        # Load data from specified datasets
        for ds in datasets:  # Iterate over datasets
            x, y = ds_readers[ds]()  # Load dataset
            self.x += x  # Add tweets to the list
            self.y += y  # Add labels to the list

        # Shuffle the data
        indices = list(range(len(self.x)))  # Create a list of indices for shuffling
        self.random.shuffle(indices)  # Shuffle the indices
        self.x = [self.x[i] for i in indices]  # Shuffle the tweets
        self.y = [self.y[i] for i in indices]  # Shuffle the labels

        # Apply limit if specified
        if limit is not None:  # If a limit is set
            self.x = self.x[:limit]  # Limit the tweets
            self.y = self.y[:limit]  # Limit the labels

        self.x = [TweetItem(text) for text in
                  tqdm.tqdm(self.x, desc="creating dataset")]  # Normalize and tokenize tweets
        self.global_vocab: Counter = self._retrieve_global_vocab()  # Retrieve global vocabulary from the dataset
        self.most_common_vocab: List[Tuple] = [el for el, _ in self.global_vocab.most_common(
            top_k_for_ngram)]  # Identify the most common vocabularies

        # Split data into training and test sets
        indices = list(range(len(self.x)))  # Create a list of indices for splitting
        self.random.shuffle(indices)  # Shuffle the indices again
        split_idx = set(indices[:int(len(self.x) * test_split_ratio)])  # Determine the split index
        self.train_x = [row for i, row in enumerate(self.x) if i not in split_idx]  # Create training set
        self.train_y = [row for i, row in enumerate(self.y) if i not in split_idx]  # Create training labels
        self.test_x = [row for i, row in enumerate(self.x) if i in split_idx]  # Create test set
        self.test_y = [row for i, row in enumerate(self.y) if i in split_idx]  # Create test labels

    def encode(self, tweet: str) -> np.ndarray:
        """
        Encode a tweet into a feature vector.

        Args:
            tweet (str): The tweet text.

        Returns:
            np.ndarray: Feature vector for the tweet.
        """
        return TweetItem(tweet).generate_feature_vector(
            embedding_index=self.emb_index,
            max_length=self.tweet_max_tokens_num,
            vocab=self.most_common_vocab,
            ngram_range=self.ngram_range
        )  # Encode the tweet into a feature vector

    def eval(self):
        """
        Switch to evaluation mode.
        """
        self.is_train = False

    def train(self):
        """
        Switch to training mode.
        """
        self.is_train = True

    def _retrieve_global_vocab(self):
        """
        Calculate global vocabulary from the dataset, limiting to the first 10,000 tweets for performance.

        Returns:
            Counter: A counter of the global vocabulary.
        """
        c = Counter()  # Initialize an empty counter
        for row in tqdm.tqdm(self.x[:10000], desc="calculating global vocab"):  # Iterate over the first 10,000 tweets
            c += row.n_grams_vocab_counter(self.ngram_range)  # Count n-grams
        return c  # Return the global vocabulary

    @property
    def emb_dim(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            int: The embedding dimension.
        """
        return self[0][0].shape[1]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get an item by index, for training or testing.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the feature vector and label.
        """
        x, y = (self.train_x[idx], self.train_y[idx]) if self.is_train else (
            self.test_x[idx], self.test_y[idx])  # Get the item
        # Convert labels to categorical values
        if y == 0:
            y = 0  # Convert label 0 to 0
        elif y in (1, 2, 3):
            y = 1  # Convert labels 1, 2, 3 to 1
        else:
            y = 2  # Convert all other labels to 2
        # Generate feature vector for the tweet
        vec = x.generate_feature_vector(embedding_index=self.emb_index,
                                        max_length=self.tweet_max_tokens_num,
                                        vocab=self.most_common_vocab,
                                        ngram_range=self.ngram_range)  # Generate the feature vector
        return vec, np.array(y, dtype=np.int64)  # Return the feature vector and label

    def __len__(self) -> int:
        """
        Return the length of the dataset, depending on training or evaluation mode.

        Returns:
            int: The length of the dataset.
        """
        return len(self.train_x) if self.is_train else len(self.test_x)


class GloVeDCNN(nn.Module):
    """
    A class representing a deep convolutional neural network (CNN) using GloVe embeddings for sentiment analysis.

    Attributes:
        emb_dim (int): Embedding dimension.
        num_filters (int): Number of filters in the convolutional layers.
        filter_sizes (list): List of filter sizes for the convolutional layers.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
    """

    def __init__(self, emb_dim, num_filters, filter_sizes, num_classes, dropout_rate):
        super(GloVeDCNN, self).__init__()  # Initialize the superclass
        # Define a series of convolutional layers based on the specified filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, filter_size) for filter_size in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters,
                            num_classes)  # A fully connected layer that outputs the predictions for each class
        self.dropout = nn.Dropout(
            dropout_rate)  # Dropout layer to prevent overfitting by randomly dropping units during training

    def forward(self, x):
        """
        Define the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with class scores.
        """
        x = x.permute(0, 2, 1)  # Rearrange tensor dimensions for convolution operations
        x = [torch.relu(conv(x)) for conv in
             self.convs]  # Apply ReLU activation function to the output of each convolutional layer
        x = [torch.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2) for conv_out in
             x]  # Apply max pooling to each convolution output to reduce its dimensionality
        x = torch.cat(x, dim=1)  # Concatenate all the convolutional layer outputs
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.fc(x)  # Pass the final concatenated tensor through the fully connected layer to get class scores
        return x


def plot_acc(train_accuracy, test_accuracy):
    """
    Plot training and testing accuracy over epochs.

    Args:
        train_accuracy (list): List of training accuracies.
        test_accuracy (list): List of testing accuracies.
    """
    epochs = list(range(1, len(train_accuracy) + 1))  # Create a list of epoch numbers
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(epochs, train_accuracy, label='Train Accuracy', marker='o')  # Plot training accuracy
    plt.plot(epochs, test_accuracy, label='Test Accuracy', marker='x')  # Plot testing accuracy
    plt.title('Train vs Test Accuracy per Epoch')  # Set the title of the plot
    plt.xlabel('Epoch')  # Label the x-axis
    plt.ylabel('Accuracy')  # Label the y-axis
    plt.legend()  # Show the legend
    plt.grid(True)  # Enable grid lines
    plt.show()  # Display the plot


def predict(model, ds, tweet: str) -> int:
    """
    Predicts the sentiment of a given tweet using the trained model.

    Parameters:
        model (nn.Module): The trained neural network model for sentiment analysis.
        ds (Dataset): The dataset object that provides the encoding method for tweets.
        tweet (str): The tweet text for which sentiment needs to be predicted.

    Returns:
        int: The index of the predicted sentiment class (0 for negative, 1 for neutral, 2 for positive).
    """
    # Mapping of class indices to sentiment categories
    categories = {0: 'negative', 1: 'neutral', 2: 'positive'}

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Encode the tweet into a tensor and add a batch dimension
        item = torch.tensor(ds.encode(tweet), dtype=torch.float32, device='cpu')[None, ...]

        # Get the model's output probabilities for the input tweet
        probs = model(item).sigmoid()

        # Determine the index of the class with the highest probability
        class_idx = probs.argmax().item()

        # Print the tweet, predicted sentiment, and probabilities
        print(f'Tweet: `{tweet}`, predicted sentiment: {categories[class_idx]}, probs: {probs.cpu().numpy().tolist()}')

    # Return the index of the predicted sentiment class
    return class_idx


def train(
        num_filters: int = 256,
        filter_sizes: Tuple = (3, 4, 5, 12),
        dropout_rate: float = 0.05,
        num_classes: int = 3,
        test_split_ratio: float = 0.1,
        batch_size: int = 2048,
        num_epochs: int = 10,
        learning_rate: float = 0.005,
        lr_step_size: int = 1,
        lr_gamma: float = 0.6,
        datasets: Tuple = ("STSGd", "Sentiment140"),
        num_workers: int = 16,
        cuda: bool = True,
        ds_limit_size: Optional[int] = None
):
    """
    Train a GloVe-based Deep Convolutional Neural Network (DCNN) on the specified datasets.

    Parameters:
        num_filters (int): Number of filters in the convolutional layers.
        filter_sizes (Tuple): Sizes of the convolutional filters.
        dropout_rate (float): Dropout rate for regularization.
        num_classes (int): Number of output classes for classification.
        test_split_ratio (float): Ratio of the dataset to be used for testing.
        batch_size (int): Number of samples per batch.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate for the optimizer.
        lr_step_size (int): Step size for the learning rate scheduler.
        lr_gamma (float): Multiplicative factor for learning rate decay.
        datasets (Tuple): Names of the datasets to be used for training.
        num_workers (int): Number of subprocesses to use for data loading.
        cuda (bool): Flag to enable CUDA for GPU training.
        ds_limit_size (Optional[int]): Limit on the size of the dataset.

    Returns:
        Tuple: A tuple containing training and testing accuracies, and the trained model and dataset.
    """

    # Create the model, dataset, and data loader
    ds = Dataset(datasets=datasets, seed=69, test_split_ratio=test_split_ratio,
                 limit=ds_limit_size)  # Initialize dataset
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            drop_last=True)  # Create data loader
    model = GloVeDCNN(ds.emb_dim, num_filters, filter_sizes, num_classes, dropout_rate)  # Initialize the model
    if cuda:
        model = model.cuda()  # Move model to GPU if CUDA is enabled

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Set loss function to CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Initialize Adam optimizer
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)  # Learning rate scheduler

    # Training loop
    train_accs, test_accs = [], []  # Lists to store training and testing accuracies
    all_targets = []  # List to store all target labels
    all_predictions = []  # List to store all predictions
    for epoch in range(num_epochs):  # Loop over epochs
        ds.train()  # Set dataset to training mode
        model.train()  # Set model to training mode for dropout, batchnorm, etc.
        total_predictions, correct_predictions = 0, 0  # Initialize counters for accuracy calculation
        for inputs, targets in tqdm.tqdm(dataloader, desc=f"Epoch {epoch}: "):  # Loop over batches
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()  # Move data to GPU if CUDA is enabled
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(inputs)  # Forward pass: compute predicted outputs by passing inputs to the model
            total_predictions += targets.size(0)  # Count total predictions for accuracy calculation
            # Count how many predictions match the targets
            correct_predictions += (outputs.argmax(dim=1) == targets).sum()  # Count correct predictions
            loss = criterion(outputs, targets)  # Calculate the loss
            loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # Perform a single optimization step (parameter update)
        train_accuracy = correct_predictions / total_predictions  # Calculate training accuracy
        train_accs.append(train_accuracy.cpu().numpy())  # Append training accuracy to the list

        ds.eval()  # Set dataset to evaluation mode
        model.eval()  # Set model to evaluation mode
        total_predictions, correct_predictions = 0, 0  # Reset counters for accuracy calculation
        with torch.no_grad():  # Disable gradient calculation for evaluation, saving memory and computations
            test_data = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=16,
                                   drop_last=True)  # Create test data loader
            for inputs, targets in tqdm.tqdm(test_data, desc=f"Testing"):  # Loop over test batches
                if cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()  # Move data to GPU if CUDA is enabled
                outputs = model(inputs)  # Forward pass: compute predicted outputs by passing inputs to the model
                test_loss = criterion(outputs, targets)  # Calculate the loss
                total_predictions += targets.size(0)  # Count total predictions for accuracy calculation
                # Count how many predictions match the targets
                correct_predictions += (outputs.argmax(dim=1) == targets).sum()  # Count correct predictions

                all_targets.extend(targets.cpu().numpy())  # Store targets for confusion matrix
                all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())  # Store predictions for confusion matrix

        test_accuracy = correct_predictions / total_predictions  # Calculate test accuracy
        test_accs.append(test_accuracy.cpu().numpy())  # Append test accuracy to the list

        # Print epoch summary
        print(f"Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {scheduler.get_last_lr()[0]:.6f}, "
              f"Train Loss: {loss.item():.4f}, Test loss: {test_loss.item():.4f}, "
              f"Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
        scheduler.step()  # Update the learning rate

    # Compute confusion matrix and F1 score
    conf_matrix = confusion_matrix(all_targets, all_predictions)  # Calculate confusion matrix
    f1 = f1_score(all_targets, all_predictions, average='weighted')  # Calculate F1 score

    print(f"Confusion Matrix:\n{conf_matrix}")  # Print confusion matrix
    print(f"F1 Score: {f1:.4f}")  # Print F1 score

    return (train_accs, test_accs), (model, ds)  # Return training and testing accuracies, model, and dataset


if __name__ == "__main__":
    """
    Main entry point for training the model on the Sentiment140 dataset and making predictions.

    This script initializes the training process, trains the model, plots the accuracy, 
    and makes a prediction on a sample input.
    """

    print("Train Sentiment140 dataset")  # Print a message indicating the start of training on the Sentiment140 dataset

    # Call the train function with specified parameters and unpack the returned values
    (train_acc, test_acc), (model, ds) = train(
        dropout_rate=0.05,  # Set the dropout rate for the model
        batch_size=256,  # Set the batch size for training
        datasets=("Sentiment140",),  # Specify the dataset to be used for training
        cuda=False,  # Disable CUDA (GPU) usage for training
        ds_limit_size=1600000  # Limit the dataset size to 1,600,000 samples
    )

    plot_acc(train_acc, test_acc)  # Plot the training and testing accuracies

    print(predict(model, ds, "Chair"))  # Make a prediction on the input "Chair" and print the result

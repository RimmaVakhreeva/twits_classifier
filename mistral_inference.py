from typing import Optional, Any

import jsonlines
import pandas as pd
from transformers import pipeline, Pipeline

model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
)


def model_inference(model: Pipeline, tweet: str) -> Optional[str]:
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": tweet},
    ]
    new_messages = model(messages,
                         temperature=0.2,
                         max_new_tokens=100,
                         do_sample=True)
    content = new_messages[0]['generated_text'][-1]['content']

    classes = ["negative", "neutral", "positive"]
    to_find = "sentiment: "
    idx = content.rfind(to_find)
    if idx != -1:
        sentiment = content[idx + len(to_find):]
        for c in classes:
            if c in sentiment.lower():
                return c
    else:
        return None


if __name__ == "__main__":
    sentiment140_data = pd.read_csv(
        "data/training.1600000.processed.noemoticon.csv",
        encoding_errors="ignore",
        header=None
    )

    sentiment140_data = sentiment140_data.rename(
        columns={0: 'target', 1: 'ids',
                 2: 'date', 3: 'flag',
                 4: 'user', 5: 'text'}
    )
    shuffled_data = sentiment140_data.sample(frac=1).reset_index(drop=True)

    for index, row in shuffled_data[:1000].iterrows():
        tweet = row['text']
        gt_sentiment = "negative" if row['target'] == 0 else "positive"
        pred_sentiment = model_inference(tweet)
        jsonlines.open("output_data_mistral7b.jsonl", "a").write({
            "tweet": tweet,
            "gt_sentiment": gt_sentiment,
            "pred_sentiment": pred_sentiment,
        })
        print(f'{index}: Tweet: "{tweet}""\nPredicted sentiment: {pred_sentiment}, GT sentiment: {gt_sentiment}\n')

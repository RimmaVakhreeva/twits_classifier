"""
This script calculates and visualizes evaluation metrics for sentiment analysis.
It reads data from a JSON Lines file, computes accuracy, F1 score, and confusion matrix,
and then displays these metrics along with a heatmap of the confusion matrix.
"""

# Import libraries
import jsonlines
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def metrics(data: jsonlines) -> tuple:
    """
    Calculate evaluation metrics for sentiment analysis.

    Parameters:
    data (jsonlines): A jsonlines object containing the data with ground truth and predicted sentiments.

    Returns:
    tuple: A tuple containing accuracy, F1 score, and confusion matrix.
    """
    filtered_data = []  # Initialize a list to hold filtered data
    for row in data:  # Iterate over each row in the data
        if row['pred_sentiment'] == 'neutral':  # Skip rows where the predicted sentiment is 'neutral'
            continue
        filtered_data.append(row)  # Append the row to filtered_data if it's not neutral

    # Extract ground truth and predicted sentiments from the filtered data
    gt_sentiments = [row['gt_sentiment'] for row in filtered_data]
    pred_sentiments = [row['pred_sentiment'] for row in filtered_data]

    # Calculate accuracy
    accuracy = accuracy_score(gt_sentiments, pred_sentiments)
    # Calculate F1 score with micro averaging
    f1 = f1_score(gt_sentiments, pred_sentiments, average='micro')
    # Generate confusion matrix
    conf_matrix = confusion_matrix(gt_sentiments, pred_sentiments)

    return accuracy, f1, conf_matrix  # Return the calculated metrics


# Load data from a JSON Lines file
data = list(jsonlines.open("output_data.jsonl", "r"))
# Print the accuracy rounded to 2 decimal places
print(f'accuracy: {round(metrics(data)[0], 2)}')
# Print the F1 score rounded to 2 decimal places
print(f'f1: {round(metrics(data)[1], 2)}')

# Create a heatmap for the confusion matrix
sns.heatmap(metrics(data)[2],
            annot=True,  # Annotate cells with the numeric value
            fmt='g',  # Format for the annotations
            xticklabels=['Positive', 'Negative'],  # X-axis labels
            yticklabels=['Positive', 'Negative'])  # Y-axis labels
plt.ylabel('Actual', fontsize=13)  # Set the Y-axis label
plt.title('Confusion Matrix', fontsize=17, pad=20)  # Set the title of the plot
plt.gca().xaxis.set_label_position('top')  # Move the X-axis label to the top
plt.xlabel('Prediction', fontsize=13)  # Set the X-axis label
plt.gca().xaxis.tick_top()  # Move the ticks to the top of the X-axis

plt.gca().figure.subplots_adjust(bottom=0.2)  # Adjust the subplot parameters
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)  # Add text to the figure
plt.show()  # Display the plot

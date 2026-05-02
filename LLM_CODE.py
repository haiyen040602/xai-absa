# ======================================================================================================================
# 1) CREATION OF TEST SETS (jsonl file) FOR THE DATASETS
# ======================================================================================================================

import jsonlines
from sklearn.model_selection import train_test_split

# File paths
input_file_path = 'NAVER_DATASET.jsonl'
output_file_path = 'NAVER_LLM_TEST_SET.jsonl'

# Load the original dataset
sentences = []
aspect_polarities = []

with jsonlines.open(input_file_path) as reader:
    for line in reader:
        text = line['text']
        sentences.append(text)

        labels = line['labels']
        aspects = [label['aspect'] for label in labels]
        polarities = [label['polarity'] for label in labels]

        aspect_polarities.append({'aspects': aspects, 'polarities': polarities})

# Split the data into training, validation, and test sets (70% training, 15% validation, 15% test)
sentences_train, sentences_remaining, aspect_polarities_train, aspect_polarities_remaining = train_test_split(
    sentences, aspect_polarities, test_size=0.3, random_state=42
)
sentences_val, sentences_test, aspect_polarities_val, aspect_polarities_test = train_test_split(
    sentences_remaining, aspect_polarities_remaining, test_size=0.5, random_state=42
)

# Prepare the test set data to be written into the new JSONL file
test_set_data = []
for sentence, aspects in zip(sentences_test, aspect_polarities_test):
    entry = {
        'text': sentence,
        'labels': [{'aspect': aspect, 'polarity': polarity} for aspect, polarity in zip(aspects['aspects'], aspects['polarities'])]
    }
    test_set_data.append(entry)

# Write the test set to a new JSONL file
with jsonlines.open(output_file_path, mode='w') as writer:
    writer.write_all(test_set_data)

print(f"Test set has been saved to: {output_file_path}")

# ======================================================================================================================
# ======================================================================================================================

# ======================================================================================================================
# 2) CODE FOR TESTING THE LLM (gpt-4)
# ======================================================================================================================

import openai
import jsonlines
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up your OpenAI API key
openai.api_key = '..........'  # Replace with your actual API key

# Load the test set data
def load_test_set(file_path):
    data = []
    print("Loading the test dataset...")
    with jsonlines.open(file_path) as reader:
        for line in reader:
            data.append(line)
    print(f"Test set loaded. Total entries: {len(data)}")
    return data

# Function to call GPT-4 for predicting aspect sentiment using the chat model endpoint
def get_aspect_sentiment(sentence, aspect):
    # Updated prompt with specific instructions for polarity options
    prompt = (f'In the following sentence: "{sentence}", '
              f'what is the sentiment towards the aspect "{aspect}"? '
              f'Respond with ONLY WORD one of the following: positive, neutral, or negative.')

    try:
        # Use the chat completion API for GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Aspect-Based Sentiment Analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0  # To ensure deterministic results
        )
        # Extract sentiment from the API response and normalize it
        sentiment = response['choices'][0]['message']['content'].strip().lower()

        # Ensure that the sentiment is one of the allowed values
        valid_polarities = {"positive", "negative", "neutral"}
        if sentiment not in valid_polarities:
            return None  # Ignore invalid responses

        return sentiment
    except Exception as e:
        print(f"Error with GPT-4 API: {e}")
        return None

# Function to evaluate GPT-4 on the test set
def evaluate_gpt4_on_absa(test_set):
    true_labels = []
    predicted_labels = []

    print("Starting evaluation...")

    # Track progress by counting entries
    total_entries = len(test_set)
    processed_entries = 0

    for entry in test_set:
        sentence = entry['text']
        for label in entry['labels']:
            aspect = label['aspect']
            true_polarity = label['polarity'].lower()

            # Get GPT-4 prediction
            predicted_polarity = get_aspect_sentiment(sentence, aspect)

            # Only proceed if the predicted polarity is valid
            if predicted_polarity is not None:
                true_labels.append(true_polarity)
                predicted_labels.append(predicted_polarity)

        # Increment processed entries and print progress
        processed_entries += 1
        if processed_entries % 100 == 0:
            print(f"Processed {processed_entries}/{total_entries} entries")

    print("Evaluation completed.")

    # Calculate metrics
    acc = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    print(f"\nResults:\nAccuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Load the test dataset
test_set_path = 'NAVER_LLM_TEST_SET.jsonl' 
test_set = load_test_set(test_set_path)

# Evaluate GPT-4 performance on the test set
evaluate_gpt4_on_absa(test_set)

# ======================================================================================================================

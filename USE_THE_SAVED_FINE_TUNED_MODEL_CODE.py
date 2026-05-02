import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('FINE_TUNED_MODEL_distilbert-base-uncased_lr_5e-6_bs_16_epoch_15_plots_85.95%')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Ensure the model is in evaluation mode
model.eval()

# Define the device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a label mapping for the classes
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}


def predict_aspect_polarities_multiple(texts_with_aspects):
    """
    Predict the sentiment polarity for each aspect in the given texts.
    Each input should be a tuple of (text, list_of_aspects).
    Returns the predicted sentiment along with the class probabilities for each text and aspect.
    """
    all_results = []

    for text, aspects in texts_with_aspects:
        text_results = []
        for aspect in aspects:
            # Prepare the input text and aspect for the model
            input_text = f"{text} [SEP] {aspect}"
            tokenized_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(device)

            # Make predictions
            with torch.no_grad():
                outputs = model(**tokenized_input)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            # Get the predicted label and probabilities for each class
            predicted_label_idx = np.argmax(probabilities, axis=1)[0]
            predicted_label = label_map[predicted_label_idx]
            prob_negative, prob_neutral, prob_positive = probabilities[0]

            text_results.append({
                'aspect': aspect,
                'predicted_polarity': predicted_label,
                'probabilities': {
                    'Negative': f"{prob_negative * 100:.2f}%",
                    'Neutral': f"{prob_neutral * 100:.2f}%",
                    'Positive': f"{prob_positive * 100:.2f}%"
                }
            })
        all_results.append({
            'text': text,
            'results': text_results
        })

    return all_results


# Example input with multiple texts and their corresponding aspects
texts_with_aspects = [
    ("The phone has an amazing camera, but the battery life is disappointing.", ["camera", "battery life"]),
    ("The service at the restaurant was excellent, but the food was average.", ["service", "food"]),
    ("I love the design of the laptop, but the performance could be better.", ["design", "performance"])
]

# Get predictions for each text and its aspects
predictions = predict_aspect_polarities_multiple(texts_with_aspects)

# Print the results
for prediction in predictions:
    print(f"Text: {prediction['text']}")
    for result in prediction['results']:
        print(f"  Aspect: {result['aspect']}")
        print(f"    Predicted Polarity: {result['predicted_polarity']}")
        print(f"    Class Probabilities: {result['probabilities']}")
    print()

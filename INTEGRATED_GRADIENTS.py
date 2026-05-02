from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients, visualization
import torch

# Load fine-tuned model and tokenizer
model_dir = "FINE_TUNED_MODEL_roberta-base_model_lr_1e-5_bs_8_epoch_15_plots_89.16%"
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

texts = [
    "The food at the restaurant was amazing, but the service was terrible.",
    "The food at the restaurant was amazing, but the service was terrible.",
    "The food at the restaurant was amazing, but the service was terrible."
]
aspects = [
    "food",
    "service",
    "restaurant"
]

# True labels for the reviews
true_labels = [2, 0, 1]  #  2: positive, 1: neutral, 0: negative

# Initialize LayerIntegratedGradients
lig = LayerIntegratedGradients(lambda input_ids, attention_mask: model(input_ids, attention_mask=attention_mask).logits, model.base_model.embeddings)

# Function to construct input and reference pairs
def construct_input_ref_pair(text, aspect, tokenizer):
    input_text = f"{text} </s></s> {aspect}"
    tokenized_input = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
    input_ids = tokenized_input['input_ids']
    attention_mask = tokenized_input['attention_mask']
    return input_ids, attention_mask

# Define label map and visualize for each text-aspect pair
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

for text, aspect, true_label in zip(texts, aspects, true_labels):
    # Construct input and reference pairs
    input_ids, attention_mask = construct_input_ref_pair(text, aspect, tokenizer)

    # Predict sentiment
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Compute attributions
    target = predicted_class
    attributions, delta = lig.attribute(inputs=input_ids, baselines=None, target=target, additional_forward_args=(attention_mask,), return_convergence_delta=True)

    # Summarize attributions
    attributions_sum = attributions.sum(dim=-1).squeeze(0) / torch.norm(attributions.sum(dim=-1)).squeeze(0)

    # Convert input IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    # Print out tokens with their corresponding attribution values
    print(f"Text: {text}")
    print(f"Aspect: {aspect}")
    print(f"Predicted Sentiment: {label_map[predicted_class]}")
    print(f"True Sentiment: {label_map[true_label]}")
    print("Word Importance Scores:")
    for token, score in zip(tokens, attributions_sum.tolist()):
        print(f"{token}: {score:.4f}")

    # Visualize attributions
    visualization.visualize_text([visualization.VisualizationDataRecord(
        attributions_sum,
        torch.max(torch.softmax(logits, dim=1)),
        predicted_class,
        label_map[true_label],  # True label
        label_map[predicted_class],  # Predicted label
        attributions_sum.sum(),
        tokens,
        delta
    )])

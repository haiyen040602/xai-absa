import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns


class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        # Handle the tuple case for XLNet
        if isinstance(output, tuple):
            self.activations = output[0]
        else:
            self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute_gradcam(self, input_text, aspect, tokenizer):
        # Combine input text and aspect

        # combined_input = f"{input_text} [SEP] {aspect}"
        combined_input = f"{input_text} </s></s> {aspect}"
        # combined_input = f"{input_text} <sep> {aspect}"

        # Tokenize the combined input
        inputs = tokenizer(combined_input, return_tensors='pt')

        self.model.eval()

        # Forward pass to get model outputs
        outputs = self.model(**inputs)

        # Get the target class index
        target_class = torch.argmax(outputs.logits, dim=1).item()

        # Get the score for the target class
        target_score = outputs.logits[0, target_class]

        # Backward pass to compute gradients
        self.model.zero_grad()
        target_score.backward()

        # Extract the gradients and the last hidden state
        gradients = self.gradients.mean(dim=1).squeeze()
        activations = self.activations.squeeze()

        # Compute weights
        weights = gradients

        # Generate relevance scores by combining weights with hidden states
        cam = torch.zeros(activations.size(0))
        for i in range(activations.size(0)):
            cam[i] = (weights * activations[i]).sum()

        # Apply ReLU to the result
        cam = torch.relu(cam)

        # Normalize the relevance scores
        cam = cam / cam.max()

        return cam, inputs['input_ids'][0], target_class


# Example
input_text = "The food at the restaurant was amazing, but the service was terrible."
aspect = "food"

# input_text = "Excellent location and friendly staff, but disappointing room cleanliness. Breakfast was adequate."
# aspect = "room cleanliness"


# Model and tokenizer selection
model_files = {
    'roberta': ('FINE_TUNED_MODEL_roberta-base_model_lr_1e-5_bs_8_epoch_15_plots_89.16%', 'roberta-base'),
    'albert': ('FINE_TUNED_MODEL_albert-base-v1_lr_3e-5_bs_128_epoch_15_plots_85.49%', 'albert-base-v1'),
    'bert': ('FINE_TUNED_MODEL_bert-base-uncased_lr_2e-5_bs_8_epoch_15_plots_86.64%', 'bert-base-uncased'),
    'distilbert': ('FINE_TUNED_MODEL_distilbert-base-uncased_lr_5e-6_bs_16_epoch_15_plots_85.95%', 'distilbert-base-uncased'),
    'xlnet': ('FINE_TUNED_MODEL_xlnet-base-cased_lr_2e-5_bs_16_epoch_15_plots_88.68%', 'xlnet-base-cased')
}

model_type = 'roberta'

# Load tokenizer and model
model_path, tokenizer_name = model_files[model_type]
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Select the last layer of the encoder based on model type
if model_type == 'roberta':
    target_layer = model.roberta.encoder.layer[-1].output
elif model_type == 'albert':
    target_layer = model.albert.encoder.albert_layer_groups[-1].albert_layers[-1].ffn_output
elif model_type == 'bert':
    target_layer = model.bert.encoder.layer[-1].output
elif model_type == 'distilbert':
    target_layer = model.distilbert.transformer.layer[-1].ffn.lin2
elif model_type == 'xlnet':
    target_layer = model.transformer.layer[-1]

# Initialize Grad-CAM
gradcam = GradCam(model, target_layer)

# Compute Grad-CAM scores for the aspect
cam_scores, input_ids, target_class = gradcam.compute_gradcam(input_text, aspect, tokenizer)

# Convert token IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# Combine tokens with their corresponding Grad-CAM scores
highlighted_text = [(token, score.item()) for token, score in zip(tokens, cam_scores)]

# Print the tokens with their relevance scores
print(f"Grad-CAM Relevance Scores for Aspect '{aspect}' in Input Text:")
for token, score in highlighted_text:
    print(f"{token}: {score:.4f}")

# Define a mapping from class index to sentiment polarity
polarity_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
predicted_polarity = polarity_mapping.get(target_class, 'unknown')

# Print the predicted polarity
print(f"\nPredicted Polarity for Aspect '{aspect}': {predicted_polarity}")

# Visualize the results in a heatmap without annotations
plt.figure(figsize=(12, 6))
ax = sns.heatmap([cam_scores.detach().numpy()], cmap='coolwarm', cbar_kws={'label': 'Relevance Score'})
ax.set_title(f'Grad-CAM Relevance Scores for Aspect "{aspect}" in Input Text')
ax.set_xlabel('Tokens')
ax.set_xticklabels(tokens, rotation=90)  # Rotate words to be vertical
plt.yticks([], [])  # Hide y-axis labels

# Show the plot
plt.show()

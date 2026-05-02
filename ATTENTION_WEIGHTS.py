import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

model_path = 'FINE_TUNED_MODEL_roberta-base_model_lr_1e-5_bs_8_epoch_15_plots_89.16%'
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=True)

# Define input text and aspect
text = "The food at the restaurant was amazing, but the service was terrible."
aspect = "food"

input_text = f"{text} </s></s> {aspect}"
# input_text = f"{text} <sep> {aspect}"
# input_text = f"{text} [SEP] {aspect}"


# Tokenize input text
tokens = tokenizer.tokenize(input_text)
token_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get model output (output_hidden_states=True to get attention weights)
with torch.no_grad():
    outputs = model(token_ids)

# Extract attention weights from the last layer
attention_weights = outputs.attentions[-1]

# Average attention weights across all attention heads
avg_attention_weights = torch.mean(attention_weights, dim=1).squeeze(0)

# Convert token IDs to words
word_tokens = tokenizer.convert_ids_to_tokens(token_ids.squeeze().tolist())

# Visualize attention weights
plt.figure(figsize=(12, 10))
plt.imshow(avg_attention_weights, cmap='coolwarm', interpolation='nearest')

plt.xticks(ticks=range(len(word_tokens)), labels=word_tokens, rotation=45)
plt.yticks(ticks=range(len(word_tokens)), labels=word_tokens)

# Annotate heatmap with attention weights
for i in range(len(word_tokens)):
    for j in range(len(word_tokens)):
        plt.text(j, i, f'{avg_attention_weights[i, j]:.2f}', ha='center', va='center', color='white')

plt.xlabel('Source Tokens')
plt.ylabel('Target Tokens')
plt.title(f'Attention Weights from the Last Layer (Aspect: {aspect})')
plt.colorbar(label='Attention Weight')
plt.show()

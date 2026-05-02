import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import shap

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
model_name_or_path = "FINE_TUNED_MODEL_roberta-base_model_lr_1e-5_bs_8_epoch_15_plots_89.16%"
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(device)

# Function to predict sentiment for a given text and aspect
def predict_sentiment(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    return predictions.detach().cpu().numpy()  # Move outputs to CPU


# Example text and aspect

text = "The food at the restaurant was amazing, but the service was terrible."
aspect = "restaurant"

# text = "Excellent location and friendly staff, but disappointing room cleanliness. Breakfast was adequate."
# aspect = "room cleanliness"


# SHAP explainer

explainer = shap.Explainer(lambda x: predict_sentiment([f"{t} </s></s> {aspect}" for t in x]), tokenizer)
# explainer = shap.Explainer(lambda x: predict_sentiment([f"{t} [SEP] {aspect}" for t in x]), tokenizer)
# explainer = shap.Explainer(lambda x: predict_sentiment([f"{t} <sep> {aspect}" for t in x]), tokenizer)

# Prepare the input text in the expected format
input_text = [text]

# Compute SHAP values
shap_values = explainer(input_text)

# Visualization
shap.initjs()
shap.text_plot(shap_values)

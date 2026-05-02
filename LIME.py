import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from lime.lime_text import LimeTextExplainer
from collections import defaultdict

model_path = 'FINE_TUNED_MODEL_roberta-base_model_lr_1e-5_bs_8_epoch_15_plots_89.16%'
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(texts, aspects):
    predictions = []
    for text, aspect in zip(texts, aspects):
        
        input_text = f"{text} </s></s> {aspect}"
        # input_text = f"{text} <sep> {aspect}"
        #input_text = f"{text} [SEP] {aspect}"

        tokenized_text = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**tokenized_text)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
        predictions.append(probabilities[0])  # Extract the probabilities array
    return np.array(predictions)

explainer = LimeTextExplainer(class_names=["negative", "neutral", "positive"])


text = "The food at the restaurant was amazing, but the service was terrible."
aspect = "restaurant"

def lime_predict(texts):
    aspects = [aspect] * len(texts)
    return predict_sentiment(texts, aspects)

def explain_instance_with_class_columns(explainer, text, predict_fn, num_features, num_samples):
    print(f"Explaining instance with text: {text}")
    explanation = explainer.explain_instance(text, predict_fn, num_features=num_features, num_samples=num_samples, top_labels=len(explainer.class_names))
    print("Explanation generated.")
    print(f"Class names: {explanation.class_names}")
    print(f"Local explanation: {explanation.local_exp}")

    class_weights = defaultdict(list)

    for class_index, class_name in enumerate(explainer.class_names):
        if class_index not in explanation.local_exp:
            print(f"Class index {class_index} not found in local_exp")
            continue

        print(f"\nProcessing class '{class_name}' with index {class_index}")
        for feature, weight in explanation.local_exp[class_index]:
            word = explanation.domain_mapper.indexed_string.word(feature)
            print(f"Feature index: {feature}, Word: {word}, Weight: {weight}")
            class_weights[class_name].append((word, weight))

    # Display the class weights
    for class_name, weights in class_weights.items():
        print(f"\nWeights for class '{class_name}':")
        for word, weight in weights:
            print(f"{word}: {weight}")

    return explanation

exp = explain_instance_with_class_columns(explainer, text, lime_predict, num_features=5, num_samples=1000)

# Display the explanation in notebook
exp.show_in_notebook()

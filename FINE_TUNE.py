from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support


file_path = 'FINAL_CLEANED_CORRECTED_SHUFFLED_DATASET_NO_DUPLICATE.jsonl'
sentences = []
aspect_polarities = []

with jsonlines.open(file_path) as reader:
    for line in reader:
        text = line['text']
        sentences.append(text)

        labels = line['labels']
        aspects = [label['aspect'] for label in labels]
        polarities = [label['polarity'] for label in labels]

        aspect_polarities.append({'aspects': aspects, 'polarities': polarities})

# Split the data into training, validation, and test sets
sentences_train, sentences_remaining, aspect_polarities_train, aspect_polarities_remaining = train_test_split(
    sentences, aspect_polarities, test_size=0.3, random_state=42
)
sentences_val, sentences_test, aspect_polarities_val, aspect_polarities_test = train_test_split(
    sentences_remaining, aspect_polarities_remaining, test_size=0.5, random_state=42
)

# Initialize the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Prepare tokenized data and labels for training
tokenized_data_train = []
labels_train = []

for text, aspect_polarity in zip(sentences_train, aspect_polarities_train):
    aspects = aspect_polarity['aspects']
    polarities = aspect_polarity['polarities']

    for aspect, polarity in zip(aspects, polarities):
        input_text = f"{text} [SEP] {aspect}"
        tokenized_text = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
        tokenized_data_train.append(tokenized_text)

        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        label = label_map[polarity]
        labels_train.append(label)

# Convert labels to tensor
labels_train = torch.tensor(labels_train)

# Create TensorDataset and DataLoader for training
dataset_train = TensorDataset(
    pad_sequence([x.input_ids.squeeze(0) for x in tokenized_data_train], batch_first=True),
    pad_sequence([x.attention_mask.squeeze(0) for x in tokenized_data_train], batch_first=True),
    labels_train
)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)

# Prepare tokenized data and labels for validation
tokenized_data_val = []
labels_val = []

for text, aspect_polarity in zip(sentences_val, aspect_polarities_val):
    aspects = aspect_polarity['aspects']
    polarities = aspect_polarity['polarities']

    for aspect, polarity in zip(aspects, polarities):
        input_text = f"{text} [SEP] {aspect}"
        tokenized_text = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
        tokenized_data_val.append(tokenized_text)

        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        label = label_map[polarity]
        labels_val.append(label)

# Convert labels to tensor for validation
labels_val = torch.tensor(labels_val)

# Create TensorDataset and DataLoader for validation
dataset_val = TensorDataset(
    pad_sequence([x.input_ids.squeeze(0) for x in tokenized_data_val], batch_first=True),
    pad_sequence([x.attention_mask.squeeze(0) for x in tokenized_data_val], batch_first=True),
    labels_val
)
dataloader_val = DataLoader(dataset_val, batch_size=8, shuffle=False)

# Prepare tokenized data and labels for test
tokenized_data_test = []
labels_test = []

for text, aspect_polarity in zip(sentences_test, aspect_polarities_test):
    aspects = aspect_polarity['aspects']
    polarities = aspect_polarity['polarities']

    for aspect, polarity in zip(aspects, polarities):
        input_text = f"{text} [SEP] {aspect}"
        tokenized_text = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
        tokenized_data_test.append(tokenized_text)

        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        label = label_map[polarity]
        labels_test.append(label)

# Convert labels to tensor for test
labels_test = torch.tensor(labels_test)

# Create TensorDataset and DataLoader for test
dataset_test = TensorDataset(
    pad_sequence([x.input_ids.squeeze(0) for x in tokenized_data_test], batch_first=True),
    pad_sequence([x.attention_mask.squeeze(0) for x in tokenized_data_test], batch_first=True),
    labels_test
)
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False)


# Function to generate confusion matrix
def generate_confusion_matrix(model, dataloader):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, batch_labels = batch

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)

            true_labels.extend(batch_labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    return conf_matrix


# Initialize lists to store losses and accuracy
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Fine-tune the model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch_num, batch in enumerate(dataloader_train, 1):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, batch_labels = batch

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if batch_num % 8 == 0:
            avg_loss = total_loss / total_batches
            print(f"Batch {batch_num}/{len(dataloader_train)} - Loss: {avg_loss:.4f}")

    avg_epoch_loss = total_loss / total_batches
    train_losses.append(avg_epoch_loss)
    print(f"\nAverage Training Loss for Epoch {epoch + 1}: {avg_epoch_loss:.4f}")

    # Calculate training accuracy
    model.eval()
    total_correct_preds_train = 0
    total_preds_train = 0
    with torch.no_grad():
        for batch in dataloader_train:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, batch_labels = batch

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)

            total_preds_train += batch_labels.size(0)
            total_correct_preds_train += (predicted == batch_labels).sum().item()

    accuracy_train = total_correct_preds_train / total_preds_train if total_preds_train > 0 else 0
    train_accuracies.append(accuracy_train)
    print(f"Training Accuracy for Epoch {epoch + 1}: {accuracy_train * 100:.2f}%")

    # Validation part
    model.eval()
    total_correct_preds_val = 0
    total_preds_val = 0
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in dataloader_val:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, batch_labels = batch

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            _, predicted = torch.max(outputs.logits, 1)

            total_preds_val += batch_labels.size(0)
            total_correct_preds_val += (predicted == batch_labels).sum().item()

            total_val_loss += outputs.loss.item()

    accuracy_val = total_correct_preds_val / total_preds_val if total_preds_val > 0 else 0
    avg_val_loss = total_val_loss / len(dataloader_val)
    val_losses.append(avg_val_loss)
    val_accuracies.append(accuracy_val)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy_val * 100:.2f}%")

# Plot training and validation losses
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# Plot training and validation accuracy
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Confusion matrix for validation set
conf_matrix_val = generate_confusion_matrix(model, dataloader_val)

classes = ['Negative', 'Neutral', 'Positive']


# Plot confusion matrix for validation set
disp = ConfusionMatrixDisplay(conf_matrix_val, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Validation Set)')
plt.show()


# Evaluation on test set
model.eval()
total_correct_preds_test = 0
total_preds_test = 0
true_labels_test = []
predicted_labels_test = []

with torch.no_grad():
    for batch in dataloader_test:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, batch_labels = batch

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)

        true_labels_test.extend(batch_labels.cpu().numpy())
        predicted_labels_test.extend(predicted.cpu().numpy())

        total_preds_test += batch_labels.size(0)
        total_correct_preds_test += (predicted == batch_labels).sum().item()

accuracy_test = total_correct_preds_test / total_preds_test if total_preds_test > 0 else 0
print(f"Accuracy on test set: {accuracy_test * 100:.2f}%")

# Compute confusion matrix for test set
conf_matrix_test = generate_confusion_matrix(model, dataloader_test)

# Compute precision, recall, F1 score and support (count of true instances) for each class and their averages
precision, recall, f1_score, support = precision_recall_fscore_support(true_labels_test, predicted_labels_test, average='macro')

# Print macro precision, recall, F1 score
print("Macro Metrics:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1_score:.4f}")

# Plot confusion matrix for test set
disp = ConfusionMatrixDisplay(conf_matrix_test, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set)')
plt.show()


# # Save the model and tokenizer
# model.save_pretrained('distilbert-base-uncased_fine_tuned_aspect_sentiment_model_lr_1e-5_bs_8_epoch_6_plots')
# tokenizer.save_pretrained('tokenizer_distilbert-base-uncased_fine_tuned_aspect_sentiment_model_lr_1e-5_bs_8_epoch_6_plots')

# Save the model and tokenizer
model.save_pretrained('FINE_TUNED_MODEL_roberta-base_model_lr_3e-6_bs_64_epoch_15_plots')
tokenizer.save_pretrained('tokenizer_roberta-base_lr_3e-6_bs_64_epoch_15_plots')

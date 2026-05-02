import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import jsonlines
import numpy as np
import re
from collections import Counter
import pickle
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Step 1: Load and Prepare the Dataset
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

# Step 2: Tokenization and Padding

# Simple tokenizer
def simple_tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())

# Build vocabulary
all_tokens = [simple_tokenizer(sentence) for sentence in sentences]
word_counts = Counter([token for tokens in all_tokens for token in tokens])
vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.items())}  # Reserve 0 for padding and 1 for unknown words
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1
vocab['[SEP]'] = len(vocab) + 1

# Save the vocabulary
with open('vocab.pkl', 'wb') as vocab_file:
    pickle.dump(vocab, vocab_file)

# Encode sentences
def encode_sentence(sentence, vocab):
    tokens = simple_tokenizer(sentence)
    encoded_tokens = [vocab.get(word, vocab['<UNK>']) for word in tokens]
    return encoded_tokens

# Convert to PyTorch tensors and pad sequences
def pad_and_tensorize(encoded_sentences, max_len):
    padded_sentences = pad_sequence([torch.tensor(sentence[:max_len]) for sentence in encoded_sentences],
                                    batch_first=True, padding_value=vocab['<PAD>'])
    return padded_sentences

# Prepare tokenized data and labels for training, validation, and test sets
def prepare_data(sentences, aspect_polarities, vocab, max_len):
    tokenized_data = []
    labels = []
    text_lengths = []  # Store lengths of sequences before padding

    for sentence, aspect_polarity in zip(sentences, aspect_polarities):
        aspects = aspect_polarity['aspects']
        polarities = aspect_polarity['polarities']

        for aspect, polarity in zip(aspects, polarities):
            input_text = f"{sentence} [SEP] {aspect}"
            encoded_sentence = encode_sentence(input_text, vocab)
            text_lengths.append(min(len(encoded_sentence), max_len))  # Get length before padding
            tokenized_data.append(encoded_sentence)

            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            label = label_map[polarity]
            labels.append(label)

    # Convert to tensors
    inputs_tensor = pad_and_tensorize(tokenized_data, max_len)
    labels_tensor = torch.tensor(labels)

    return inputs_tensor, labels_tensor, torch.tensor(text_lengths)

# Maximum sequence length
max_len = 50

# Prepare the datasets
train_inputs, train_labels, train_lengths = prepare_data(sentences_train, aspect_polarities_train, vocab, max_len)
val_inputs, val_labels, val_lengths = prepare_data(sentences_val, aspect_polarities_val, vocab, max_len)
test_inputs, test_labels, test_lengths = prepare_data(sentences_test, aspect_polarities_test, vocab, max_len)

# Create DataLoaders (with lengths)
train_dataset = TensorDataset(train_inputs, train_labels, train_lengths)
val_dataset = TensorDataset(val_inputs, val_labels, val_lengths)
test_dataset = TensorDataset(test_inputs, test_labels, test_lengths)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Step 3: Define the LSTM Model (Unidirectional)
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<PAD>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=False, dropout=dropout, batch_first=True)  # Unidirectional
        self.fc = nn.Linear(hidden_dim, output_dim)  # No *2 since unidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)  # Ensure lengths are on CPU
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        hidden = self.dropout(hidden[-1])  # Last hidden state of the LSTM
        return self.fc(hidden)

# Initialize model parameters
embedding_dim = 100
hidden_dim = 256
output_dim = 3
n_layers = 2
dropout = 0.3

model = LSTM(len(vocab), embedding_dim, hidden_dim, output_dim, n_layers, dropout)

# Step 4: Train the LSTM Model
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Ensure the model is on the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch_idx, batch in enumerate(iterator, 1):
        text, labels, lengths = batch
        text = text.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        predictions = model(text, lengths).squeeze(1)
        loss = criterion(predictions, labels)
        acc = (predictions.argmax(dim=1) == labels).float().mean()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(iterator)} - Loss: {loss.item():.4f} - Acc: {acc.item():.4f}")

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in iterator:
            text, labels, lengths = batch
            text = text.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            predictions = model(text, lengths).squeeze(1)

            loss = criterion(predictions, labels)
            acc = (predictions.argmax(dim=1) == labels).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            # Store the true labels and predicted labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.argmax(dim=1).cpu().numpy())

    return epoch_loss / len(iterator), epoch_acc / len(iterator), all_labels, all_preds

n_epochs = 10

for epoch in range(n_epochs):
    print(f"\nEpoch {epoch + 1}/{n_epochs}")
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)[:2]
    print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc * 100:.2f}%')

# Step 5: Evaluate the Model on the Test Set

print("\nEvaluating on Test Set...")
test_loss, test_acc, true_labels_test, predicted_labels_test = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

# Step 6: Compute Macro Precision, Recall, F1-Score, and Accuracy
precision, recall, f1_score, support = precision_recall_fscore_support(true_labels_test, predicted_labels_test, average='macro')
accuracy = accuracy_score(true_labels_test, predicted_labels_test)

print(f"Macro Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model and vocabulary
torch.save(model.state_dict(), 'lstm_baseline_model.pth')

import pandas as pd
import re
import numpy as np
import nltk
from collections import Counter
import torch
from sklearn.model_selection import train_test_split

# Download the punkt tokenizer model
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def preprocess(file_path, sequence_length=25):

    # Load dataset
    IMDB_review_data = pd.read_csv(file_path)

    # Lowercase all text.
    IMDB_review_data['review'] = IMDB_review_data['review'].str.lower()

    # Remove punctuation and special characters.
    IMDB_review_data['review'] = IMDB_review_data['review'].apply(lambda x: re.sub(r'<.*?>', ' ', x))
    IMDB_review_data['review'] = IMDB_review_data['review'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    IMDB_review_data['review'] = IMDB_review_data['review'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    # Tokenize sentences (using nltk.word_tokenize)
    tokenized = IMDB_review_data['review'].apply(nltk.word_tokenize)

    # Keep only the top 10,000 most frequent words.
    all_tokens = [word for review in tokenized for word in review]
    frequency = Counter(all_tokens)
    vocabulary = {word: i+2 for i, (word, _) in enumerate(frequency.most_common(10000))}
    vocabulary['<infrequent>'] = 0 
    vocabulary['<frequent>'] = 1

    # Convert each review to a sequence of token IDs.
    sequences = []
    for review in tokenized:
        encoded = [vocabulary.get(word, vocabulary['<frequent>']) for word in review]
        sequences.append(torch.tensor(encoded, dtype=torch.long))

    # Pad or truncate sequences to fixed lengths
    padded = torch.zeros((len(sequences), sequence_length), dtype=torch.long)
    i = 0
    while i < len(sequences):
        seq = sequences[i]
        length = min(len(seq), sequence_length)
        padded[i, :length] = seq[:length]
        i += 1
    y = IMDB_review_data['sentiment']

    # Split train-test sets (50%-50%)
    X_train, X_test, y_train, y_test = train_test_split(
        padded, y, test_size=0.5, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, vocabulary





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.metrics import accuracy_score, f1_score

def RNN_algorithm(file_path, activation_function='tanh', optimizer_function='adam', sequence_length=25, stability_strategy='none', epochs_number=5):

    # Preprocess data
    X_train, X_test, y_train, y_test, vocabulary = preprocess(file_path, sequence_length=sequence_length)
    vocab_size = len(vocabulary)

    # Extra step, which attach our algorithm to GPU if available, which can significantly speed up the training process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert the label, where positive -> True -> 1
    # On the other hand, negative -> False -> 0
    y_train = (y_train == "positive").astype(int).values
    y_test = (y_test == "positive").astype(int).values
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Transform data to fit GPU memory
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    # Activation function
    if activation_function == 'relu':
        activation = nn.ReLU()
    elif activation_function == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = nn.Tanh()

    # Model setup
    embedding = nn.Embedding(vocab_size, 100, padding_idx=0)
    rnn = nn.RNN(
        input_size=100, hidden_size=64, num_layers=2,
        dropout=0.5, batch_first=True, nonlinearity='tanh'
    )
    dropout = nn.Dropout(0.5)
    fc = nn.Linear(64, 1)
    output_activation = nn.Sigmoid()

    # Pass model to GPU
    embedding, rnn, dropout, fc, output_activation = (
        embedding.to(device),
        rnn.to(device),
        dropout.to(device),
        fc.to(device),
        output_activation.to(device)
    )

    # Loss and optimizer
    criterion = nn.BCELoss()
    parameters = list(embedding.parameters()) + list(rnn.parameters()) + list(fc.parameters())
    if optimizer_function.lower() == 'sgd':
        optimizer = optim.SGD(parameters, lr=0.001)
    elif optimizer_function.lower() == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=0.001)
    else:
        optimizer = optim.Adam(parameters, lr=0.001)

    # Train the model
    epoch_losses = []
    epoch_times = []
    for epoch in range(epochs_number):
        embedding.train(), rnn.train(), fc.train()
        epoch_loss = 0
        start_time = time.time()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            embedded = embedding(X_batch)
            rnn_out, hidden = rnn(embedded)
            last_hidden = hidden[-1]
            dropped = dropout(last_hidden)
            activated = activation(dropped)
            y_pred = output_activation(fc(activated)).squeeze(1)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            # Gradient clip option
            if stability_strategy == 'clip':
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Record epoch results
        average_train_loss = epoch_loss / len(train_loader)
        time_consume_epoch = time.time() - start_time
        epoch_losses.append(average_train_loss)
        epoch_times.append(time_consume_epoch)
        print(f"Epoch [{epoch+1}/{epochs_number}] finish")

    average_epoch_time = sum(epoch_times) / len(epoch_times)
    print("\n RNN training complete")

    # Gather the test results that will be used for evaluation
    embedding.eval(), rnn.eval(), fc.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            embedded = embedding(X_batch)
            rnn_out, hidden = rnn(embedded)
            last_hidden = hidden[-1]
            dropped = dropout(last_hidden)
            activated = activation(dropped)
            y_pred = output_activation(fc(activated)).squeeze(1)
            preds = (y_pred > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"\n test result are | Accuracy: {acc:.4f} | F1: {f1:.4f} | Avg Epoch Time: {average_epoch_time:.2f}s")

    return acc, f1, epoch_losses, epoch_times, average_epoch_time





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.metrics import accuracy_score, f1_score

def LSTM_algorithm(file_path, activation_function='tanh', optimizer_function='adam', sequence_length=25, stability_strategy='none', epochs_number=5):

    # Preprocess data
    X_train, X_test, y_train, y_test, vocabulary = preprocess(file_path, sequence_length=sequence_length)
    vocab_size = len(vocabulary)

    # Attach to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    y_train = (y_train == "positive").astype(int).values
    y_test = (y_test == "positive").astype(int).values
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    # Activation function
    if activation_function == 'relu':
        activation = nn.ReLU()
    elif activation_function == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = nn.Tanh()

    # Model setup
    embedding = nn.Embedding(vocab_size, 100, padding_idx=0)
    lstm = nn.LSTM(
        input_size=100, hidden_size=64, num_layers=2,
        dropout=0.5, batch_first=True
    )
    dropout = nn.Dropout(0.5)
    fc = nn.Linear(64, 1)
    output_activation = nn.Sigmoid()

    # Move to GPU
    embedding, lstm, dropout, fc, output_activation = (
        embedding.to(device),
        lstm.to(device),
        dropout.to(device),
        fc.to(device),
        output_activation.to(device)
    )

    # Loss function and optimizer
    criterion = nn.BCELoss()
    parameters = list(embedding.parameters()) + list(lstm.parameters()) + list(fc.parameters())
    if optimizer_function.lower() == 'sgd':
        optimizer = optim.SGD(parameters, lr=0.01)
    elif optimizer_function.lower() == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=0.001)
    else:
        optimizer = optim.Adam(parameters, lr=0.001)

    # Training the model
    epoch_losses = []
    epoch_times = []
    for epoch in range(epochs_number):
        embedding.train(), lstm.train(), fc.train()
        epoch_loss = 0
        start_time = time.time()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            embedded = embedding(X_batch)
            lstm_out, (hidden, cell) = lstm(embedded)
            last_hidden = hidden[-1]
            dropped = dropout(last_hidden)
            activated = activation(dropped)
            y_pred = output_activation(fc(activated)).squeeze(1)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            # Gradient clip option
            if stability_strategy == 'clip':
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Record results
        average_train_loss = epoch_loss / len(train_loader)
        time_consume_epoch = time.time() - start_time
        epoch_losses.append(average_train_loss)
        epoch_times.append(time_consume_epoch)
        print(f"Epoch [{epoch+1}/{epochs_number}] finish")

    average_epoch_time = sum(epoch_times) / len(epoch_times)
    print("\n LSTM training complete")

    # Gather the test results that will be used for evaluation
    embedding.eval(), lstm.eval(), fc.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            embedded = embedding(X_batch)
            lstm_out, (hidden, cell) = lstm(embedded)
            last_hidden = hidden[-1]
            dropped = dropout(last_hidden)
            activated = activation(dropped)
            y_pred = output_activation(fc(activated)).squeeze(1)
            preds = (y_pred > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"\n test result are | Accuracy: {acc:.4f} | F1: {f1:.4f} | Avg Epoch Time: {average_epoch_time:.2f}s")

    return acc, f1, epoch_losses, epoch_times, average_epoch_time





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.metrics import accuracy_score, f1_score

def Bidirectional_LSTM_algorithm(file_path, activation_function='tanh', optimizer_function='adam', sequence_length=25, stability_strategy='none', epochs_number=5):

    # Preprocess data
    X_train, X_test, y_train, y_test, vocabulary = preprocess(file_path, sequence_length=sequence_length)
    vocab_size = len(vocabulary)

    # Attach to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data + pass data to GPU model
    y_train = (y_train == "positive").astype(int).values
    y_test = (y_test == "positive").astype(int).values
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    # Activation function
    if activation_function == 'relu':
        activation = nn.ReLU()
    elif activation_function == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = nn.Tanh()

    # Model setup + pass to GPU
    embedding = nn.Embedding(vocab_size, 100, padding_idx=0)
    lstm = nn.LSTM(
        input_size=100, hidden_size=64, num_layers=2,
        dropout=0.5, batch_first=True, bidirectional=True
    )
    dropout = nn.Dropout(0.5)
    # Note: bidirectional doubles the hidden size (64 * 2)
    fc = nn.Linear(64 * 2, 1)
    output_activation = nn.Sigmoid()
    embedding, lstm, dropout, fc, output_activation = (
        embedding.to(device),
        lstm.to(device),
        dropout.to(device),
        fc.to(device),
        output_activation.to(device)
    )

    # Loss function and optimizer
    criterion = nn.BCELoss()
    parameters = list(embedding.parameters()) + list(lstm.parameters()) + list(fc.parameters())
    if optimizer_function.lower() == 'sgd':
        optimizer = optim.SGD(parameters, lr=0.01)
    elif optimizer_function.lower() == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=0.001)
    else:
        optimizer = optim.Adam(parameters, lr=0.001)

    # Training the model
    epoch_losses = []
    epoch_times = []
    for epoch in range(epochs_number):
        embedding.train(), lstm.train(), fc.train()
        epoch_loss = 0
        start_time = time.time()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            embedded = embedding(X_batch)
            lstm_out, (hidden, cell) = lstm(embedded)
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
            dropped = dropout(hidden_cat)
            activated = activation(dropped)
            y_pred = output_activation(fc(activated)).squeeze(1)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            # Gradient clip option
            if stability_strategy == 'clip':
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Record epoch information for tracking
        average_train_loss = epoch_loss / len(train_loader)
        time_consume_epoch = time.time() - start_time
        epoch_losses.append(average_train_loss)
        epoch_times.append(time_consume_epoch)
        print(f"Epoch [{epoch+1}/{epochs_number}] finish")
    average_epoch_time = sum(epoch_times) / len(epoch_times)
    print("\n Bidirectional LSTM training complete")

    # Gather the test results that will be used for evaluation
    embedding.eval(), lstm.eval(), fc.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            embedded = embedding(X_batch)
            lstm_out, (hidden, cell) = lstm(embedded)
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
            dropped = dropout(hidden_cat)
            activated = activation(dropped)
            y_pred = output_activation(fc(activated)).squeeze(1)
            preds = (y_pred > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"\n test result are | Accuracy: {acc:.4f} | F1: {f1:.4f} | Avg Epoch Time: {average_epoch_time:.2f}s")

    return acc, f1, epoch_losses, epoch_times, average_epoch_time





import os
import matplotlib.pyplot as plt
import pandas as pd
import time

# Setup the path. Here, please set your own paths accordingly.
file_path = r"E:\学校内容\DATA 641\DATA 641 HW3\data\IMDB_Dataset.csv"
results_dir = r"E:\学校内容\DATA 641\DATA 641 HW3\results"
plots_dir = os.path.join(results_dir, "plots")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Parameter arrangement. Here, to make the algorithm consume less time, we only use the epoch number of 5 and keep it all the time
architectures = {
    "RNN": RNN_algorithm,
    "LSTM": LSTM_algorithm,
    "Bidirectional LSTM": Bidirectional_LSTM_algorithm
}
activation_functions = ["sigmoid", "relu", "tanh"]
optimizers = ["adam", "sgd", "rmsprop"]
sequence_lengths = [25, 50, 100]
stability_strategies = ["none", "clip"]
epochs_number = 5
results = []

# loop to all combinations
# Keep track of which run we are at
total_runs = len(architectures) * len(activation_functions) * len(optimizers) * len(sequence_lengths) * len(stability_strategies)
run_count = 1
for arch_name, model_func in architectures.items():
    for act in activation_functions:
        for opt in optimizers:
            for stab in stability_strategies:
                acc_values = []
                f1_values = []
                for seq_len in sequence_lengths:
                    print("-------------------------------------------")
                    print(f"Run parameters of the number: {run_count}/{total_runs}: {arch_name} | act={act} | opt={opt} | seq={seq_len} | clip={stab}")
                    print("-------------------------------------------")
                    start_time = time.time()

                    # Train model and get the results
                    try:
                        acc, f1, epoch_losses, epoch_times, avg_epoch_time = model_func(
                            file_path=file_path,
                            activation_function=act,
                            optimizer_function=opt,
                            sequence_length=seq_len,
                            stability_strategy=stab,
                            epochs_number=epochs_number
                        )
                        acc_values.append(acc)
                        f1_values.append(f1)

                        # Graph of Training Loss vs. Epochs
                        plt.figure(figsize=(9, 6))
                        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linewidth=2, color="royalblue")
                        plt.title(
                            f"Training Loss vs. Epochs\n"
                            f"Model: {arch_name}\n"
                            f"activation: {act}\n"
                            f"optimizer: {opt}\n"
                            f"Sequence={seq_len}\n"
                            f"Stability={stab}"
                        )
                        plt.xlabel("Epoch")
                        plt.ylabel("Training Loss")
                        plt.grid(True)
                        plt.tight_layout()
                        loss_plot_path = os.path.join(
                            plots_dir, f"{arch_name.replace(' ', '_')}_Loss_Act-{act}_Opt-{opt}_Seq-{seq_len}_Clip-{stab}.png"
                        )
                        plt.savefig(loss_plot_path)
                        plt.close()
                        total_time = time.time() - start_time
                        results.append({
                            "Architecture": arch_name,
                            "Activation": act,
                            "Optimizer": opt,
                            "Sequence_Length": seq_len,
                            "Gradient_Clip": "Yes" if stab == "clip" else "No",
                            "Accuracy": round(acc, 4),
                            "F1_Score": round(f1, 4),
                            "Avg_Epoch_Time(s)": round(avg_epoch_time, 2),
                            "Total_Train_Time(s)": round(total_time, 2)
                        })
                    except Exception as e:
                        print(f"Error in {arch_name} ({act}, {opt}, Seq={seq_len}, Clip={stab}): {e}")
                    run_count += 1

                # Graph of Accuracy & F1 vs. Sequence Length
                plt.figure(figsize=(9, 6))
                plt.plot(sequence_lengths, acc_values, marker='o', linewidth=2, label="Accuracy", color="royalblue")
                plt.plot(sequence_lengths, f1_values, marker='s', linestyle='--', linewidth=2, label="F1 Score", color="darkorange")
                plt.title(
                    f"Accuracy & F1 vs. Sequence Length\n"
                    f"Model: {arch_name}\n"
                    f"activation: {act}\n"
                    f"optimizer: {opt}\n"
                    f"Stability={stab}"
                )
                plt.xlabel("Sequence Length")
                plt.ylabel("Score")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                acc_plot_path = os.path.join(
                    plots_dir, f"{arch_name.replace(' ', '_')}_AccF1_Act-{act}_Opt-{opt}_Clip-{stab}.png"
                )
                plt.savefig(acc_plot_path)
                plt.close()

# Save the results to a CSV file
results_df = pd.DataFrame(results)
csv_path = os.path.join(results_dir, "evaluation_summary.csv")
results_df.to_csv(csv_path, index=False)

print("\nComplete.")
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

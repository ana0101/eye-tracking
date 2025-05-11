import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    """
    A simple feedforward neural network for regression tasks.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.7)
        self.layer2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.7)
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.7)
        self.output_layer = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x
    

def evaluate_regression_model(model, test_loader, criterion):
    model.eval()
    test_pred = []
    test_true = []
    with torch.no_grad():
        for features, true_trt in test_loader:
            outputs = model(features)
            outputs = outputs.squeeze(-1)
            test_pred.append(outputs.cpu().numpy())
            test_true.append(true_trt.cpu().numpy())
        test_pred = np.concatenate(test_pred, axis=0)
        test_true = np.concatenate(test_true, axis=0)
        loss_test = criterion(torch.tensor(test_pred), torch.tensor(test_true)).item()
        pearson_test = pearsonr(test_true, test_pred)[0]
        spearman_test = spearmanr(test_true, test_pred)[0]
        r2_test = r2_score(test_true, test_pred)
    return loss_test, r2_test, pearson_test, spearman_test


def train_regression_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, save_path='best_nn_model.pth', num_epochs=60):
    best_loss = float('inf')
    model.train()
    for epoch in range(num_epochs):
        for i, (features, true_trt) in enumerate(train_loader):
            train_pred = model(features)
            train_pred = train_pred.squeeze(-1)
            loss = criterion(train_pred, true_trt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        loss_dev, r2_dev, pearson_dev, spearman_dev = evaluate_regression_model(model, validation_loader, criterion)
        if loss_dev < best_loss:
            best_loss = loss_dev
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0:
            print(f'Validation loss: {loss_dev}')
            print(f'R^2 on validation set: {r2_dev}')
            print(f'Pearson correlation on validation set: {pearson_dev}')
            print(f'Spearman correlation on validation set: {spearman_dev}')

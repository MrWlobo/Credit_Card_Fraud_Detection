import logging
import pandas as pd
from zenml import step
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class BasicCreditCardFraudModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.linear_relu_stack(x)

    def train_epoch(self, dataloader, optimizer):
        self.train()
        total_loss = 0.0
        for X, y in dataloader:
            pred = self(X).squeeze(1)
            loss = self.loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.eval()
        num_batches = len(dataloader)
        test_loss, correct = 0.0, 0.0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X).squeeze(1)
                test_loss += self.loss_fn(pred, y).item()
                
                predicted_labels = (pred > 0).float()
                correct += (predicted_labels == y).sum().item()

        avg_loss = test_loss / num_batches
        accuracy = correct / len(dataloader.dataset)
        return avg_loss, accuracy


@step
def train_and_validate_model(
    train_dataset: Dataset,
    val_dataset: Dataset, 
    feature_count: int,
    epochs: int = 10,
    batch_size: int = 64,
) -> None:
    logging.info("Creating data loaders")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    

    model = BasicCreditCardFraudModel(input_size=feature_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logging.info("Starting model training")
    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader, optimizer)
        val_loss, val_acc = model.evaluate(val_loader)
        logging.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

@step
def evaluate_model(
    test_dataset: Dataset,
    model: nn.Module,
    loss_fn: nn.Loss,
    batch_size: int = 64,
):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_acc = model.evaluate(test_loader, loss_fn)
    logging.info(f"Final Test Evaluation | Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%")

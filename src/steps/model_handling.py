import logging
from typing import Annotated
from sklearn.metrics import precision_score, recall_score, average_precision_score
from zenml import step
import numpy as np
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

        all_targets = []
        all_probs = []

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X).squeeze(1)
                test_loss += self.loss_fn(pred, y).item()
                
                predicted_labels = (pred > 0).float()
                correct += (predicted_labels == y).sum().item()

        avg_loss = test_loss / num_batches
        all_probs_np = np.array(all_probs)
        all_targets_np = np.array(all_targets)
        all_preds_np = (all_probs_np >= 0.5).astype(int)

        accuracy = correct / len(dataloader.dataset)
        precision = precision_score(all_targets_np, all_preds_np, zero_division=0)
        recall = recall_score(all_targets_np, all_preds_np, zero_division=0)
        auprc = average_precision_score(all_targets_np, all_probs_np)
        return avg_loss, accuracy, precision, recall, auprc


@step
def train_and_validate_model(
    train_dataset: Dataset,
    val_dataset: Dataset, 
    feature_count: int,
    epochs: int = 10,
    batch_size: int = 32,
) -> nn.Module:
    logging.info("Creating data loaders")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BasicCreditCardFraudModel(input_size=feature_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logging.info("Starting model training")
    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader, optimizer)
        val_loss, val_acc, val_pre, val_rec, val_auprc = model.evaluate(val_loader)
        logging.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {val_acc*100:.2f}% | Precision: {val_pre:.2f} | Recall: {val_rec:.2f} | AUPRC: {val_auprc:.2f}"
        )

    return model

@step
def evaluate_model(
    test_dataset: Dataset,
    model: nn.Module,
    batch_size: int = 32,
) -> Annotated[float, "test_accuracy"]:
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_acc, test_pre, test_rec, test_auprc = model.evaluate(test_loader)
    logging.info(f"Final Test Evaluation | Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}% | Precision: {test_pre:.2f} | Recall: {test_rec:.2f} | AUPRC: {test_auprc:.2f}")
    return test_acc, test_pre, test_rec, test_auprc

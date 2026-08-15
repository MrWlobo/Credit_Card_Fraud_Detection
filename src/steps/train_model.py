import logging
import pandas as pd
from zenml import step
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class BasicCreditCardFraudModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


@step
def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset, 
    test_dataset: Dataset, 
    scaler: StandardScaler,
) -> None:
        logging.info("Creating data loaders")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
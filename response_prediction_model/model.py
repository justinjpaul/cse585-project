import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from load_data import load_dataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import BertModel

MODEL_NAME = "google-bert/bert-base-uncased"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(585)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps")

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3

dataset = load_dataset(Path("data/chatbot_conversations.json"), MODEL_NAME, DEVICE)

# Split into train (80%) and test (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


class BertSeqLengthPredictionModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)

        # Fix the weights of the pretrained model
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation

        output = self.cls(logits)
        output = self.fc1(output)
        # output = self.relu(self.cls(logits))
        # output = self.relu(self.fc1(output))
        output = self.fc2(output).squeeze(-1)
        return output


def main():
    # Fine-tuning procedure (Example)
    model = BertSeqLengthPredictionModel(
        config=BertModel.config_class(), model_name=MODEL_NAME, hidden_dim=128
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    torch.save(model.state_dict(), "response_prediction_model/models/epoch0.pt")
    # Training Loop (Example)
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].squeeze(-1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        torch.save(
            model.state_dict(), f"response_prediction_model/models/epoch{epoch+1}.pt"
        )


if __name__ == "__main__":
    main()

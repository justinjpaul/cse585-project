from pathlib import Path

import torch.nn as nn
import wandb
from load_data import load_dataset
from torch.utils.data import DataLoader, random_split
from transformers import BertModel, Trainer, TrainingArguments
from utils import DEVICE, MODEL_NAME, set_seed

set_seed(585)


# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 3
TRAIN_TEST_SPLIT = 0.8  # Split into train (80%) and test (20%)

dataset = load_dataset(Path("data/chatbot_conversations.json"), MODEL_NAME, DEVICE)
train_size = int(TRAIN_TEST_SPLIT * len(dataset))
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
        for param in self.bert.parameters():
            param.requires_grad = False

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
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.fc2(output).squeeze(-1)
        return output


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss


def main():
    wandb.init(
        # set the wandb project where this run will be logged
        project="cse585bert",
        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "Bert Fine Tune",
            "dataset": "chatbot_conversations",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
    )

    model = BertSeqLengthPredictionModel(
        config=BertModel.config_class(), model_name=MODEL_NAME, hidden_dim=128
    ).to(DEVICE)

    training_args = TrainingArguments(
        output_dir="response_prediction_model/checkpoints",  # Checkpoint directory
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        logging_strategy="epoch",  # Log metrics at the end of each epoch
        save_strategy="epoch",  # Save model checkpoints at each epoch
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        report_to="wandb",  # Log to wandb
    )
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()

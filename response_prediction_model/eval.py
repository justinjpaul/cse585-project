import torch
from model import BertSeqLengthPredictionModel, test_loader
from tqdm import tqdm
from transformers import BertModel

DEVICE = torch.device("mps")
MODEL_NAME = "google-bert/bert-base-uncased"
model = BertSeqLengthPredictionModel(
    config=BertModel.config_class(), model_name=MODEL_NAME, hidden_dim=128
).to(DEVICE)

state = torch.load("response_prediction_model/models1/epoch3.pt")
model.load_state_dict(state)
# criterion = nn.MSELoss()

for ind, batch in enumerate(tqdm(test_loader)):
    # Start Editing
    with torch.no_grad():
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        print(labels)
        print(output)
        if ind == 5:
            break

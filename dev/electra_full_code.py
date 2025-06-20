import os
import gc
import json
import tqdm
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import ElectraTokenizer, ElectraForSequenceClassification, AdamW
from torch.cuda.amp import autocast, GradScaler

# 0. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./electra", exist_ok=True)

# 1. Load IMDb dataset (subset)
dataset = load_dataset("imdb")
train_data = dataset['train'].shuffle(seed=42).select(range(2000))
test_data = dataset['test'].select(range(1000))  # no need to shuffle test set

# 2. Tokenizer
tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 3. DataLoaders
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

# 4. Model and Optimizer
model = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator", num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()

# 5. Training Loop
train_losses = []
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['labels'] = batch.pop('label')

        optimizer.zero_grad()

        with autocast():  # mixed precision training
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# 6. Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['labels'] = batch.pop('label')

        with autocast():
            outputs = model(**batch)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())
        torch.cuda.empty_cache()

# 7. Metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

print("\nEvaluation Metrics:")
for k, v in metrics.items():
    print(f"{k.title()}: {v:.4f}")

# 8. Save metrics + loss history
with open("./electra/metrics.json", "w") as f:
    json.dump(metrics, f)

with open("./electra/loss_history.json", "w") as f:
    json.dump(train_losses, f)

# 9. Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("./electra/training_loss.png")
plt.show()

# 10. Cleanup
gc.collect()
torch.cuda.empty_cache()

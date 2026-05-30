import torch
import glob
import random
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets.xray_dataset import XrayDataset
from models.resnet import get_model
from utils.collate import custom_collate

BATCH_SIZE = 16
EPOCHS = 15
LR = 5e-5
VAL_SPLIT = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"

image_paths = glob.glob("/content/drive/MyDrive/NIH/images/*.png")
random.shuffle(image_paths)

dataset = XrayDataset(image_paths)

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

model = get_model().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def compute_accuracy(outputs, labels):
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    return (preds == labels).float().mean()

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()
    train_loss, train_acc = 0, 0

    for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += compute_accuracy(outputs, labels).item()

    model.eval()
    val_loss, val_acc = 0, 0

    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_acc += compute_accuracy(outputs, labels).item()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("🔥 Saved best model")

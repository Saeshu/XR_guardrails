import torch
from torch.utils.data import DataLoader
from datasets.xray_dataset import XrayDataset
from models.resnet import get_model
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"

# load images
image_paths = glob.glob("data/images/*.png")

dataset = XrayDataset(image_paths)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = get_model().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

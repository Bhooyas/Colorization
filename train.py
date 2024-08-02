import torch
from config import *
from utils import *
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from model import *
import torch.optim as optim
from tqdm import tqdm
from safetensors.torch import save_model

seed = 555
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

transform = transforms.Compose([transforms.ToTensor()])
dataset = CoilDataset(f"{data_dir}/subset", transform=transform)
train_size = round(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"{len(train_dataset) = }")
print(f"{len(test_dataset) = }")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = Unet(1, 3).to(device)
loss_fn = PerceptualLoss(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
model_size = sum(p.numel() for p in model.parameters())
print(f"{model_size = :,}")

for epoch in range(epochs):
    pbar = tqdm(train_loader)
    pbar.set_description(f"Epoch {epoch+1}/{epochs}")
    for inputs, outputs in pbar:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        predicted = model(inputs)
        loss = loss_fn(predicted, outputs)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"Loss": loss.item()})
    with torch.no_grad():
        model.eval()
        losses = []
        for inputs, outputs in test_loader:
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            predicted = model(inputs)
            loss = loss_fn(predicted, outputs)
            losses.append(loss.item())
        print(f"Validation loss: {sum(losses)/len(losses)}")

save_model(model, model_weights)

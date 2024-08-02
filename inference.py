import torch
from config import *
import torchvision.transforms as transforms
from model import *
from safetensors.torch import load_model
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from random import shuffle

model = Unet(1, 3)
model.eval()
load_model(model, model_weights)

transform = transforms.Compose([transforms.ToTensor()])
idx = 0
test_images = glob(f"{data_dir}/subset/images/*__0.jpg")
shuffle(test_images)
test_images = test_images[:num_samples]
fig, axs = plt.subplots(3, len(test_images))
for test_image in test_images:
    img = Image.open(test_image)
    axs[0, idx].imshow(img)
    axs[0, idx].axis("off")
    img = img.convert("L")
    axs[1, idx].imshow(img, cmap="gray")
    axs[1, idx].axis("off")
    with torch.no_grad():
        img_tensor = transform(img).unsqueeze(0)
        pred = model(img_tensor).squeeze(0).numpy()

    axs[2, idx].imshow(pred.transpose(1, 2, 0))
    axs[2, idx].axis("off")
    idx += 1

plt.show()

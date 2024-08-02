from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

class CoilDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        super(CoilDataset, self).__init__()
        self.image_paths = glob(f"{data_dir}/images/*.jpg")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        output = Image.open(self.image_paths[idx])
        input = output.convert("L")
        if self.transform:
            output = self.transform(output)
            input = self.transform(input)
        return input, output

class PerceptualLoss(nn.Module):

    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.vgg = nn.Sequential(*list(vgg.children())[:16]).to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    def forward(self, pred, output):
        # pred = (pred - self.mean) / self.std
        # output = (output - self.mean) / self.std
        pred_feat = self.vgg(pred)
        output_feat = self.vgg(output)
        loss = nn.functional.mse_loss(pred_feat, output_feat)
        return loss

if __name__ == "__main__":
    import torchvision.transforms as transforms
    dataset = CoilDataset(data_dir="./data/subset", transform=transforms.Compose([transforms.ToTensor()]))
    print(len(dataset))
    input, output = dataset[0]
    print(input.shape, output.shape)

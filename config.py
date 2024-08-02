import torch

data_dir = "./data" # The directory of training images
subset_size = 20 # The number of images to select for subset

# Training Parameters
batch_size = 32 # Batch size
lr = 0.001 # Learning rate
epochs = 20 # Epochs
model_weights = "unet.safetensors" # Path to save the model weigths
device = "cuda" if torch.cuda.is_available() else "cpu" # Device to train on

# Inference
num_samples = 5 # Number of samples for inferencing

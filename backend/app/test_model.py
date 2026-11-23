
import torch
from app.model import build_model

# Use CPU for inference
device = torch.device("cpu")

# Create the VAE architecture
model = build_model()

# Load the trained checkpoint
checkpoint = torch.load(
    "models/celeba_vae.pth",
    map_location=device,
    weights_only=False
)

# Load trained weights into the model
model.load_state_dict(checkpoint["model_state_dict"])

# Set model to evaluation mode
model.eval()

print("Model loaded successfully!")
print("Training epochs:", checkpoint["epochs"])

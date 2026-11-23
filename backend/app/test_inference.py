
import torch
from PIL import Image
from torchvision import transforms

from app.model import build_model


# 1. Use CPU
device = torch.device("cpu")


# 2. Load the model
model = build_model()

checkpoint = torch.load(
    "models/celeba_vae.pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# 3. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


# 4. Load one test image
image = Image.open("app/test_face.jpg").convert("RGB")

image_tensor = transform(image).unsqueeze(0)


# 5. Load smiling direction vector
smiling_vector = torch.load(
    "models/smiling_vector.pt",
    map_location=device,
    weights_only=True
)


# 6. Encode the image
with torch.no_grad():
    z_mean, z_log_var, z = model.encoder(image_tensor)


# 7. Perform smiling arithmetic
alpha = 1.0

edited_z = z_mean + alpha * smiling_vector


# 8. Generate edited image
with torch.no_grad():
    generated_image = model.decoder(edited_z)


# 9. Verify output
print("Input shape:", image_tensor.shape)
print("Latent shape:", z_mean.shape)
print("Generated image shape:", generated_image.shape)

print("Inference successful!")

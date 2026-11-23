
from PIL import Image
from torchvision import transforms
from io import BytesIO


# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


def preprocess_image(image_bytes: bytes):
    # Convert uploaded image bytes into a PIL image
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Apply resizing and convert image into a PyTorch tensor
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

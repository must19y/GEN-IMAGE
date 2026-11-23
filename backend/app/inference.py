
import torch
from pathlib import Path
from torchvision.transforms.functional import to_pil_image

from app.model import build_model
from app.preprocessing import preprocess_image


# Paths to our saved model files
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "celeba_vae.pth"
VECTOR_PATH = BASE_DIR / "models" / "smiling_vector.pt"


def load_inference_assets():

    # We are using CPU for inference
    device = torch.device("cpu")

    # Build the VAE architecture
    model = build_model()

    # Load trained model weights
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load smiling direction vector
    smiling_vector = torch.load(
        VECTOR_PATH,
        map_location=device,
        weights_only=True
    )

    return model, smiling_vector, device


def generate_face(image_bytes, alpha, model, smiling_vector, device):

    # 1. Preprocess uploaded image
    image_tensor = preprocess_image(image_bytes)

    # 2. Move input tensor to the selected device
    image_tensor = image_tensor.to(device)

    # 3. Encode image into latent representation
    with torch.no_grad():
        z_mean, z_log_var, z = model.encoder(image_tensor)

        # 4. Apply smiling direction
        edited_z = z_mean + alpha * smiling_vector

        # 5. Decode modified latent representation
        generated_image = model.decoder(edited_z)

    # 6. Convert output tensor into a PIL image
    generated_image = generated_image.squeeze(0).cpu()
    output_image = to_pil_image(generated_image)

    return output_image

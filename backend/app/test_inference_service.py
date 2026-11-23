
from app.inference import load_inference_assets, generate_face


# Load model and smiling vector
model, smiling_vector, device = load_inference_assets()

# Read test image
with open("app/test_face.jpg", "rb") as image_file:
    image_bytes = image_file.read()

# Generate edited face
output_image = generate_face(
    image_bytes=image_bytes,
    alpha=1.0,
    model=model,
    smiling_vector=smiling_vector,
    device=device
)

# Save generated image
output_image.save("app/generated_face.png")

print("Inference successful!")
print("Generated image saved!")
print("Output size:", output_image.size)

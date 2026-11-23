
from app.preprocessing import preprocess_image

# Read our existing test image
with open("app/test_face.jpg", "rb") as image_file:
    image_bytes = image_file.read()

# Preprocess it
image_tensor = preprocess_image(image_bytes)

print("Preprocessed image shape:", image_tensor.shape)
print("Minimum pixel value:", image_tensor.min().item())
print("Maximum pixel value:", image_tensor.max().item())


from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from io import BytesIO

from app.inference import load_inference_assets, generate_face


# Store our loaded model and smiling vector
inference_assets = {}


# Load model when application starts
@asynccontextmanager
async def lifespan(app: FastAPI):

    model, smiling_vector, device = load_inference_assets()

    inference_assets["model"] = model
    inference_assets["smiling_vector"] = smiling_vector
    inference_assets["device"] = device

    print("VAE model loaded successfully!")

    yield

    # Application shutdown
    inference_assets.clear()


# Create FastAPI application
app = FastAPI(
    title="CelebA Face Editor API",
    description="Edit facial expressions using a trained VAE.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://gen-image-blush.vercel.app",
        "https://gen-image-git-main-gen-image.vercel.app",
        "https://gen-image-k6l6tbdfe-gen-image.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "model_loaded": "model" in inference_assets
    }


# Image editing endpoint
@app.post("/edit")
async def edit_face(
    image: UploadFile = File(...),
    alpha: float = Form(1.0)
):

    # Validate uploaded file
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    # Read uploaded image bytes
    image_bytes = await image.read()
##noticethatevenifweareusingthreadweareusingawaitontheasynconeright
    try:
        # Run synchronous PyTorch inference in a worker thread
        output_image = await run_in_threadpool(
            generate_face,
            image_bytes,
            alpha,
            inference_assets["model"],
            inference_assets["smiling_vector"],
            inference_assets["device"]
        )

        # Convert generated PIL image into PNG bytes
        image_buffer = BytesIO()
        output_image.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        # Return generated image
        return StreamingResponse(
            image_buffer,
            media_type="image/png"
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Image processing failed: {str(error)}"
        )

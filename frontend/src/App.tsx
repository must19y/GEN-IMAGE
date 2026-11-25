
import { useEffect, useState } from "react";

function App() {
  // Selected image from the user's computer
  const [image, setImage] = useState<File | null>(null);

  // Smile intensity
  const [alpha, setAlpha] = useState<number>(1);

  // Image previews
  const [originalPreview, setOriginalPreview] = useState<string | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);

  // Application status
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Release temporary browser image URLs when they are replaced or removed
  useEffect(() => {
    return () => {
      if (originalPreview) {
        URL.revokeObjectURL(originalPreview);
      }
    };
  }, [originalPreview]);

  useEffect(() => {
    return () => {
      if (generatedImage) {
        URL.revokeObjectURL(generatedImage);
      }
    };
  }, [generatedImage]);

  // Handle image selection
  const handleImageChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const selectedImage = event.target.files?.[0];

    if (!selectedImage) return;

    if (!selectedImage.type.startsWith("image/")) {
      setError("Please select a valid image.");
      return;
    }

    setImage(selectedImage);
    setOriginalPreview(URL.createObjectURL(selectedImage));

    // Clear previous results
    setGeneratedImage(null);
    setError(null);
  };

  // Send image and alpha to FastAPI
  const handleGenerate = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);
    setGeneratedImage(null);

    try {
      // Create multipart/form-data request
      const formData = new FormData();

      formData.append("image", image);
      formData.append("alpha", alpha.toString());

      // Call FastAPI through the Vite proxy
      const response = await fetch(`${import.meta.env.VITE_API_URL || "/api"}/edit`, {//for production vite to communicate with render
        method: "POST",
        body: formData,
      });

      // Handle unsuccessful responses
      if (!response.ok) {
        const errorMessage = await response.text();
        throw new Error(errorMessage || "Image generation failed.");
      }

      // Convert returned PNG bytes into a browser Blob
      const imageBlob = await response.blob();

      // Create a temporary URL for displaying the image
      const imageUrl = URL.createObjectURL(imageBlob);

      setGeneratedImage(imageUrl);

    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong."
      );

    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white px-5 py-10">

      <div className="max-w-5xl mx-auto">

        {/* Header */}
        <header className="text-center mb-10">
          <h1 className="text-4xl font-bold tracking-tight">
            AI Face Editor
          </h1>

          <p className="text-slate-400 mt-3">
            Modify facial expressions using a Variational Autoencoder
          </p>
        </header>

        {/* Main application */}
        <main className="bg-slate-900 border border-slate-800 rounded-2xl p-6 md:p-8">

          {/* Image upload */}
          <section className="mb-8">

            <label
              htmlFor="image-upload"
              className="block text-sm font-semibold mb-3"
            >
              Upload a face image
            </label>

            <input
              id="image-upload"
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="block w-full text-sm text-slate-300
                file:mr-4 file:py-2 file:px-4
                file:rounded-lg file:border-0
                file:bg-blue-600 file:text-white
                hover:file:bg-blue-700
                cursor-pointer"
            />

          </section>

          {/* Alpha slider */}
          <section className="mb-8">

            <div className="flex justify-between items-center mb-3">

              <label
                htmlFor="alpha"
                className="text-sm font-semibold"
              >
                Smile intensity
              </label>

              <span className="bg-blue-600 px-3 py-1 rounded-full text-sm">
                {alpha.toFixed(1)}
              </span>

            </div>

            <input
              id="alpha"
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={alpha}
              onChange={(event) => setAlpha(Number(event.target.value))}
              className="w-full accent-blue-500 cursor-pointer"
            />

            <div className="flex justify-between text-xs text-slate-400 mt-2">
              <span>Original (0)</span>
              <span>Stronger smile (2)</span>
            </div>

          </section>

          {/* Generate button */}
          <button
            onClick={handleGenerate}
            disabled={!image || loading}
            className="w-full bg-blue-600 hover:bg-blue-700
              disabled:bg-slate-700 disabled:cursor-not-allowed
              text-white font-semibold py-3 rounded-lg
              transition-colors"
          >
            {loading ? "Generating image..." : "Generate edited face"}
          </button>

          {/* Loading indicator */}
          {loading && (
            <p className="text-center text-blue-400 mt-4 text-sm">
              Please wait while the VAE processes your image...
            </p>
          )}

          {/* Error message */}
          {error && (
            <p className="text-red-400 text-center mt-4 text-sm">
              {error}
            </p>
          )}

          {/* Image previews */}
          <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-10">

            {/* Original image */}
            <div className="bg-slate-800 rounded-xl p-5 text-center">

              <h2 className="font-semibold mb-4">
                Original Image
              </h2>

              {originalPreview ? (
                <img
                  src={originalPreview}
                  alt="Original face"
                  className="w-full max-w-sm mx-auto rounded-lg"
                />
              ) : (
                <div className="h-64 flex items-center justify-center text-slate-500">
                  Select an image to preview
                </div>
              )}

            </div>

            {/* Generated image */}
            <div className="bg-slate-800 rounded-xl p-5 text-center">

              <h2 className="font-semibold mb-4">
                Generated Image
              </h2>

              {generatedImage ? (
                <img
                  src={generatedImage}
                  alt="Generated face"
                  className="w-full max-w-sm mx-auto rounded-lg"
                />
              ) : (
                <div className="h-64 flex items-center justify-center text-slate-500">
                  Your edited image will appear here
                </div>
              )}

            </div>

          </section>

        </main>

        <footer className="text-center text-slate-500 text-xs mt-8">
          Powered by PyTorch, FastAPI and React
        </footer>

      </div>

    </div>
  );
}

export default App;

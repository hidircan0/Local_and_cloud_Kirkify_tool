import os
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

# 1. Set Dynamic Directory Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
# Targeting the v2 safetensors file in the folder
# Repo root is two levels up from kirkifiers/kirkify_local/
lora_path = os.path.abspath(os.path.join(current_dir, "..", "..", "charliekirk-model", "Charlie_Kirk_🕊️_v2-Qwen_Image.safetensors"))
image_path = os.path.join(current_dir, "foto.jpg") # Enter the name of the image you want to process here

print("Setting up the image generation pipeline on CPU...")

# Pulling the base model specified in the README
base_model = "Qwen/Qwen-Image"

try:
    # 1. Load the Base Model in Image-to-Image format
    pipe = AutoPipelineForImage2Image.from_pretrained(
        base_model,
        torch_dtype=torch.float16,  # Compresses files to 16-bit in RAM while reading
        low_cpu_mem_usage=True,     # Optimizes memory usage when loading the model into RAM
        use_safetensors=True
    )
    pipe.to("cpu")

    # 2. Inject the LoRA Weights into the Model
    print("Applying the LoRA weights...")
    pipe.load_lora_weights(lora_path)

except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# 3. Read and Prepare the Reference Photo
if not os.path.exists(image_path):
    print(f"Error: '{image_path}' not found.")
    exit()

init_image = Image.open(image_path).convert("RGB")
# Resizing the image to 512x512 to prevent processing from taking hours on CPU
init_image = init_image.resize((512, 512))

# 4. Run the Model (Inference)
print("Model is redrawing based on the original photo...")
print("Since you are on the CPU, this process may take quite a while. You can leave it running in the background.")

prompt = "Ch4rlie K!rk face, photorealistic, meme realism, detailed"
negative_prompt = "bad quality, blurry, deformed, distorted face"

result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=0.65, 
        guidance_scale=7.5
).images[0]

# 5. Save the Output
output_path = os.path.join(current_dir, "kirkify_sonuc.png")
result.save(output_path)

print("\n" + "="*40)
print(f"PROCESS COMPLETE! New photo saved to: {output_path}")
print("="*40)
import os
import replicate
from dotenv import load_dotenv
import requests

load_dotenv()
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

source_path = "foto.jpg"
target_path = "charlie_kirk.jpg"

if not os.path.exists(source_path) or not os.path.exists(target_path):
    print(f"Error: Either 'foto.jpg' or 'charlie_kirk.jpg' is missing!")
    exit()

print("Connecting to Replicate, the server is loading the model (this may take a moment)...")

try:
    model = replicate.models.get("codeplugtech/face-swap")
    version = model.versions.list()[0]

    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    
    output = client.run(
        f"codeplugtech/face-swap:{version.id}",
        input={
            "input_image": open(source_path, "rb"),
            "swap_image": open(target_path, "rb")
        }
    )
    
    image_url = output.url if hasattr(output, "url") else str(output)
    
    img_data = requests.get(image_url).content
    
    with open("api_sonuc.png", "wb") as f:
        f.write(img_data)

    print("\n" + "="*50)
    print("SUCCESS! Face swap completed: 'api_sonuc.png'.")
    print("="*50)

except Exception as e:
    print(f"An error occurred: {e}")
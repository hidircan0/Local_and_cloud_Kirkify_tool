from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Ana dizindeki kirkify klasörünün tam yolunu alıyoruz
model_yolu = os.path.expanduser("~/kirkify")

# Modeli lokal klasörden yüklüyoruz
tokenizer = AutoTokenizer.from_pretrained(model_yolu)
model = AutoModelForCausalLM.from_pretrained(
    model_yolu, 
    device_map="auto", 
    torch_dtype="auto" # GPU belleğini optimize etmek için
)

print("Model başarıyla lokalden yüklendi!")

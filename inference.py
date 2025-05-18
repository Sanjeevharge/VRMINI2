import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel
import torch
import os
import gdown
import zipfile

# ========================
# STEP 1: Download LoRA Checkpoint from Google Drive
# ========================
file_id = "1xJGWn-O5w8VMh5wem5Pv4Xgop5vLLCTt"
output_path = "best-checkpoint.zip"

if not os.path.exists("blip2-lora"):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall("blip2-lora")

# ========================
# STEP 2: Main Inference Logic
# ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    # Load processor and base model
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    base_model     = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float32,
    device_map="auto"
    )

    # Load LoRA adapter
    lora_path = "blip2-lora"
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()  # No need to manually .to(device) because of device_map="auto"

    generated_answers = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, text=question, return_tensors="pt").to(base_model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=30)
                answer = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        except Exception as e:
            answer = "error"

        answer = str(answer).strip().lower()
        generated_answers.append(answer)
        print(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()

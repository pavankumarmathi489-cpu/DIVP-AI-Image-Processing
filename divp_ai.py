import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"

print("🤖 Loading local AI image-captioning model...")
try:
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    caption_model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    caption_model.eval()
except Exception as e:
    print("\n❌ Could not load BLIP.")
    print("On the first run, check your internet connection.")
    print("Technical error:", e)
    raise SystemExit(1)

print("✅ Local AI model loaded successfully!\n")


def generate_caption(image):
    if image is None:
        return "Unable to analyze image."

    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(rgb)
    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.inference_mode():
        output = caption_model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    if not caption:
        return "No description generated."

    return caption[0].upper() + caption[1:]


def save_histogram(image, title, path):
    plt.figure(figsize=(8, 5))
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_and_show_results(original, equalized, original_caption,
                           equalized_caption, filename, output_folder):

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    axes[0].imshow(original_rgb)
    axes[0].axis("off")
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")

    axes[1].imshow(equalized, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title(
        "Equalized Image (Processed)",
        fontsize=14,
        fontweight="bold"
    )

    fig.suptitle(
        "Digital Image Processing + Local AI Analysis\n" + filename,
        fontsize=16,
        fontweight="bold"
    )

    fig.text(
        0.25, 0.08,
        "AI Description:\n" + original_caption,
        ha="center", va="center", fontsize=11, wrap=True
    )

    fig.text(
        0.75, 0.08,
        "AI Description:\n" + equalized_caption,
        ha="center", va="center", fontsize=11, wrap=True
    )

    plt.tight_layout(rect=[0, 0.18, 1, 0.92])

    combined_path = os.path.join(
        output_folder, f"{filename}_AI_analysis.png"
    )

    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return combined_path


def process_image(path, output_folder):

    print("\n" + "=" * 65)
    print("🔄 Processing:", os.path.basename(path))
    print("=" * 65)

    image = cv2.imread(path)

    if image is None:
        print("❌ Could not load image. Skipping.")
        return

    filename = os.path.splitext(os.path.basename(path))[0]

    print("⚙️ Converting to grayscale...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("⚙️ Applying histogram equalization...")
    equalized = cv2.equalizeHist(gray)

    print("🤖 Analyzing original image...")
    original_caption = generate_caption(image)
    print("   Original:", original_caption)

    print("🤖 Analyzing processed image...")
    equalized_caption = generate_caption(equalized)
    print("   Equalized:", equalized_caption)

    gray_path = os.path.join(output_folder, f"{filename}_grayscale.jpg")
    eq_path = os.path.join(output_folder, f"{filename}_equalized.jpg")

    cv2.imwrite(gray_path, gray)
    cv2.imwrite(eq_path, equalized)

    save_histogram(
        gray,
        f"Grayscale Histogram - {filename}",
        os.path.join(output_folder, f"{filename}_histogram.png")
    )

    save_histogram(
        equalized,
        f"Equalized Histogram - {filename}",
        os.path.join(output_folder, f"{filename}_equalized_histogram.png")
    )

    text_path = os.path.join(
        output_folder, f"{filename}_AI_description.txt"
    )

    with open(text_path, "w", encoding="utf-8") as f:
        f.write("DIGITAL IMAGE PROCESSING + AI ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write("Original Image AI Description:\n")
        f.write(original_caption + "\n\n")
        f.write("Processed/Equalized Image AI Description:\n")
        f.write(equalized_caption + "\n")

    combined_path = save_and_show_results(
        image,
        equalized,
        original_caption,
        equalized_caption,
        filename,
        output_folder
    )

    print("✅ Grayscale saved:", gray_path)
    print("✅ Equalized saved:", eq_path)
    print("✅ AI description saved:", text_path)
    print("✅ Combined result saved:", combined_path)
    print("✅ Processing completed.")


def main():

    root = tk.Tk()
    root.withdraw()

    print("🖼️ Select one or more image files...")

    image_paths = filedialog.askopenfilenames(
        title="Select Image Files",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")
        ]
    )

    if not image_paths:
        print("❌ No images selected. Exiting.")
        root.destroy()
        return

    print("📂 Select output folder...")

    output_folder = filedialog.askdirectory(
        title="Select Output Folder"
    )

    if not output_folder:
        print("❌ No output folder selected. Exiting.")
        root.destroy()
        return

    root.destroy()

    for index, path in enumerate(image_paths, start=1):
        print(f"\n📌 Image {index}/{len(image_paths)}")

        try:
            process_image(path, output_folder)
        except Exception as e:
            print("❌ Error while processing:", os.path.basename(path))
            print("   ", e)

    print("\n🎉 ALL SELECTED IMAGES PROCESSED!")


if __name__ == "__main__":
    main()
# Digital Image Processing + AI Analysis

A Python-based Digital Image Processing (DIP/DIVP) project that combines traditional image processing with local AI image captioning.

The project:
- Selects one or more image files using a graphical file-selection dialog.
- Converts each image to grayscale.
- Applies histogram equalization to the grayscale image.
- Generates an AI description of both the original and processed images using the BLIP image-captioning model.
- Saves the processed images, histograms, AI descriptions, and a combined comparison image.

## Features

- Multiple image selection
- Grayscale image conversion
- Histogram equalization
- Original-image AI captioning
- Processed-image AI captioning
- Original and equalized histogram generation
- Automatic saving of results
- Combined visual comparison of original and processed images
- Local BLIP model through Hugging Face Transformers

## Technologies Used

- Python
- OpenCV
- Matplotlib
- Pillow
- PyTorch
- Hugging Face Transformers
- Tkinter
- BLIP (`Salesforce/blip-image-captioning-base`)

## Project Structure

```text
DIVP-AI-Image-Processing/
│
├── divp_ai.py
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
```

## Requirements

- Python 3.x
- Internet connection for the first BLIP model download
- A system capable of running PyTorch
- Tkinter (normally included with Python on Windows)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/DIVP-AI-Image-Processing.git
cd DIVP-AI-Image-Processing
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

## Running the Project

Run:

```bash
python divp_ai.py
```

The program will:

1. Ask you to select one or more image files.
2. Ask you to select an output folder.
3. Convert each selected image to grayscale.
4. Apply histogram equalization.
5. Generate an AI description for the original image.
6. Generate an AI description for the equalized image.
7. Save the generated results in the selected output folder.
8. Display a comparison of the original and processed images.

## Output Files

For an input image named `example.jpg`, the program generates files similar to:

```text
example_grayscale.jpg
example_equalized.jpg
example_histogram.png
example_equalized_histogram.png
example_AI_description.txt
example_AI_analysis.png
```

### Output Description

| File | Purpose |
|---|---|
| `*_grayscale.jpg` | Grayscale version of the input image |
| `*_equalized.jpg` | Histogram-equalized image |
| `*_histogram.png` | Histogram of the grayscale image |
| `*_equalized_histogram.png` | Histogram after equalization |
| `*_AI_description.txt` | AI-generated descriptions |
| `*_AI_analysis.png` | Combined visual comparison with AI descriptions |

## AI Model

This project uses:

**Salesforce BLIP Image Captioning Base**

Model name:

```text
Salesforce/blip-image-captioning-base
```

The model is loaded through Hugging Face Transformers.

On the first run, the model may need to be downloaded. Therefore, an internet connection is required during the initial model setup. After the model has been downloaded and cached, subsequent runs can normally use the cached model.

## Image Processing Pipeline

```text
Input Image
     │
     ├──────────────► AI Captioning ──────────► Original Description
     │
     ▼
Grayscale Conversion
     │
     ▼
Histogram Equalization
     │
     ├──────────────► AI Captioning ──────────► Processed Description
     │
     ├──────────────► Histogram
     │
     └──────────────► Equalized Histogram
                         │
                         ▼
                 Combined Analysis
```

## Notes

- Supported image formats include JPG, JPEG, PNG, BMP, TIF and TIFF.
- The project processes the selected images one by one.
- The output folder is selected by the user at runtime.
- The AI descriptions are generated using the BLIP image-captioning model.
- The project is designed as a Digital Image Processing project enhanced with local AI analysis.

## Author

**G. Jagadeesh**

Digital Image Processing + AI Analysis Project

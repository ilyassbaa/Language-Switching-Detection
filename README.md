# Code-Switching Detection: Darija, French, and English

## Project Overview
This project implements a Natural Language Processing (NLP) pipeline designed to detect and classify **Code-Switching** in trilingual Moroccan text. The system automatically identifies segments of **Darija**, **French**, and **English** within mixed-language sentences.

This project was developed as part of a module at **Faculté Polydisciplinaire de Khouribga (FP Khouribga)**.

## Authors
* **Ilyass Bahri**
* **Oussama El Hichami**

## Project Architecture
The system utilizes a fine-tuned **XLM-RoBERTa** model to perform token classification. The pipeline is designed for local processing, ensuring data privacy and utilizing AMD GPU acceleration (via DirectML).



## Key Features
* **Trilingual Detection:** Classifies input tokens into Darija (DA), French (FR), and English (EN) labels.
* **Automated Data Factory:** A custom pipeline that performs cleaning, sanitization, and pseudo-labeling of raw datasets.
* **User Interface:** A real-time web interface built with **Gradio** to visualize token-level language identification.
* **Hardware Optimized:** Leverages local AMD GPU resources for high-performance training and inference.

## Technologies Used
* **Languages:** Python 3.10
* **Frameworks:** PyTorch, Hugging Face Transformers (`XLM-RoBERTa`)
* **Libraries:** Pandas (Data manipulation), Gradio (Web UI), Torch-DirectML (GPU support)
* **Data Format:** CSV-based token classification schema

## Getting Started

### Prerequisites
Ensure you have Python 3.11 installed. Clone this repository and set up your virtual environment:

```bash
# Clone the repository
git clone [https://github.com/ilyassbaa/Language-Switching-Detection.git](https://github.com/ilyassbaa/Language-Switching-Detection.git)
cd Language-Switching-Detection

# Create and activate virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
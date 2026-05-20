# Code-Switching Detection (Darija, French, English)

## Project Overview
This project implements a Natural Language Processing (NLP) pipeline to detect and classify code-switching in trilingual Moroccan text (Darija, French, English). It utilizes a fine-tuned XLM-RoBERTa model and an automated data factory for pseudo-labeling.

## Features
- **Trilingual Detection:** Classifies tokens into Darija (DA), French (FR), and English (EN) tags.
- **Automated Pipeline:** Includes an end-to-end data processing factory that cleans and pseudo-labels data.
- **Robustness:** Handles CSV structural integrity and linguistic fragmentation.

## Technologies Used
- Python 3.10
- Hugging Face Transformers (XLM-RoBERTa)
- PyTorch / GPU Acceleration
- Pandas

## Setup
1. Clone this repository: `git clone [YOUR_REPO_URL]`
2. Install dependencies: `pip install -r requirements.txt` (Run `pip freeze > requirements.txt` to generate this).
3. Ensure your `data/` and `saved_model/` directories are present.
4. Run the pipeline: `python pipeline.py`

## License
MIT License
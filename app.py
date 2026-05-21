import gradio as gr
from transformers import pipeline
import re

# 1. Load the AI you just trained!
print("Loading the custom Code-Switching model...")
try:
    # Aggregation strategy "simple" is perfect for highlighting whole words
    classifier = pipeline("token-classification", model="./saved_model", tokenizer="./saved_model", aggregation_strategy="simple")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 2. The Logic: Process the text and format it for the UI
def detect_code_switching(text):
    results = classifier(text)
    
    # CRITICAL: This mapping must match the labels you used in your training script!
    # If you used B-DA and I-DA in your training, this needs to match.
    id2label = {
        'LABEL_0': 'O', 
        'LABEL_1': 'B-DA', 'LABEL_2': 'I-DA',  # Darija
        'LABEL_3': 'B-FR', 'LABEL_4': 'I-FR',  # French
        'LABEL_5': 'B-EN', 'LABEL_6': 'I-EN'   # English
    }
    
    # Map AI predictions to characters
    char_labels = [None] * len(text)
    for entity in results:
        raw_label = entity.get('entity_group', entity.get('entity', 'LABEL_0'))
        actual_tag = id2label.get(raw_label, 'O')
        
        # Categorize by Language
        if "DA" in actual_tag: label = "Darija"
        elif "FR" in actual_tag: label = "French"
        elif "EN" in actual_tag: label = "English"
        else: label = None
        
        for i in range(entity['start'], entity['end']):
            char_labels[i] = label
            
    # Step: Split sentence into clean, whole words
    tokens = re.split(r'(\s+)', text)
    highlighted_output = []
    current_idx = 0
    
    # Majority Vote Algorithm
    for token in tokens:
        if token.isspace():
            highlighted_output.append((token, None))
            current_idx += len(token)
            continue
            
        word_labels = char_labels[current_idx:current_idx+len(token)]
        valid_labels = [l for l in word_labels if l is not None]
        
        if valid_labels:
            # Winner takes all
            dominant_label = max(set(valid_labels), key=valid_labels.count)
            highlighted_output.append((token, dominant_label))
        else:
            highlighted_output.append((token, None))
            
        current_idx += len(token)
        
    return highlighted_output

# 3. Visual Interface (Optimized for Mobile)
theme = gr.themes.Soft()
with gr.Blocks(theme=theme, css=".gradio-container {max-width: 900px !important; margin: auto !important;}") as app:
    gr.Markdown("# 🇲🇦 Moroccan Code-Switching Detector")
    gr.Markdown("Detecting Darija, French, and English in mixed-text sentences.")
    
    # Using a Column layout ensures the button stacks under the textbox on small screens
    with gr.Column():
        text_input = gr.Textbox(
            label="Enter a mixed sentence", 
            placeholder="E.g., Chof dik la voiture it looks amazing...", 
            lines=3 # Bigger input area for thumbs
        )
        analyze_btn = gr.Button("Analyze Sentence", variant="primary")
        
    # The output box
    output = gr.HighlightedText(
        label="Detection Results",
        color_map={"Darija": "#22c55e", "French": "#3b82f6", "English": "#ef4444"}
    )
    
    # Optional: Example inputs for quick testing
    gr.Examples(
        examples=["Chof dik la voiture it looks amazing", "Ana need ndir mise-a-jour", "slm monsieur 3afak you can add chwiya diyal les mots"],
        inputs=text_input
    )
    
    analyze_btn.click(fn=detect_code_switching, inputs=text_input, outputs=output)

if __name__ == "__main__":
    app.launch(share=False)
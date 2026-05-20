import gradio as gr
from transformers import pipeline
import re

# 1. Load the AI you just trained!
print("Loading the custom Code-Switching model...")
try:
    # We use aggregation_strategy="simple" to stitch sub-words back together
    classifier = pipeline("token-classification", model="./saved_model", tokenizer="./saved_model", aggregation_strategy="simple")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure your training is 100% finished before running this app!")

# 2. The Logic: Process the text and format it for the UI
def detect_code_switching(text):
    results = classifier(text)
    
    id2label = {
        'LABEL_0': 'O', 
        'LABEL_1': 'B-AR', 'LABEL_2': 'I-AR', 
        'LABEL_3': 'B-FR', 'LABEL_4': 'I-FR', 
        'LABEL_5': 'B-EN', 'LABEL_6': 'I-EN'
    }
    
    # Step A: Map the AI's predictions to every single character in the sentence
    char_labels = [None] * len(text)
    for entity in results:
        raw_label = entity.get('entity_group', entity.get('entity', 'LABEL_0'))
        actual_tag = id2label.get(raw_label, 'O')
        
        if "AR" in actual_tag: label = "Darija (Arabizi)"
        elif "FR" in actual_tag: label = "French"
        elif "EN" in actual_tag: label = "English"
        else: label = None
        
        for i in range(entity['start'], entity['end']):
            char_labels[i] = label
            
    # Step B: Split the sentence into clean, whole words
    tokens = re.split(r'(\s+)', text)
    highlighted_output = []
    current_idx = 0
    
    # Step C: The Majority Vote Algorithm
    for token in tokens:
        # Keep spaces uncolored
        if token.isspace():
            highlighted_output.append((token, None))
            current_idx += len(token)
            continue
            
        # Look at the AI's labels for the characters inside this specific word
        word_labels = char_labels[current_idx:current_idx+len(token)]
        valid_labels = [l for l in word_labels if l is not None]
        
        if valid_labels:
            # Which language has the most letters in this word? The winner takes all!
            dominant_label = max(set(valid_labels), key=valid_labels.count)
            highlighted_output.append((token, dominant_label))
        else:
            highlighted_output.append((token, None))
            
        current_idx += len(token)
        
    return highlighted_output

# 3. The Visual Interface (The Presentation GUI)
theme = gr.themes.Soft()
with gr.Blocks() as app: # REMOVED theme=theme from here
    gr.Markdown("# 🇲🇦 Moroccan Code-Switching Detector")
    gr.Markdown("An NLP model trained to instantly detect Darija, French, and English mixed in a single sentence.")
    
    with gr.Row():
        text_input = gr.Textbox(label="Enter a mixed sentence", placeholder="E.g., Chof dik la voiture it looks amazing wlh.", scale=3)
        analyze_btn = gr.Button("Analyze Sentence", variant="primary", scale=1)
        
    # The output box will color-code the words based on the AI's predictions
    output = gr.HighlightedText(
        label="Detection Results",
        color_map={"Darija (Arabizi)": "green", "French": "blue", "English": "red"}
    )
    
    analyze_btn.click(fn=detect_code_switching, inputs=text_input, outputs=output)

# 4. Launch the app (MOVED theme down here, turned share OFF)
if __name__ == "__main__":
    app.launch(theme=theme, share=False)
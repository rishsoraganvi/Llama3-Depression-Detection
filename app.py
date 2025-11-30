import gradio as gr
from unsloth import FastLanguageModel
import torch

# 1. Configuration
# We load the model you just pushed to Hugging Face!
model_name = "rishsoraganvi/Llama-3-Depression-Detector" 
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 2. Load Model & Tokenizer
print(f"⏳ Loading model from {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

# 3. Define Prompt Structure
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 4. Prediction Function
def predict_depression(text):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Analyze this tweet for signs of depression. Provide a risk label and reasoning.",
                text,
                "",
            )
        ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
    result = tokenizer.batch_decode(outputs)[0]
    
    # Clean up output
    clean_result = result.split("### Response:")[-1].strip()
    clean_result = clean_result.replace("<|eot_id|>", "")
    return clean_result

# 5. Build Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Mental Health Risk Detector")
    gr.Markdown(f"Model loaded from: {model_name}")
    
    with gr.Row():
        input_text = gr.Textbox(label="Enter Tweet", placeholder="I feel so empty...", lines=3)
        output_text = gr.Textbox(label="AI Analysis", lines=4)
        
    btn = gr.Button("Analyze")
    btn.click(predict_depression, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    demo.launch()

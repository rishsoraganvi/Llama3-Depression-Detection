# 🧠 Llama 3 Depression Detection System

## 📌 Overview
An instruction-tuned LLM fine-tuned to detect linguistic markers of depression in social media text. Built using **Meta Llama 3 (8B)**, **Unsloth**, and **QLoRA** on a T4 GPU.

## 🚀 Key Features
* **Explainable AI:** Outputs both a risk label (High/Low) and clinical reasoning.
* **Efficient Training:** Fine-tuned 8B parameters on consumer hardware using 4-bit quantization.
* **Real-World Data:** Trained on the Kaggle "Sentiment Analysis for Mental Health" dataset.

## 🛠️ Tech Stack
* **Model:** Meta Llama 3 8B Instruct
* **Framework:** PyTorch, Unsloth, Hugging Face Transformers
* **Technique:** QLoRA (Quantized Low-Rank Adaptation)
* **Deployment:** Gradio (Web Interface)

## 📊 Results
* **Accuracy:** ~90% on validation set
* **Recall:** Optimized for high sensitivity to minimize false negatives.

## 💻 How to Run
1. Open the notebook in Google Colab.
2. Install dependencies: `pip install unsloth`
3. Load the model from Hugging Face: `your-username/Llama-3-Depression-Detector`

## 📜 License
MIT

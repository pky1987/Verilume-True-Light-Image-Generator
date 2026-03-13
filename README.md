# ✨ Verilume: True Light Image Generator

<p align="center">
<img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/-Gradio-FF6B6B?style=for-the-badge&logo=gradio&logoColor=white" />
<img src="https://img.shields.io/badge/-MIT%20License-000000?style=for-the-badge&logo=opensourceinitiative&logoColor=white" />
<img src="https://img.shields.io/badge/-Diffusers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
<img src="https://img.shields.io/badge/-Transformers-FFB000?style=for-the-badge&logo=huggingface&logoColor=black" />
<img src="https://img.shields.io/badge/-DashScope-6C63FF?style=for-the-badge&logo=alibabacloud&logoColor=white" />
<img src="https://img.shields.io/badge/-Pillow-8B5CF6?style=for-the-badge&logo=python&logoColor=white" />
</p>

**Verilume** is a high-fidelity image generation and editing framework built upon the **Qwen-Image** series. It integrates state-of-the-art prompt engineering with large-scale vision-language models to provide precise, semantic, and visually stunning results.

---

## 🚀 Key Features

- **🔍 Intelligent Prompt Polishing**: Automatically transforms brief user inputs into world-class, descriptive English or Hindi prompts using Qwen-Plus/Max.
- **🖼️ Semantic Image Editing**: Modify images with natural language instructions (Add, Replace, Style Change) using the `Qwen-Image-Edit` pipeline.
- **⚡ Zero-GPU Optimization**: Native support for Hugging Face Spaces `spaces.GPU` decorators, ensuring efficient resource management.
- **🌐 Multilingual Core**: Full support for English, Hindi, and Chinese script detection and specialized rewriting.
- **📐 Flexible Aspect Ratios**: Precision generation for 1:1, 16:9, 9:16, 4:3, and 3:4 cinematic compositions.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/pky1987/Verilume-True-Light-Image-Generator.git
cd verilume
```

### 2. Environment Setup
We recommend using a virtual environment (Conda or venv):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. API Configuration
Verilume uses **DashScope** for prompt enhancement. Set your API key in your environment variables:
```bash
export DASH_API_KEY="your_api_key_here"  # On Windows: set DASH_API_KEY=your_api_key_here
```

---

## 🖥️ Usage

### A. Launch Web UI (Gradio)
Run the interactive web interface for image editing:
```bash
python api.py
```
Access the app at `http://localhost:7860`.

### B. Direct Image Generation
Use the `generate_prompt.py` script for high-quality generation with automated prompt polishing:
```bash
python generate_prompt.py
```

---

## 📂 Project Structure

```text
├── api.py                 # Gradio Web Application
├── generate_prompt.py     # Batch/CLI Generation Script
├── requirements.txt       # Project Dependencies
├── LICENSE                # MIT License
└── src/
    └── tools/
        ├── prompt.py      # DashScope API integration
        └── prompt_utils.py# Multi-language polishing logic & Edit-mode enhancement
```

---

## 💡 Prompt Engineering Tips

To get the most out of Verilume:
1. **Be Specific**: "A cat" → "A sleek Russian Blue cat sitting on a velvet cushion."
2. **Use Quotes for Text**: To render text exactly, use double quotes: `Add a sign saying "WELCOME HOME"`.
3. **Control Mood**: Include lighting keywords like "Golden hour", "Cinematic lighting", or "Muted tones".

---

## 🌟 Acknowledgments

Special thanks to the [QwenLM team](https://github.com/QwenLM/Qwen-Image) for their groundbreaking work on the Qwen-Image foundation models and the [Aliyun DashScope](https://help.aliyun.com/zh/model-studio/) platform.

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

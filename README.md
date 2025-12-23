# Mini-GPT: PyTorch Implementation with Lazy Loading

A lightweight, educational implementation of a GPT-style autoregressive language model (Decoder-only Transformer) in PyTorch. 

This project demonstrates how to train a Large Language Model (LLM) on consumer hardware by using **Lazy Loading** and **Memory Mapping** to handle datasets larger than available RAM.

## 🚀 Key Features

* **Architecture:** GPT-2 style Decoder-only Transformer with Pre-LayerNorm and Causal Masking.
* **Lazy Loading:** Uses `np.memmap` to stream data directly from the hard drive during training. Zero RAM overhead for the dataset.
* **Tokenization:** Implements Byte-Level BPE (Byte Pair Encoding) via the Hugging Face `tokenizers` library.
* **Hardware Aware:** Automatically detects and accelerates training on:
    * NVIDIA GPUs (`cuda`)
    * Apple Silicon Macs (`mps`)
    * CPU (fallback)
* **Visualizations:** Real-time plotting of Loss and Accuracy during training.

## 🛠️ Installation & Setup

It is recommended to use a virtual environment to manage dependencies.

### ⚠️ Prerequisite: Python Version
**Important:** To use NVIDIA GPU acceleration on Windows, you must use **Python 3.10, 3.11, or 3.12**. 
* *Python 3.13 is currently **not** supported by PyTorch GPU binaries.*
* *Ensure you have the **64-bit** version of Python installed.*

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/mini-gpt.git](https://github.com/yourusername/mini-gpt.git)
cd mini-gpt
```
### 2. Create a Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows(cmd):**
```bash
# Ensure you are using Python 3.11 or 3.12
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

**For NVIDIA GPU (Windows):**
You must install the specific CUDA-enabled version of PyTorch before other dependencies. Run this command:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install Remaining Libraries
pip install -r requirements.txt
```

## 🏃 Usage

### 1. Prepare Data
Place your raw text training data in a file named `input.txt` in the root directory.
* *Note: The code handles large files efficiently, so feel free to use datasets larger than your RAM.*

### 2. Train the Model
Run the main script to start training. The script will automatically clean the data, train the BPE tokenizer, and begin the training loop.

```bash
python main.py
```

* **Training Artifacts:**
    * `dataset.bin`: Binary memory-mapped token data.
    * `tokenizer.json`: Saved BPE tokenizer.
    * `model.pth`: Saved model checkpoint.
    * `dataset_preview.csv`: A preview of how your data is being tokenized and windowed.

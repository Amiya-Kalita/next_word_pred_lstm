# 📘 **LSTM Next-Word Prediction Web App**

This project implements a **word-level LSTM model** trained to perform **next-word prediction** using natural-language text.
A lightweight **Streamlit web interface** is included, allowing users to enter a text prompt and generate the next sequence of words interactively.

The project is built with **PyTorch** and **Streamlit**, making it easy to train, test, and deploy as a web application.

---

## 🚀 Features

* Trainable **LSTM next-word prediction model**
* Clean and intuitive **Streamlit UI**
* Adjustable **max-word generation** slider
* GPU-friendly training notebook (Google Colab)
* Predictive text generation with sampling (temperature scaling)

---

## 📂 Project Structure

```
.
├── app.py                 # Streamlit web interface
├── lstm_nextword.pt       # Saved trained PyTorch model
├── 1661-0.txt             # Training dataset (Sherlock Holmes)
├── README.md              # Project documentation
└── notebook.ipynb         # (Optional) Colab training notebook
```

---

## 🧠 Model Overview

The model is a **single-layer LSTM** trained on word sequences.
Key components:

* Word embeddings
* LSTM hidden layer (256 units)
* Fully connected output layer
* Cross-entropy loss for next-word prediction

Input: *previous N words*
Output: *probability distribution over the next word*

---

## 📊 Training (Google Colab)

Training is done in a clean and simple Colab notebook using PyTorch.

You can run:

```python
torch.save(model.state_dict(), "lstm_nextword.pt")
```

After training, download the `.pt` file and place it in your project directory.

---

## 🌐 Running the Web App

### **1. Install dependencies**

```bash
pip install torch streamlit numpy
```

### **2. Run the Streamlit server**

```bash
streamlit run app.py
```

### **3. Open your browser**

Streamlit will print a local URL (usually):

```
http://localhost:8501
```

---

## 🖥️ Web App Preview

Users can:

* Enter a text prompt
* Select how many words to generate
* View the extended generated sentence

The interface is clean, responsive, and designed for quick experimentation.

---

## ⚙️ How Inference Works

The app:

1. Loads the training vocabulary
2. Loads the saved PyTorch model
3. Converts input text → token IDs
4. Performs iterative LSTM forward passes
5. Samples next-word predictions
6. Returns the fully generated text

Temperature scaling and randomness allow natural-looking generation.

---

## 🚀 Deployment Options

You can deploy the Streamlit app on:

* **Streamlit Cloud**
* **HuggingFace Spaces**
* **Render**
* **AWS EC2**
* **Docker**

If you'd like, I can generate deployment files (Dockerfile, requirements.txt, HF Space template).

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🤝 Contributing

Pull requests, improvements, and suggestions are welcome.
For major changes, please open an issue first to discuss what you’d like to update.


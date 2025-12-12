import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from collections import Counter

# --------------------------------------------------------------
# Load and rebuild vocabulary (same method as training)
# --------------------------------------------------------------

text_path = "data.txt"  # place same file used during training
text = open(text_path, "r", encoding="utf-8").read()
text = text.lower()
text = re.sub(r"[^a-z0-9'\s]+", " ", text)
words = text.split()

counts = Counter(words)
vocab = ["<pad>", "<unk>"] + sorted(counts.keys(), key=lambda x: -counts[x])
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

# tokens
data = [stoi.get(w, stoi["<unk>"]) for w in words]

seq_len = 6
vocab_size = len(vocab)

# --------------------------------------------------------------
# Define LSTM Model (same structure as training)
# --------------------------------------------------------------

class WordLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=256, n_layers=1, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        out = out[:, -1, :]
        return self.fc(out)

# load model
device = torch.device("cpu")
model = WordLSTM(vocab_size).to(device)
model.load_state_dict(torch.load("lstm_nextword.pt", map_location=device))
model.eval()

# --------------------------------------------------------------
# Prediction function
# --------------------------------------------------------------

def predict_text(seed_text, max_words=20, temperature=1.0):
    model.eval()
    seed_tokens = re.sub(r"[^a-z0-9'\s]+", " ", seed_text.lower()).split()
    if len(seed_tokens) == 0:
        return "Please enter some text."

    idxs = [stoi.get(w, stoi["<unk>"]) for w in seed_tokens]
    out_words = seed_tokens.copy()

    for _ in range(max_words):
        cur = idxs[-seq_len:]
        if len(cur) < seq_len:
            cur = [stoi["<pad>"]] * (seq_len - len(cur)) + cur

        xb = torch.tensor([cur], dtype=torch.long)
        with torch.no_grad():
            logits = model(xb)
            probs = F.softmax(logits / temperature, dim=-1).numpy().ravel()

        next_idx = np.random.choice(len(probs), p=probs)
        next_word = itos.get(next_idx, "<unk>")

        out_words.append(next_word)
        idxs.append(next_idx)

    return " ".join(out_words)

# --------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------

st.title("🧠 Next Word Prediction LSTM Model")
st.write("Enter text below and let the model continue your sentence.")

seed_text = st.text_input("Enter your seed text:")
max_words = st.slider("How many words do you want to generate?", 1, 100, 20)

if st.button("Generate"):
    result = predict_text(seed_text, max_words=max_words)
    st.subheader("Generated Text:")
    st.write(result)

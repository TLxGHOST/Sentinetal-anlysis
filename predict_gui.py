import tensorflow as tf
import numpy as np
import pickle
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved model and tokenizer

#model path me apne fine tuned model ka name dalna aur uske corresponding pickle file ka name bhi dalna
MODEL_PATH = "model_1.h5"
TOKENIZER_PATH = "tokenizer.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

def process_input(text):
    """Preprocess input text to match training format."""
    sequences = tokenizer.texts_to_sequences([text])
    padded_input = pad_sequences(sequences, maxlen=100, padding="post")
    return padded_input

def predict_text():
    """Gets text input from the user and predicts sentiment."""
    user_input = text_entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return

    processed_input = process_input(user_input)
    prediction = model.predict(processed_input)[0][0]  # Get single prediction

    sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
    result_label.config(text=f"Prediction: {sentiment} ({prediction:.2f})")

# GUI Setup
root = tk.Tk()
root.title("Sentiment Analysis Predictor")
root.geometry("400x250")

tk.Label(root, text="Enter Text:", font=("Arial", 12)).pack(pady=5)
text_entry = tk.Text(root, height=5, width=40)
text_entry.pack(pady=5)

predict_button = tk.Button(root, text="Predict", command=predict_text, font=("Arial", 12))
predict_button.pack(pady=10)

result_label = tk.Label(root, text="Prediction: ", font=("Arial", 14, "bold"))
result_label.pack(pady=5)

root.mainloop()

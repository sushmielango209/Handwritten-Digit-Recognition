# Handwritten-Digit-Recognition-MNIST-
MNIST Handwritten Digit Recognition

This project uses Convolutional Neural Networks (CNNs) to recognize handwritten digits from the MNIST dataset. It demonstrates how deep learning can be applied to educational technology, such as smart exam-checking tools.


## 📌 Problem Statement

Handwritten digits vary widely in style and clarity. Automating their recognition can improve grading systems and accessibility tools. This project builds a digit classifier using TensorFlow/Keras and optionally includes a simple UI for real-time digit prediction.

## 🎯 Objective

- Train a CNN model to classify digits (0–9) from the MNIST dataset.
- Achieve over **98% accuracy** on test data.
- (Optional) Build a UI to draw digits and predict them using the trained model.

## 🛠️ Tech Stack

- Python 3.8+
- TensorFlow / Keras
- NumPy, Matplotlib
- Tkinter (for optional UI)
- VS Code (development environment)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition

# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install tensorflow matplotlib numpy pillow
```

---

## 🚀 Running the Project

### 1. Train the CNN Model

```bash
python mnist_cnn.py
```

This script:
- Loads and preprocesses MNIST data
- Builds and trains a CNN
- Evaluates accuracy
- Saves the model as `mnist_model.h5`

### 2. Run the Digit Drawing UI (Optional)

```bash
python digit_ui.py
```

Draw a digit in the window and click **Predict** to see the model's output.

---

## 📁 Project Structure

```
mnist-digit-recognition/
│
├── mnist_cnn.py         # CNN model training and evaluation
├── digit_ui.py          # Tkinter-based digit drawing interface
├── mnist_model.h5       # Saved trained model
├── README.md            # Project documentation
├── venv/                # Virtual environment
```

---

## 📊 Sample Output

```
Epoch 5/5
Train accuracy: 99.12%
Validation accuracy: 98.45%
Test accuracy: 98.47%
```

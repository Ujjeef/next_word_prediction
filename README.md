# Next Word Prediction using LSTM

A Deep Learning application that predicts the next word in a sentence using a trained Long Short-Term Memory (LSTM) neural network. Built using TensorFlow and Streamlit.

## Project Overview

Next Word Prediction is a Natural Language Processing (NLP) task that predicts the most probable next word based on the context of previously entered words.

This project uses an LSTM-based neural network to learn sequential patterns in text and generate next-word predictions through an interactive web application.

## Features

- Predicts the next word based on user input.
- Uses a trained LSTM neural network.
- Converts text into numerical sequences using a Keras Tokenizer.
- Applies sequence padding before model inference.
- Uses TensorFlow SavedModel for model loading and prediction.
- Provides an interactive frontend using Streamlit.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep Learning framework |
| Keras | Neural network and tokenization utilities |
| LSTM | Sequential text prediction |
| NumPy | Numerical computations |
| Streamlit | Web application interface |
| Pickle | Saving and loading tokenizer and sequence length |

## Project Structure

```text
next_word_prediciton/
│
├── app.py                  # Streamlit frontend and prediction logic
│
├── lstm_saved_model/       # Exported trained LSTM model
│   ├── assets/
│   ├── variables/
│   ├── saved_model.pb
│   └── fingerprint.pb
│
├── tokenizer.pkl           # Saved tokenizer
├── max_len.pkl             # Maximum sequence length (745)
├── .gitignore
└── README.md
```

## How It Works

1. **User Input:** The user enters a sentence through the Streamlit interface.
2. **Tokenization:** The tokenizer converts the input text into numerical word indices.
3. **Sequence Padding:** The input sequence is padded to the required length of 745 tokens.
4. **LSTM Prediction:** The trained model processes the sequence and generates probability scores for possible next words.
5. **Word Selection:** The word with the highest predicted probability is selected.
6. **Output:** The predicted next word is displayed on the web interface.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ujjeef/next_word_prediciton.git
```

### 2. Navigate to the project directory

```bash
cd next_word_prediciton
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate the environment:

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install tensorflow==2.13.1 streamlit==1.38.0 numpy==1.24.3
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal:

```text
http://localhost:8501
```

## Example

**Input:**

```text
life is about
```

**Output:**

The application displays the next word predicted by the trained LSTM model.

*Note: The predicted word depends on the patterns learned by the model during training.*

## Learning Outcomes

- Understanding Recurrent Neural Networks and LSTM architecture.
- Learning text preprocessing and tokenization.
- Converting textual data into numerical sequences.
- Understanding sequence padding and model inference.
- Exporting and loading trained Deep Learning models.
- Integrating a trained Deep Learning model with a Streamlit frontend.

## Future Enhancements

- Display multiple probable next words.
- Show prediction confidence scores.
- Implement sentence completion.
- Improve the frontend and user experience.

## Author

**Shaik Ujjeef**

Computer Science Undergraduate | AI/ML & Deep Learning Enthusiast

- GitHub: [@Ujjeef](https://github.com/Ujjeef)
- Repository: [next_word_prediciton](https://github.com/Ujjeef/next_word_prediciton)

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.get_logger().setLevel("ERROR")


# ==========================================
# 2. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "lstm_saved_model"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"


# ==========================================
# 3. PAGE CONFIGURATION + STYLING
# ==========================================

st.set_page_config(
    page_title="LSTM Next Word Prediction",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 820px; }

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        padding: 2rem 2rem 1.6rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; }
    .hero p  { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 1.02rem; }

    .pred-card {
        background: linear-gradient(135deg, #eef2ff, #fdf2f8);
        border: 1px solid #e0e7ff;
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .pred-label { color: #6b7280; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .pred-word  { font-size: 2.6rem; font-weight: 800; color: #4f46e5; line-height: 1.2; }
    .pred-prob  { color: #6b7280; font-size: 0.95rem; }

    .sentence-box {
        background: #f9fafb;
        border-left: 5px solid #7c3aed;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 1.15rem;
        color: #111827;
    }
    .sentence-box .gen { color: #db2777; font-weight: 700; }

    div.stButton > button { border-radius: 10px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 4. KERAS TOKENIZER COMPATIBILITY
# ==========================================

class CompatibleUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if (
            module == "keras.src.legacy.preprocessing.text"
            and name == "Tokenizer"
        ):
            from tensorflow.keras.preprocessing.text import Tokenizer
            return Tokenizer

        return super().find_class(module, name)


# ==========================================
# 5. LOAD TOKENIZER, MAX LENGTH, MODEL
# ==========================================

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = CompatibleUnpickler(file).load()

    with open(MAX_LEN_PATH, "rb") as file:
        max_len = pickle.load(file)

    return tokenizer, int(max_len)


@st.cache_resource
def load_model():
    model = tf.saved_model.load(str(MODEL_DIR))
    return model.signatures["serving_default"]


# ==========================================
# 6. PREDICTION LOGIC
# ==========================================

def get_input_spec(infer):
    """Return (input_name or None, input_spec) of the single model input."""
    positional_specs, keyword_specs = infer.structured_input_signature

    if keyword_specs:
        if len(keyword_specs) != 1:
            raise ValueError("This app expects a model with one input tensor.")
        name, spec = next(iter(keyword_specs.items()))
        return name, spec

    if len(positional_specs) != 1:
        raise ValueError("This app expects a model with one input tensor.")
    return None, positional_specs[0]


def get_expected_len(infer, max_len):
    """Use the model's real input length (often max_len - 1)."""
    _, spec = get_input_spec(infer)
    try:
        length = spec.shape[1]
    except Exception:
        length = None
    return int(length) if length else max_len


def predict_top_k(text, tokenizer, max_len, infer, k=5):
    """Return a list of (word, probability) sorted from best to worst."""
    sequence = tokenizer.texts_to_sequences([text])

    if not sequence[0]:
        raise ValueError(
            "None of the words you typed are in the training vocabulary. "
            "Try different words."
        )

    expected_len = get_expected_len(infer, max_len)

    padded = pad_sequences(
        sequence, maxlen=expected_len, padding="pre", truncating="pre"
    )

    input_name, spec = get_input_spec(infer)
    tensor = tf.convert_to_tensor(padded, dtype=spec.dtype)

    if input_name:
        prediction = infer(**{input_name: tensor})
    else:
        prediction = infer(tensor)

    probs = next(iter(prediction.values())).numpy()[0].copy()

    probs[0] = 0.0  # index 0 is padding, never a real word

    top_idx = np.argsort(probs)[::-1]

    results = []
    for idx in top_idx:
        word = tokenizer.index_word.get(int(idx))
        if word:
            results.append((word, float(probs[idx])))
        if len(results) == k:
            break

    return results


def generate_words(text, tokenizer, max_len, infer, n_words):
    """Greedy generation of n_words continuation words."""
    generated = []
    current = text.strip()

    for _ in range(n_words):
        word, _prob = predict_top_k(current, tokenizer, max_len, infer, k=1)[0]
        generated.append(word)
        current = f"{current} {word}"

    return generated


# ==========================================
# 7. SESSION STATE + CALLBACKS
# ==========================================

if "sentence" not in st.session_state:
    st.session_state.sentence = ""


def set_sentence(text):
    st.session_state.sentence = text


def append_word(word):
    st.session_state.sentence = f"{st.session_state.sentence.strip()} {word}".strip()


def clear_sentence():
    st.session_state.sentence = ""


# ==========================================
# 8. HEADER
# ==========================================

st.markdown(
    """
    <div class="hero">
        <h1>🧠 LSTM Next Word Prediction</h1>
        <p>Type the start of a sentence and let your trained LSTM
        predict what comes next.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 9. MAIN APPLICATION
# ==========================================

try:
    if not (MODEL_DIR / "saved_model.pb").exists():
        st.error("SavedModel file is missing (lstm_saved_model/saved_model.pb).")
        st.stop()
    if not TOKENIZER_PATH.exists():
        st.error("tokenizer.pkl is missing.")
        st.stop()
    if not MAX_LEN_PATH.exists():
        st.error("max_len.pkl is missing.")
        st.stop()

    with st.spinner("Loading your trained model..."):
        tokenizer, max_len = load_tokenizer()
        infer = load_model()

    expected_len = get_expected_len(infer, max_len)

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Number of suggestions", 3, 8, 5)
        n_words = st.slider("Words to auto-generate", 1, 15, 5)

        st.divider()
        st.subheader("📊 Model info")
        st.metric("Vocabulary size", f"{len(tokenizer.word_index):,}")
        st.metric("Model input length", expected_len)
        if expected_len != max_len:
            st.caption(f"max_len.pkl = {max_len}, but the model expects {expected_len}. "
                       f"The app uses {expected_len}.")

    # ---------- Input ----------
    st.text_input(
        "✍️ Enter your sentence",
        key="sentence",
        placeholder="Example: life is about",
        help="Press Enter to predict.",
    )

    st.caption("Quick examples:")
    examples = ["life is about", "i want to", "once upon a", "the world is"]
    ex_cols = st.columns(len(examples))
    for col, ex in zip(ex_cols, examples):
        col.button(ex, key=f"ex_{ex}", on_click=set_sentence, args=(ex,),
                   use_container_width=True)

    sentence = st.session_state.sentence.strip()

    # ---------- Output ----------
    if sentence:
        try:
            results = predict_top_k(sentence, tokenizer, max_len, infer, k=top_k)
        except ValueError as err:
            st.warning(str(err))
            results = []

        if results:
            best_word, best_prob = results[0]

            st.markdown(
                f"""
                <div class="pred-card">
                    <div class="pred-label">Most likely next word</div>
                    <div class="pred-word">{best_word}</div>
                    <div class="pred-prob">confidence: {best_prob:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**👆 Click a suggestion to add it to your sentence**")
            chip_cols = st.columns(len(results))
            for i, (col, (word, prob)) in enumerate(zip(chip_cols, results)):
                col.button(
                    f"{word}\n{prob:.0%}",
                    key=f"chip_{i}_{word}",
                    on_click=append_word,
                    args=(word,),
                    use_container_width=True,
                )

            with st.expander("📈 Probability breakdown", expanded=True):
                for word, prob in results:
                    left, right = st.columns([1, 4])
                    left.write(f"**{word}**")
                    right.progress(min(max(prob, 0.0), 1.0), text=f"{prob:.2%}")

            if best_prob > 0.95:
                st.info(
                    "The model is extremely confident. If it gives the same word "
                    "for every input, the model may be undertrained or collapsed."
                )

            st.divider()

            g1, g2 = st.columns([3, 1])
            if g1.button(f"✨ Auto-generate next {n_words} words",
                         type="primary", use_container_width=True):
                with st.spinner("Generating..."):
                    words = generate_words(sentence, tokenizer, max_len, infer, n_words)
                st.markdown(
                    f'<div class="sentence-box">{sentence} '
                    f'<span class="gen">{" ".join(words)}</span></div>',
                    unsafe_allow_html=True,
                )
            g2.button("🗑️ Clear", on_click=clear_sentence, use_container_width=True)
    else:
        st.info("Type a few words above (or pick an example) to get started.")

except Exception as error:
    st.error("Something went wrong.")
    st.exception(error)
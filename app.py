import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")

st.title("✏️ Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) in the box below and click **Predict**.")

# ── Load model (cached so it only loads once) ────────────────────────────────
@st.cache_resource
def load_model():
    """Load the saved CNN model from disk."""
    try:
        model = tf.keras.models.load_model("mnist_cnn.h5")
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

model = load_model()

# ── Drawing canvas ────────────────────────────────────────────────────────────
st.subheader("Draw here:")

canvas_result = st_canvas(
    fill_color="black",          # background colour
    stroke_width=18,             # pen thickness
    stroke_color="white",        # pen colour (white on black, like MNIST)
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):

    # Check that the user actually drew something
    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
        st.stop()

    # canvas_result.image_data is a numpy array (H, W, 4) — RGBA
    img_array = canvas_result.image_data.astype("uint8")

    # Check for a blank canvas (all pixels are black / zero)
    if img_array[:, :, :3].sum() == 0:
        st.warning("The canvas looks empty. Please draw a digit before predicting.")
        st.stop()

    # Convert RGBA → grayscale PIL image
    img = Image.fromarray(img_array, mode="RGBA").convert("L")

    # Resize to 28×28 (same size as MNIST training images)
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array and normalize to 0–1
    img_np = np.array(img).astype("float32") / 255.0

    # Reshape to (1, 28, 28, 1) — batch size 1, single channel
    img_np = img_np.reshape(1, 28, 28, 1)

    # Run prediction
    predictions = model.predict(img_np, verbose=0)   # shape: (1, 10)
    predicted_digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    # ── Display results ───────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Predicted Digit", value=str(predicted_digit))

    with col2:
        st.metric(label="Confidence", value=f"{confidence:.1f}%")

    # Show a confidence bar for all 10 digits
    st.subheader("Confidence per digit:")
    prob_dict = {str(i): float(predictions[0][i]) for i in range(10)}
    st.bar_chart(prob_dict)

    # Friendly feedback based on confidence level
    if confidence >= 90:
        st.success("High confidence prediction!")
    elif confidence >= 60:
        st.info("Moderate confidence — try drawing more clearly if this looks wrong.")
    else:
        st.warning("Low confidence — the model is unsure. Try redrawing the digit.")

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")

st.title("✏️ Handwritten Digit Recognizer")

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


def preprocess_for_prediction(img: Image.Image) -> np.ndarray:
    """
    Convert any PIL image to the 28×28 grayscale format the model expects.
    Handles both drawn (white digit on black) and uploaded (dark digit on light)
    images by auto-detecting which background is dominant and inverting if needed.
    """
    # Convert to grayscale
    img = img.convert("L")

    # Auto-invert: MNIST expects white digit on black background.
    # If the image is mostly light (e.g. a photo on white paper), invert it.
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)
        arr = np.array(img)

    # Crop to the bounding box of the digit to remove excess whitespace
    bbox = Image.fromarray(arr).getbbox()
    if bbox:
        img = Image.fromarray(arr).crop(bbox)

    # Resize to 28×28 with padding to preserve aspect ratio
    img.thumbnail((20, 20), Image.LANCZOS)
    padded = Image.new("L", (28, 28), 0)
    offset = ((28 - img.width) // 2, (28 - img.height) // 2)
    padded.paste(img, offset)

    # Normalize and reshape for the model
    img_np = np.array(padded).astype("float32") / 255.0
    return img_np.reshape(1, 28, 28, 1)


def show_prediction(img_np: np.ndarray):
    """Run the model and display results."""
    predictions = model.predict(img_np, verbose=0)   # shape: (1, 10)
    predicted_digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Digit", value=str(predicted_digit))
    with col2:
        st.metric(label="Confidence", value=f"{confidence:.1f}%")

    st.subheader("Confidence per digit:")
    prob_dict = {str(i): float(predictions[0][i]) for i in range(10)}
    st.bar_chart(prob_dict)

    if confidence >= 90:
        st.success("High confidence prediction!")
    elif confidence >= 60:
        st.info("Moderate confidence — try a clearer image if this looks wrong.")
    else:
        st.warning("Low confidence — the model is unsure. Try a different image or redraw.")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_draw, tab_upload = st.tabs(["✏️ Draw", "📁 Upload Image"])

# ── Tab 1: Draw ───────────────────────────────────────────────────────────────
with tab_draw:
    st.write("Draw a digit (0–9) in the box below and click **Predict**.")
    st.subheader("Draw here:")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("🔍 Predict", key="predict_draw", use_container_width=True):
        if canvas_result.image_data is None:
            st.warning("Please draw a digit first.")
            st.stop()

        img_array = canvas_result.image_data.astype("uint8")

        if img_array[:, :, :3].sum() == 0:
            st.warning("The canvas looks empty. Please draw a digit before predicting.")
            st.stop()

        # Canvas is already white-on-black RGBA — convert and resize to 28×28
        img_resized = Image.fromarray(img_array, mode="RGBA").convert("L").resize((28, 28), Image.LANCZOS)
        img_np = np.array(img_resized).astype("float32") / 255.0
        img_np = img_np.reshape(1, 28, 28, 1)

        show_prediction(img_np)

# ── Tab 2: Upload ─────────────────────────────────────────────────────────────
with tab_upload:
    st.write("Upload a photo or scan of a handwritten digit (0–9).")
    st.caption("Works best with a single digit on a plain background. Supports JPG, PNG, BMP, WEBP.")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        col_preview, col_spacer = st.columns([1, 2])
        with col_preview:
            st.image(img, caption="Uploaded image", use_container_width=True)

        if st.button("🔍 Predict", key="predict_upload", use_container_width=True):
            img_np = preprocess_for_prediction(img)
            show_prediction(img_np)

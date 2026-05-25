"""
Handwritten Digit Classification — Flask Backend
Loads a lightweight trained model (digit_model.npz) and serves predictions
for uploaded or camera-captured digit images.
"""

import os
import io
from pathlib import Path
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

# ─── App Configuration ───────────────────────────────────────────────
app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB upload limit

# ─── Load Model Once at Startup ──────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "digit_model.npz")).resolve()


def load_digit_model(model_path: Path) -> dict[str, np.ndarray]:
    data = np.load(model_path)
    return {
        "samples": data["samples"].astype("float32"),
        "labels": data["labels"].astype("int64"),
    }


def predict_digit(tensor: np.ndarray) -> tuple[int | None, float]:
    vector = tensor.reshape(-1).astype("float32")
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None, 0.0

    vector = vector / norm
    similarities = model["samples"] @ vector
    best_indices = np.argsort(similarities)[-9:]

    scores = {}
    for index in best_indices:
        label = int(model["labels"][index])
        scores[label] = scores.get(label, 0.0) + float(max(similarities[index], 0.0))

    predicted_digit = max(scores, key=scores.get)
    confidence = min(max(scores[predicted_digit] / 9.0, 0.0), 1.0)
    return predicted_digit, confidence


try:
    if not MODEL_PATH.exists():
        print(f"[WARNING] Model file not found at {MODEL_PATH}. Training a fresh model now.")
        from train_model import main as train_model_main

        train_model_main()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found after training at {MODEL_PATH}")

    model = load_digit_model(MODEL_PATH)
    model_load_error = None
    print(f"[SUCCESS] Model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    model_load_error = str(e)
    print(f"[ERROR] Failed to load model from {MODEL_PATH}: {e}")

# ─── Allowed Extensions ──────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


def allowed_file(filename: str) -> bool:
    """Check whether the uploaded file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, str]:
    """
    Clean preprocessing for MNIST-style digit recognition.
    Returns (tensor, status_message).
    """
    # ── 1. Decode image ──────────────────────────────────────────
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None, "Invalid image format or corrupted data."

    # ── 2. Convert to grayscale ──────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 3. Resize to manageable size for digit extraction ────────
    if gray.shape[0] > 400 or gray.shape[1] > 400:
        scale = min(400 / gray.shape[0], 400 / gray.shape[1])
        gray = cv2.resize(gray, None, fx=scale, fy=scale)

    # ── 4. Apply Gaussian blur for noise reduction ──────────────
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── 5. Otsu thresholding ─────────────────────────────────────
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── 6. Ensure digit is WHITE (255) on BLACK (0) background ──
    # MNIST convention: digit pixels are 255, background is 0.
    # We assume the background occupies the majority of the frame.
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # ── 6b. Guard against empty/uniform images ──────────────────
    if np.max(thresh) == 0 or np.all(thresh == thresh[0,0]):
        return None, "No handwriting detected. Please draw something on the canvas."

    # ── 7. Light morphological cleanup ──────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # ── 8. Find bounding box of digit ───────────────────────────
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, "No handwriting detected in the image. Please draw more clearly."

    # Get largest contour (the digit)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Check for minimum size to filter out noise
    if w < 8 or h < 8:
        return None, "Drawing is too small or faint to be recognized as a digit."

    # Extract digit region
    digit = thresh[y:y+h, x:x+w]

    # ── 9. Calculate optimal size preserving aspect ratio ───────
    digit_h, digit_w = digit.shape
    aspect_ratio = digit_w / digit_h if digit_h > 0 else 1
    
    if aspect_ratio >= 1:
        target_w = 20
        target_h = max(1, int(20 / aspect_ratio))
    else:
        target_h = 20
        target_w = max(1, int(20 * aspect_ratio))

    digit_resized = cv2.resize(digit, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # ── 10. Create 28x28 canvas and center the digit ────────────
    canvas = np.zeros((28, 28), dtype=np.uint8)
    
    offset_x = (28 - target_w) // 2
    offset_y = (28 - target_h) // 2
    
    canvas[offset_y:offset_y+target_h, offset_x:offset_x+target_w] = digit_resized

    # ── 11. Normalize to [0, 1] range ───────────────────────────
    canvas_normalized = canvas.astype("float32") / 255.0
    tensor = canvas_normalized.reshape(1, 28, 28, 1)

    return tensor, "Success"


def preprocess_camera_image(image_bytes: bytes) -> tuple[np.ndarray, str]:
    """
    Advanced preprocessing for CAMERA captured images across all digits (0-9).
    Matches MNIST stroke characteristics and uses Center of Mass alignment.
    """
    # 1. Decode image
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None, "Invalid image format."

    # 2. Convert to grayscale & Resize for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] > 1000 or gray.shape[1] > 1000:
        scale = 1000 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 3. Enhance contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Noise reduction
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # 5. Robust Thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12
    )

    # 6. Morphological cleanup - Use CLOSE to fill gaps without bloating the digit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 7. Find contours and isolate the digit
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "No handwriting detected."

    # Filter contours
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 0.05 < aspect_ratio < 20: 
            candidates.append(cnt)

    if not candidates:
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        all_points = np.concatenate(candidates)
        largest_contour = all_points

    x, y, w, h = cv2.boundingRect(largest_contour)
    digit_roi = thresh[y:y+h, x:x+w]

    # 8. Resize to fit in 20x20 box (standard MNIST size)
    # Using INTER_AREA which is best for downsampling
    if w > h:
        new_w = 20
        new_h = int(h * (20 / w))
    else:
        new_h = 20
        new_w = int(w * (20 / h))
    
    new_w, new_h = max(1, new_w), max(1, new_h)
    digit_resized = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 9. Place in 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[oy:oy+new_h, ox:ox+new_w] = digit_resized

    # 10. Center by Center of Mass
    moments = cv2.moments(canvas)
    if moments['m00'] > 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        shift_x = 14 - cx
        shift_y = 14 - cy
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, M, (28, 28))

    # 11. Subtle Blur to match MNIST characteristics (makes it less sensitive to noise)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    # 12. Normalize and return tensor
    tensor = canvas.astype("float32") / 255.0
    tensor = tensor.reshape(1, 28, 28, 1)

    return tensor, "Success"




# ─── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Return deployment diagnostics for Render health checks and debugging."""
    return jsonify({
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "model_file_exists": MODEL_PATH.exists(),
        "model_file_size": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else None,
        "model_load_error": model_load_error,
    }), 200 if model is not None else 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept an image via POST, preprocess it, and return the predicted digit.
    """
    if model is None:
        return jsonify({
            "prediction": None,
            "message": "Model not loaded. Check server logs.",
            "error": model_load_error,
            "model_path": str(MODEL_PATH),
            "model_file_exists": MODEL_PATH.exists()
        }), 500

    image_bytes = None
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"prediction": None, "message": "No file selected."}), 400
        image_bytes = file.read()
    elif request.data:
        image_bytes = request.data
    else:
        return jsonify({"prediction": None, "message": "No image data received."}), 400

    # ── Preprocess & Predict ──────────────────────────────────────
    try:
        source = request.form.get("source", "upload")
        if source == "camera":
            tensor, status = preprocess_camera_image(image_bytes)
        else:
            tensor, status = preprocess_image(image_bytes)
            
        if tensor is None:
            return jsonify({"prediction": None, "message": status})
    except Exception as e:
        return jsonify({"prediction": None, "message": f"Processing error: {str(e)}"}), 400

    try:
        predicted_digit, confidence = predict_digit(tensor)
        
        # Log for debugging
        print(f"Prediction: {predicted_digit}, Confidence: {confidence:.4f}")
    except Exception as e:
        return jsonify({"prediction": None, "message": f"Prediction error: {str(e)}"}), 500

    # ── Confidence Threshold ──────────────────────────────────────
    if confidence < 0.35:
        return jsonify({
            "prediction": None,
            "confidence": round(confidence, 4),
            "message": "Ambiguous handwriting. Could you draw it more clearly?"
        })

    return jsonify({
        "prediction": predicted_digit,
        "confidence": round(confidence, 4),
        "message": f"Looks like a {predicted_digit}!"
    })


import base64

@app.route("/predict_canvas", methods=["POST"])
def predict_canvas():
    """
    Accept a base64 encoded image from the canvas, preprocess it, and return the prediction.
    """
    if model is None:
        return jsonify({
            "prediction": None,
            "message": "Model not loaded.",
            "error": model_load_error,
            "model_path": str(MODEL_PATH),
            "model_file_exists": MODEL_PATH.exists()
        }), 500

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"prediction": None, "message": "Missing image data."}), 400

        # Decode base64 image
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"prediction": None, "message": "Failed to decode image."}), 400

        # ── Preprocessing ──────────────────────────────────────────
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize to 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 3. Invert colors (Canvas is black digit on white, Model expects white on black)
        inverted = 255 - resized
        
        # 4. Normalize
        normalized = inverted.astype("float32") / 255.0
        
        # 5. Reshape to (1, 28, 28, 1)
        tensor = normalized.reshape(1, 28, 28, 1)

        # ── Prediction ──────────────────────────────────────────────
        predicted_digit, confidence = predict_digit(tensor)

        # ── Confidence Handling ─────────────────────────────────────
        if confidence < 0.5:
            return jsonify({
                "prediction": None,
                "message": "No valid handwritten number detected"
            })

        return jsonify({
            "prediction": predicted_digit,
            "confidence": round(confidence, 4),
            "message": "Digit detected successfully"
        })

    except Exception as e:
        return jsonify({"prediction": None, "message": f"Error: {str(e)}"}), 500


# ─── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

/**
 * Digit Recognizer — Client-side Logic
 * Handles file upload, camera capture, preview, and prediction API calls.
 */

(function () {
    "use strict";

    // ─── DOM References ──────────────────────────────────────────────
    const uploadBtn = document.getElementById("uploadBtn");
    const cameraBtn = document.getElementById("cameraBtn");
    const fileInput = document.getElementById("fileInput");
    const previewArea = document.getElementById("previewArea");
    const previewImage = document.getElementById("previewImage");
    const clearBtn = document.getElementById("clearBtn");
    const cameraView = document.getElementById("cameraView");
    const cameraFeed = document.getElementById("cameraFeed");
    const captureBtn = document.getElementById("captureBtn");
    const closeCameraBtn = document.getElementById("closeCameraBtn");
    const captureCanvas = document.getElementById("captureCanvas");
    const predictSection = document.getElementById("predictSection");
    const predictBtn = document.getElementById("predictBtn");
    const spinner = document.getElementById("spinner");
    const resultArea = document.getElementById("resultArea");
    const resultContent = document.getElementById("resultContent");

    const drawBtn = document.getElementById("drawBtn");
    const drawSection = document.getElementById("drawSection");
    const mainCanvas = document.getElementById("mainCanvas");
    const clearCanvasBtn = document.getElementById("clearCanvasBtn");
    const predictCanvasBtn = document.getElementById("predictCanvasBtn");

    let currentBlob = null;   // Holds the image blob ready for upload
    let cameraStream = null;  // MediaStream reference
    let isFrontCamera = false; // Track if using front-facing camera
    let imageSource = "upload"; // Track if image is from camera or upload

    // ─── Canvas State ────────────────────────────────────────────────
    const ctx = mainCanvas.getContext("2d");
    let isDrawing = false;

    function initCanvas() {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);
        ctx.strokeStyle = "black";
        ctx.lineWidth = 15;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
    }
    initCanvas();

    // ─── Input Switching ─────────────────────────────────────────────
    uploadBtn.addEventListener("click", () => {
        stopCamera();
        hideDraw();
        fileInput.click();
    });

    cameraBtn.addEventListener("click", async () => {
        hidePreview();
        hideDraw();
        hideResult();
        // ... (existing camera logic below)
    });

    drawBtn.addEventListener("click", () => {
        stopCamera();
        hidePreview();
        hideResult();
        showDraw();
    });

    // ─── Drawing Logic ───────────────────────────────────────────────
    function getPos(e) {
        const rect = mainCanvas.getBoundingClientRect();
        // Handle touch vs mouse
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        
        // Account for CSS scaling if any
        const scaleX = mainCanvas.width / rect.width;
        const scaleY = mainCanvas.height / rect.height;
        
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDraw(e) {
        isDrawing = true;
        const { x, y } = getPos(e);
        ctx.beginPath();
        ctx.moveTo(x, y);
        e.preventDefault();
    }

    function draw(e) {
        if (!isDrawing) return;
        const { x, y } = getPos(e);
        ctx.lineTo(x, y);
        ctx.stroke();
        e.preventDefault();
    }

    function stopDraw() {
        isDrawing = false;
        ctx.closePath();
    }

    mainCanvas.addEventListener("mousedown", startDraw);
    mainCanvas.addEventListener("mousemove", draw);
    window.addEventListener("mouseup", stopDraw);

    mainCanvas.addEventListener("touchstart", startDraw, { passive: false });
    mainCanvas.addEventListener("touchmove", draw, { passive: false });
    mainCanvas.addEventListener("touchend", stopDraw);

    clearCanvasBtn.addEventListener("click", initCanvas);

    predictCanvasBtn.addEventListener("click", async () => {
        hideResult();
        predictCanvasBtn.disabled = true;
        spinner.classList.add("active");

        try {
            const dataUrl = mainCanvas.toDataURL("image/png");
            const res = await fetch("/predict_canvas", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataUrl })
            });
            const data = await res.json();
            renderResult(data);
        } catch (err) {
            renderResult({ prediction: null, message: "Network error — server may be offline." });
        } finally {
            predictCanvasBtn.disabled = false;
            spinner.classList.remove("active");
        }
    });

    // ─── File Input Change ───────────────────────────────────────────
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;

        currentBlob = file;
        imageSource = "upload"; // Set source to upload
        const reader = new FileReader();
        reader.onload = (ev) => {
            previewImage.src = ev.target.result;
            showPreview();
        };
        reader.readAsDataURL(file);
    });

    // ─── Camera Flow ─────────────────────────────────────────────────
    cameraBtn.addEventListener("click", async () => {
        try {
            // Try back camera first, fall back to any camera
            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "environment", width: 300, height: 300 },
                    audio: false,
                });
                isFrontCamera = false;
            } catch {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 300, height: 300 },
                    audio: false,
                });
                // Check if the granted camera is front-facing
                const track = stream.getVideoTracks()[0];
                const settings = track.getSettings();
                isFrontCamera = !settings.facingMode || settings.facingMode === "user";
            }
            cameraStream = stream;
            cameraFeed.srcObject = cameraStream;
            // Mirror preview for front-facing cameras so it feels natural
            cameraFeed.style.transform = isFrontCamera ? "scaleX(-1)" : "none";
            cameraView.classList.add("active");
        } catch (err) {
            alert("Camera access denied or unavailable.\n" + err.message);
        }
    });

    captureBtn.addEventListener("click", () => {
        if (!cameraStream) return;

        const width = cameraFeed.videoWidth;
        const height = cameraFeed.videoHeight;
        captureCanvas.width = width;
        captureCanvas.height = height;

        const captureCtx = captureCanvas.getContext("2d");

        // If front camera, flip horizontally so the digit isn't mirrored
        if (isFrontCamera) {
            captureCtx.translate(width, 0);
            captureCtx.scale(-1, 1);
        }
        captureCtx.drawImage(cameraFeed, 0, 0, width, height);
        captureCtx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform

        captureCanvas.toBlob((blob) => {
            if (!blob) return;
            currentBlob = blob;
            imageSource = "camera"; // Set source to camera
            previewImage.src = URL.createObjectURL(blob);
            showPreview();
            stopCamera();
        }, "image/png");
    });

    closeCameraBtn.addEventListener("click", stopCamera);

    // ─── Clear / Reset ───────────────────────────────────────────────
    clearBtn.addEventListener("click", () => {
        currentBlob = null;
        fileInput.value = "";
        hidePreview();
        hideResult();
    });

    // ─── Predict File/Camera ─────────────────────────────────────────
    predictBtn.addEventListener("click", async () => {
        if (!currentBlob) return;

        hideResult();
        predictBtn.disabled = true;
        spinner.classList.add("active");

        try {
            const formData = new FormData();
            formData.append("file", currentBlob, "digit.png");
            formData.append("source", imageSource);

            const res = await fetch("/predict", { method: "POST", body: formData });
            const data = await res.json();

            renderResult(data);
        } catch (err) {
            renderResult({ prediction: null, message: "Network error — server may be offline." });
        } finally {
            predictBtn.disabled = false;
            spinner.classList.remove("active");
        }
    });

    // ─── Render Result ───────────────────────────────────────────────
    function renderResult(data) {
        resultArea.classList.add("active");

        if (data.prediction !== null && data.prediction !== undefined) {
            const confidence = data.confidence !== undefined
                ? (data.confidence * 100).toFixed(1)
                : "—";

            resultContent.innerHTML = `
                <div class="result-digit">${data.prediction}</div>
                <div class="result-confidence">Confidence: ${confidence}%</div>
                <div class="result-message">${data.message}</div>
            `;
        } else {
            resultContent.innerHTML = `
                <div class="result-icon-error">⚠️</div>
                <div class="result-message error">${data.message || "Something went wrong."}</div>
            `;
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────
    function showPreview() {
        cameraView.classList.remove("active");
        drawSection.classList.remove("active");
        previewArea.classList.add("active");
        predictSection.classList.add("active");
        hideResult();
    }

    function hidePreview() {
        previewArea.classList.remove("active");
        predictSection.classList.remove("active");
    }

    function showDraw() {
        drawSection.classList.add("active");
        hideResult();
    }

    function hideDraw() {
        drawSection.classList.remove("active");
    }

    function hideResult() {
        resultArea.classList.remove("active");
        resultContent.innerHTML = "";
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach((t) => t.stop());
            cameraStream = null;
        }
        cameraFeed.srcObject = null;
        cameraView.classList.remove("active");
    }
})();

import sys, json, base64, io, os
from pathlib import Path
from typing import Dict, Any

# Lazy import heavy libs so failure can be reported gracefully
try:
    from ultralytics import YOLO  # type: ignore
    import cv2  # type: ignore
    from PIL import Image
    import numpy as np
except Exception as e:  # pragma: no cover
    print(json.dumps({"ok": False, "error": f"Import failure: {e}"}))
    sys.exit(1)

MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")  # generic model; replace with custom face model if available
# NOTE: For better cat/dog face results you should train / fine-tune a custom model. Here we approximate using generic objects.

# Attempt to load model once
_model = None

def load_model():
    global _model
    if _model is None:
        try:
            _model = YOLO(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    return _model

# Simple mapping of class names to detect faces (approx). If you have a dedicated cat/dog face model, adjust accordingly.
PET_CLASSES = {"cat", "dog"}

def encode_image(img_bgr) -> str:
    _, buf = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buf).decode('utf-8')

def crop_to_square(img):
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0+side, x0:x0+side]

def detect_and_crop(path: Path) -> Dict[str, Any]:
    model = load_model()
    src = cv2.imread(str(path))
    if src is None:
        raise ValueError("Could not read image")

    # Run inference
    results = model.predict(source=str(path), imgsz=640, verbose=False)
    best_box = None
    best_conf = 0.0
    pet_type = "Unknown"

    for r in results:
        boxes = r.boxes
        names = r.names
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls.item())
            name = names.get(cls_id, str(cls_id))
            conf = float(box.conf.item())
            if name.lower() in PET_CLASSES and conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
                pet_type = name.capitalize()

    cropped = None
    steps = []

    if best_box:
        x1,y1,x2,y2 = map(int, best_box)
        h, w = src.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        face = src[y1:y2, x1:x2]
        steps.append("Face detection & cropping")
        if face.size > 0:
            cropped = face
    else:
        # fallback center crop
        cropped = crop_to_square(src)
        steps.append("Center square crop (no face detected)")

    # Resize to 224x224
    resized = cv2.resize(cropped, (224,224), interpolation=cv2.INTER_AREA)
    steps.append("Resize to 224x224")

    # Encode outputs
    original_b64 = encode_image(src)
    processed_b64 = encode_image(resized)

    return {
        "ok": True,
        "pet_type": pet_type,
        "confidence": round(best_conf * 100, 2),
        "steps": steps,
        "original_base64": original_b64,
        "processed_base64": processed_b64
    }

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Usage: process_image.py <image_path>"}))
        return
    path = Path(sys.argv[1])
    if not path.exists():
        print(json.dumps({"ok": False, "error": "Image path not found"}))
        return
    try:
        result = detect_and_crop(path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))

if __name__ == "__main__":
    main()

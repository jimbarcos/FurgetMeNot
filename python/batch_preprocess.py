import argparse
import os
import sys
import cv2  # type: ignore
from pathlib import Path
from typing import Tuple, Optional
import json
import hashlib

try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    print(f"[FATAL] Cannot import ultralytics: {e}")
    sys.exit(1)


PET_CLASSES = {"cat", "dog"}


def load_model(model_path: str):
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"[FATAL] Failed to load model '{model_path}': {e}")
        sys.exit(1)


def safe_imread(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def crop_to_square(img):
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def detect_face(model, image_path: Path) -> Tuple[Optional[Tuple[int, int, int, int]], str, float]:
    """Return (box, pet_type, confidence) selecting best cat/dog box if present."""
    results = model.predict(source=str(image_path), imgsz=640, verbose=False)
    best_box = None
    best_conf = 0.0
    pet_type = "Unknown"
    for r in results:
        boxes = getattr(r, 'boxes', None)
        names = getattr(r, 'names', {})
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls.item())
            name = names.get(cls_id, str(cls_id)).lower()
            conf = float(box.conf.item())
            if name in PET_CLASSES and conf > best_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                best_box = (x1, y1, x2, y2)
                best_conf = conf
                pet_type = name.capitalize()
    return best_box, pet_type, best_conf


def preprocess_single(model, image_path: Path, out_root: Path, min_conf: float):
    try:
        img = safe_imread(image_path)
    except Exception as e:
        print(f"[WARN] Skipping unreadable file {image_path}: {e}")
        return None

    box, pet_type, conf = detect_face(model, image_path)

    label_dir = "Unknown"
    if pet_type in ("Cat", "Dog") and conf >= min_conf:
        label_dir = pet_type + 's'  # Cats / Dogs

    if box and label_dir != "Unknown":
        x1, y1, x2, y2 = box
        h, w = img.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        crop = img[y1:y2, x1:x2]
    else:
        crop = crop_to_square(img)

    try:
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"[WARN] Resize failed for {image_path}: {e}")
        return None

    target_dir = out_root / label_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # Generate deterministic yet unique-ish filename to avoid collisions
    stem_hash = hashlib.sha1(str(image_path).encode('utf-8')).hexdigest()[:10]
    out_name = f"{image_path.stem}_{stem_hash}.jpg"
    out_path = target_dir / out_name

    if not cv2.imwrite(str(out_path), resized):
        print(f"[WARN] Failed to write {out_path}")
        return None

    return {
        "source": str(image_path),
        "output": str(out_path),
        "label": label_dir,
        "confidence": round(conf * 100, 2),
        "pet_type": pet_type,
    }


def iter_image_files(root: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def parse_args():
    ap = argparse.ArgumentParser(description="Batch preprocess pet images (face detect + resize 224x224)")
    ap.add_argument('--input', '-i', required=True, help='Input folder containing images (recursively searched)')
    ap.add_argument('--output', '-o', default='Preprocessed', help='Output root folder (default: Preprocessed)')
    ap.add_argument('--model', '-m', default=os.environ.get('YOLO_MODEL_PATH', 'yolov8n.pt'), help='YOLO model path (default: yolov8n.pt)')
    ap.add_argument('--min-conf', type=float, default=0.15, help='Minimum confidence to classify as Cat/Dog (default 0.15)')
    ap.add_argument('--json-report', help='Optional path to write a JSON summary report')
    return ap.parse_args()


def main():
    args = parse_args()
    in_root = Path(args.input)
    if not in_root.exists():
        print(f"[ERROR] Input folder not found: {in_root}")
        sys.exit(1)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)

    files = list(iter_image_files(in_root))
    total = len(files)
    if total == 0:
        print("[INFO] No image files found.")
        return

    print(f"[INFO] Processing {total} images...")
    results = []
    for idx, img_path in enumerate(files, start=1):
        info = preprocess_single(model, img_path, out_root, args.min_conf)
        if info:
            results.append(info)
        if idx % 25 == 0 or idx == total:
            print(f"  - {idx}/{total} done")

    cats = len([r for r in results if r['label'] == 'Cats'])
    dogs = len([r for r in results if r['label'] == 'Dogs'])
    unknown = len([r for r in results if r['label'] == 'Unknown'])

    print(f"[SUMMARY] Cats: {cats} | Dogs: {dogs} | Unknown: {unknown} | Total saved: {len(results)}")
    if args.json_report:
        try:
            with open(args.json_report, 'w', encoding='utf-8') as f:
                json.dump({"results": results}, f, indent=2)
            print(f"[INFO] JSON report written to {args.json_report}")
        except Exception as e:
            print(f"[WARN] Failed to write JSON report: {e}")


if __name__ == '__main__':
    main()

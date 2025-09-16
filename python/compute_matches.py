import os
import sys
import json
import base64
import hashlib
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF C++ logs

try:
    import numpy as np  # type: ignore
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import backend as K  # ensure K is defined for Lambda layers
    from tensorflow.keras.preprocessing import image as keras_image
    import warnings, logging
    # Suppress common noisy warnings & TF/Absl verbosity so only JSON is printed
    warnings.filterwarnings("ignore")
    try:
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass
    try:
        import absl.logging as absl_logging  # type: ignore
        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass
except Exception as e:
    print(json.dumps({"ok": False, "error": f"Import failure: {e}"}))
    sys.exit(1)


def _euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def _contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = K.cast(y_true, y_pred.dtype)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def _enhanced_contrastive_loss(y_true, y_pred, margin=1.0):
    return _contrastive_loss(y_true, y_pred, margin=margin)

def _pearson_correlation_distance(vectors):
    x, y = vectors
    x_mean = K.mean(x, axis=1, keepdims=True)
    y_mean = K.mean(y, axis=1, keepdims=True)
    xm = x - x_mean
    ym = y - y_mean
    r_num = K.sum(xm * ym, axis=1, keepdims=True)
    r_den = K.sqrt(K.sum(K.square(xm), axis=1, keepdims=True) * K.sum(K.square(ym), axis=1, keepdims=True) + K.epsilon())
    r = r_num / r_den
    return 1 - r

class FallbackCapsuleLayer(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return inputs

class FallbackPrimaryCapsule(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return inputs

class FallbackEnhancedCapsuleLayer(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return inputs

class FallbackMultiHeadAttention(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        if isinstance(inputs, (list, tuple)):
            return inputs[0]
        return inputs

def build_extended_custom_objects():
    return {
        'euclidean_distance': _euclidean_distance,
        'contrastive_loss': _contrastive_loss,
        'enhanced_contrastive_loss': _enhanced_contrastive_loss,
        'pearson_correlation_distance': _pearson_correlation_distance,
        'CapsuleLayer': FallbackCapsuleLayer,
        'PrimaryCapsule': FallbackPrimaryCapsule,
        'EnhancedCapsuleLayer': FallbackEnhancedCapsuleLayer,
        'MultiHeadAttention': FallbackMultiHeadAttention,
        'K': K,
    }

def load_siamese_model(model_path: Path, debug: bool = False):
    logs: List[str] = []
    def log(msg):
        logs.append(msg)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = None
    attempt_log: List[str] = []
    # Attempt 1
    try:
        log('Attempt1: load_model compile=False')
        model = tf.keras.models.load_model(str(model_path), compile=False)
        attempt_log.append('Attempt1=OK')
        log('Attempt1 success')
    except Exception as e1:
        attempt_log.append(f'Attempt1=FAIL({e1})')
        log(f'Attempt1 failed: {e1}')
    # Attempt 2 with custom objects
    if model is None:
        try:
            log('Attempt2: load_model with extended custom_objects')
            ext = build_extended_custom_objects()
            model = tf.keras.models.load_model(str(model_path), custom_objects=ext, compile=False)
            attempt_log.append('Attempt2=OK')
            log('Attempt2 success')
        except Exception as e2:
            attempt_log.append(f'Attempt2=FAIL({e2})')
            log(f'Attempt2 failed: {e2}')
    if model is None:
        attempt_log.append('Attempt3=SKIP')
        log('Attempt3 skipped (no reconstruction)')
    if model is None:
        raise RuntimeError(f"All load attempts failed: {';'.join(attempt_log)}")
    # Extract base embedding pathway if siamese with 2 inputs (distance output)
    try:
        if len(model.inputs) == 2 and model.output_shape[-1] == 1:
            log('Detected siamese distance model; locating distance Lambda layer')
            distance_layer = None
            for lyr in model.layers[::-1]:
                if lyr.name == 'distance':
                    distance_layer = lyr
                    break
            if distance_layer is None:
                log('No distance layer found; keeping model as-is')
            else:
                # The distance layer takes list [emb_a, emb_p]
                inbound = distance_layer.input
                if isinstance(inbound, (list, tuple)) and len(inbound) == 2:
                    emb_tensor = inbound[0]  # anchor embedding
                    # Build embedding model mapping anchor input -> embedding tensor
                    anchor_input = model.inputs[0]
                    embedding_model = tf.keras.Model(inputs=anchor_input, outputs=emb_tensor, name='embedding_model')
                    model = embedding_model
                    log(f'Embedding model constructed from distance layer input: {emb_tensor.name}')
        else:
            log('Model not recognized as two-input distance model; using directly')
    except Exception as e:
        log(f'Embedding extraction warning: {e}')
    return model, logs, attempt_log


def preprocess_img(path: Path, target_size=(224, 224)) -> np.ndarray:
    img = keras_image.load_img(str(path), target_size=target_size)
    arr = keras_image.img_to_array(img)
    # Normalize similar to MobileNetV2 expectations (if trained that way)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr


def preprocess_bytes(b64: str, target_size=(224, 224)) -> np.ndarray:
    raw = base64.b64decode(b64)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        return preprocess_img(Path(tmp.name), target_size)


def compute_embedding(model, arr: np.ndarray) -> np.ndarray:
    batch = np.expand_dims(arr, axis=0)
    try:
        emb = model.predict(batch, verbose=0)
    except ValueError as ve:
        # Possibly still a siamese expecting two inputs; duplicate batch
        if 'expects 2 input' in str(ve).lower():
            emb = model.predict([batch, batch], verbose=0)
        else:
            raise
    if isinstance(emb, list):
        emb = emb[0]
    emb = np.squeeze(emb)
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def list_images(folder: Path) -> List[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]


def main():
    # Args: <processed_image_input> <pet_type> <preprocessed_dir> [top_k] [--debug]
    # processed_image_input can be either a path or base64 string.
    if len(sys.argv) < 4:
        print(json.dumps({"ok": False, "error": "Usage: compute_matches.py <processed_image_path|base64> <pet_type> <preprocessed_dir> [top_k] [--debug]"}))
        return
    processed_input = sys.argv[1]
    pet_type = sys.argv[2].lower()
    pre_dir = Path(sys.argv[3])
    # Detect optional args
    debug = False
    remaining = sys.argv[4:]
    top_k = 3
    for item in remaining:
        if item == '--debug':
            debug = True
        else:
            try:
                top_k = int(item)
            except ValueError:
                pass

    # Resolve model path flexibly
    candidate_paths = []
    # Environment override
    env_model = os.getenv('SIAMESE_MODEL_PATH')
    if env_model:
        candidate_paths.append(Path(env_model))
    # Common relative folders
    candidate_paths.append(Path('models') / 'best_model.h5')
    candidate_paths.append(Path('model') / 'best_model.h5')
    candidate_paths.append(Path(__file__).resolve().parent.parent / 'models' / 'best_model.h5')
    candidate_paths.append(Path(__file__).resolve().parent.parent / 'model' / 'best_model.h5')

    chosen_model_path = None
    load_logs: List[str] = []
    attempt_log: List[str] = []
    last_error = None
    model = None
    for p in candidate_paths:
        if p.exists():
            load_logs.append(f'Trying load: {p}')
            try:
                model, lg, att = load_siamese_model(p, debug=debug)
                load_logs.extend(lg)
                attempt_log.extend(att)
                chosen_model_path = p
                load_logs.append(f'Loaded model from {p}')
                break
            except Exception as ie:
                last_error = ie
                load_logs.append(f'Failed loading from {p}: {ie}')
        else:
            load_logs.append(f'Candidate missing: {p}')
    if model is None:
        err_msg = f"Model not found. Checked: {[str(c) for c in candidate_paths]}"
        if last_error:
            err_msg += f" Last error: {last_error}"
        print(json.dumps({"ok": False, "error": err_msg, "stage": "model_load", "debug": load_logs}))
        return

    runtime_logs = []
    def rlog(msg):
        if debug:
            runtime_logs.append(msg)

    # Allow multi-type (comma separated) e.g. "cat,dog" to union both sets
    requested_types = [t.strip() for t in pet_type.split(',') if t.strip()]
    resolved_subdirs = []
    for t in requested_types:
        if t.startswith('cat'):
            sub = pre_dir / 'Cats'
        elif t.startswith('dog'):
            sub = pre_dir / 'Dogs'
        else:
            continue
        if sub.exists():
            resolved_subdirs.append((t, sub))
    if not resolved_subdirs:
        print(json.dumps({"ok": True, "matches": [], "warning": "No valid pet subsets resolved", "debug": load_logs}))
        return

    # Determine if first arg is a path
    query_image_mode = None
    query_image_path = None
    try:
        inp_path = Path(processed_input)
        if inp_path.exists() and inp_path.is_file():
            query_image_mode = 'file'
            query_image_path = str(inp_path.resolve())
            rlog("Loading query image from file path (assumed preprocessed)")
            query_arr = preprocess_img(inp_path)
        else:
            query_image_mode = 'base64'
            rlog("Decoding query image from base64 input (will preprocess in-memory)")
            query_arr = preprocess_bytes(processed_input)
        query_emb = compute_embedding(model, query_arr)
        rlog(f"Query embedding shape: {query_emb.shape}")
        if query_emb.ndim == 0 or (query_emb.ndim == 1 and query_emb.shape[0] == 1):
            raise ValueError("Model produced a scalar distance instead of an embedding. Embedding extraction failed.")
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Query embedding failed: {e}", "stage": "query_embedding", "debug": load_logs + runtime_logs, "query_image_mode": query_image_mode, "query_image_path": query_image_path}))
        return

    scored: List[Tuple[str, float, str]] = []  # (path, sim, subset)
    for label, subset_dir in resolved_subdirs:
        imgs = list_images(subset_dir)
        rlog(f"Images found in subset {subset_dir}: {len(imgs)}")
        for img_path in imgs:
            try:
                arr = preprocess_img(img_path)
                emb = compute_embedding(model, arr)
                sim = cosine_similarity(query_emb, emb)
                scored.append((str(img_path), sim, label))
            except Exception as e:
                rlog(f"Skipping image {img_path}: {e}")
                continue

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    # Prepare matches (encode thumbnails base64)
    out_matches: List[Dict] = []
    for rank, (path_str, sim, subset_label) in enumerate(top, start=1):
        try:
            with open(path_str, 'rb') as f:
                b64_thumb = base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            b64_thumb = None
        out_matches.append({
            "rank": rank,
            "path": path_str,
            "score": round(sim * 100, 2),
            "thumb_base64": b64_thumb,
            "subset": subset_label
        })

    print(json.dumps({
        "ok": True,
        "matches": out_matches,
        "count": len(out_matches),
        "query_image_mode": query_image_mode,
        "query_image_path": query_image_path,
        "embedding_dim": int(query_emb.shape[0]) if hasattr(query_emb, 'shape') and query_emb.ndim == 1 else None,
        "debug": load_logs + runtime_logs,
        "attempts": attempt_log
    }))


if __name__ == '__main__':
    main()

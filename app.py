# ==========================================================
# PHASE 3 — Pattern Detection & Validation API
# ==========================================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import uvicorn
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List
import uvicorn
import cv2
import numpy as np
import base64
import io
import os
import tempfile
import uuid
import logging
import time
import zipfile
from PIL import Image
from blue_point_detector import detect_blue_points, draw_detected_points

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None


# ==================== CONFIGURATION ====================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = None
IMAGE_SIZE          = 256
PIXELS_PER_CM       = 100
CONFIDENCE_THRESHOLD = 0.25
MASK_THRESHOLD      = 0.5
MODEL_CACHE_DIR     = os.path.join(BASE_DIR, ".model_cache")

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== FASTAPI APP ====================
app = FastAPI(title="Phase 3 — Blue Point Validation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== LOAD MODEL ====================
model = None


def _is_supported_model_file(path: str) -> bool:
    supported_suffixes = (
        ".pt",
        ".torchscript",
        ".onnx",
        ".engine",
        ".mlpackage",
        ".tflite",
        ".pb",
        ".mnn",
        ".ncnn",
        ".rknn",
        ".bin",
    )
    return os.path.isfile(path) and path.lower().endswith(supported_suffixes)


def _find_torch_archive_root(path: str) -> str | None:
    required_files = {"data.pkl", "version", "byteorder"}
    if not os.path.isdir(path):
        return None

    direct_files = set(os.listdir(path))
    if required_files.issubset(direct_files):
        return path

    for entry in os.listdir(path):
        candidate = os.path.join(path, entry)
        if not os.path.isdir(candidate):
            continue
        candidate_files = set(os.listdir(candidate))
        if required_files.issubset(candidate_files):
            return candidate
    return None


def _latest_mtime(path: str) -> float:
    latest = os.path.getmtime(path)
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            latest = max(latest, os.path.getmtime(root))
            for name in files:
                latest = max(latest, os.path.getmtime(os.path.join(root, name)))
    return latest


def _repack_unpacked_torch_model(model_dir: str) -> str:
    archive_root = _find_torch_archive_root(model_dir)
    if archive_root is None:
        raise FileNotFoundError(f"No unpacked torch archive found under {model_dir}")

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    output_name = f"{os.path.basename(model_dir.rstrip(os.sep))}_restored.pt"
    output_path = os.path.join(MODEL_CACHE_DIR, output_name)

    if os.path.exists(output_path) and os.path.getmtime(output_path) >= _latest_mtime(model_dir):
        return output_path

    archive_parent = os.path.dirname(archive_root)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for root, _, files in os.walk(archive_root):
            for name in sorted(files):
                full_path = os.path.join(root, name)
                relative_path = os.path.relpath(full_path, archive_parent)
                file_mtime = max(os.path.getmtime(full_path), 315532800)
                zip_info = zipfile.ZipInfo(
                    relative_path,
                    date_time=time.localtime(file_mtime)[:6],
                )
                zip_info.compress_type = zipfile.ZIP_STORED
                with open(full_path, "rb") as src:
                    archive.writestr(zip_info, src.read())

    logger.warning("Rebuilt unpacked torch model from %s to %s", model_dir, output_path)
    return output_path


def resolve_model_path() -> str:
    env_model_path = os.getenv("MODEL_PATH")
    candidates = [
        env_model_path,
        os.path.join(BASE_DIR, "best.pt"),
        os.path.join(BASE_DIR, "best.onnx"),
        os.path.join(BASE_DIR, "best.torchscript"),
        os.path.join(PROJECT_ROOT, "best.pt"),
        os.path.join(PROJECT_ROOT, "best.onnx"),
        os.path.join(PROJECT_ROOT, "best.torchscript"),
        os.path.join(PROJECT_ROOT, "model_phase_2", "best.pt"),
        os.path.join(PROJECT_ROOT, "model_phase_2", "best.onnx"),
        os.path.join(PROJECT_ROOT, "model_phase_2", "best.torchscript"),
    ]

    for candidate in candidates:
        if candidate and _is_supported_model_file(candidate):
            return candidate

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            archive_root = _find_torch_archive_root(candidate)
            if archive_root is not None:
                return _repack_unpacked_torch_model(candidate)

    raise FileNotFoundError(
        "Could not find YOLO weights in a supported file format. Set MODEL_PATH or place "
        "best.pt in model_phase_3, the project root, or model_phase_2."
    )

def load_model_on_startup():
    global model, MODEL_PATH
    try:
        if YOLO is None:
            raise ModuleNotFoundError(
                "ultralytics is not installed. Install dependencies from requirements.txt."
            )

        MODEL_PATH = resolve_model_path()
        model = YOLO(MODEL_PATH, task="segment")
        logger.info(f"✓ YOLO model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        model = None
        MODEL_PATH = None

load_model_on_startup()

# ==================== SESSION STORE ====================
sessions: Dict[str, Any] = {}

# ==================== PYDANTIC MODELS ====================
class Point(BaseModel):
    x: float
    y: float

class MirrorRequest(BaseModel):
    session_id: str
    right_ear_points: List[Point]
    piercing_type: str = None


# ==================== UTILITIES ====================

def _round_float(value: float, ndigits: int) -> float:
    multiplier = 10 ** ndigits
    return float(int(value * multiplier + 0.5)) / multiplier


def image_to_base64(image_array):
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        rgb = image_array
    pil_image = Image.fromarray(rgb)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def segment_and_normalize(image_array):
    """YOLO-segment the ear and normalize to IMAGE_SIZE x IMAGE_SIZE."""
    if model is None:
        raise Exception("YOLO model not loaded")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image_array)
        results = model.predict(source=tmp_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        r = results[0]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if r.masks is None or len(r.masks.data) == 0:
        raise ValueError("No ear detected in the image")

    mask = r.masks.data[0].cpu().numpy()
    mask = (mask > MASK_THRESHOLD).astype(np.uint8)
    mask = cv2.resize(mask, (image_array.shape[1], image_array.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

    segmented = image_array.copy()
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return cv2.resize(segmented, (IMAGE_SIZE, IMAGE_SIZE))

    cy, cx = coords.mean(axis=0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    dist_y     = max(abs(y_max - cy), abs(cy - y_min))
    dist_x     = max(abs(x_max - cx), abs(cx - x_min))
    max_radius = max(dist_y, dist_x)
    max_dim    = int(max_radius * 2 * 1.15)
    half_dim   = max_dim // 2

    x1, y1 = int(cx - half_dim), int(cy - half_dim)
    x2, y2 = x1 + max_dim,       y1 + max_dim

    img_h, img_w = segmented.shape[:2]
    safe_x1, safe_y1 = max(0, x1), max(0, y1)
    safe_x2, safe_y2 = min(img_w, x2), min(img_h, y2)
    cropped = segmented[safe_y1:safe_y2, safe_x1:safe_x2]

    pad_top    = max(0, -y1)
    pad_bottom = max(0, y2 - img_h)
    pad_left   = max(0, -x1)
    pad_right  = max(0, x2 - img_w)
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom,
                                     pad_left, pad_right, cv2.BORDER_REPLICATE)

    ch, cw = cropped.shape[:2]
    if ch != cw:
        dim = max(ch, cw)
        cropped = cv2.copyMakeBorder(cropped, 0, dim - ch, 0, dim - cw, cv2.BORDER_REPLICATE)

    return cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


def draw_landmarks_with_lines(image, points,
                               color=(240, 248, 255), # AliceBlue (Pearl)
                               text_color=(255, 255, 255), # White
                               line_color=(220, 220, 220), # Silver
                               is_closed=False,
                               is_dashed=False,
                               label_side="right",
                               is_gold=False):
    img = image.copy()
    num_pts = len(points)
    for i in range(num_pts):
        if i < num_pts - 1:
            pt1 = (int(points[i][0]),   int(points[i][1]))
            pt2 = (int(points[i+1][0]), int(points[i+1][1]))
            
            if is_dashed:
                # Custom dashed line
                dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                dash_len = 8
                if dist > dash_len:
                    num_segments = int(dist / dash_len)
                    for j in range(num_segments):
                        if j % 2 == 0:
                            s = j / num_segments
                            e = (j + 1) / num_segments
                            p1 = (int(pt1[0] + (pt2[0] - pt1[0]) * s), int(pt1[1] + (pt2[1] - pt1[1]) * s))
                            p2_sub = (int(pt1[0] + (pt2[0] - pt1[0]) * e), int(pt1[1] + (pt2[1] - pt1[1]) * e))
                            cv2.line(img, p1, p2_sub, line_color, 1) # Thinner professional line
                else:
                    cv2.line(img, pt1, pt2, line_color, 1)
            else:
                if not is_gold:
                    cv2.line(img, pt1, pt2, line_color, 1)
                
        elif is_closed and num_pts > 2:
            pt1 = (int(points[i][0]),     int(points[i][1]))
            pt2 = (int(points[0][0]),     int(points[0][1]))
            cv2.line(img, pt1, pt2, line_color, 2)
            
    for i, (x, y) in enumerate(points):
        if is_gold:
            # Draw elegant silver/white stud
            cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), -1) # White in BGR
            cv2.circle(img, (int(x), int(y)), 5, (180, 180, 180), 1)   # Silver border
            # Tiny highlight
            cv2.circle(img, (int(x)-1, int(y)-1), 1, (255, 255, 255), -1)
        else:
            # Draw elegant pearl landmark
            cv2.circle(img, (int(x), int(y)), 6, (255, 255, 255), -1)
            cv2.circle(img, (int(x), int(y)), 6, (200, 200, 200), 1)
        
        if not is_gold:
            # Offset labels based on which side they should appear
            off_x = 10 if label_side == "right" else -24
            
            # Shadow for text
            cv2.putText(img, str(i + 1), (int(x) + off_x + 1, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(img, str(i + 1), (int(x) + off_x, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    return img


def detect_blue_markers_live(image):
    """
    Detect physical blue marker dots on a live/normalized ear image.
    Relaxed HSV range + morphological cleanup (no circularity filter).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 100, 70])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        if (4 * np.pi * area / (perimeter * perimeter)) < 0.2:
            continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            pts.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    pts.sort(key=lambda p: p[1])
    return pts


def get_point_guidance(digital, live):
    dx = digital[0] - live[0]
    dy = digital[1] - live[1]
    dist_px = float(np.hypot(dx, dy))
    tol   = 5
    mm_pp = 10.0 / PIXELS_PER_CM

    if dist_px <= tol:
        return "CORRECT ✓", _round_float(dist_px / PIXELS_PER_CM, 3)

    hdir, vdir = "", ""
    if   dx >  tol: hdir = f"RIGHT {abs(dx)*mm_pp:.1f}mm"
    elif dx < -tol: hdir = f"LEFT  {abs(dx)*mm_pp:.1f}mm"
    if   dy >  tol: vdir = f"DOWN  {abs(dy)*mm_pp:.1f}mm"
    elif dy < -tol: vdir = f"UP    {abs(dy)*mm_pp:.1f}mm"

    msg = " & ".join(filter(None, [hdir, vdir]))
    return f"Move → {msg}", _round_float(dist_px / PIXELS_PER_CM, 3)


# ==================== API ENDPOINTS ====================

@app.get("/")
def root():
    # Serve index.html as the root page
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "Phase 3 API is running", "model_loaded": model is not None}


# ----------------------------------------------------------
# STEP 1: Upload both ears → segment + check for blue points
# ----------------------------------------------------------
@app.post("/segment")
async def segment_ears(
    rightEar: UploadFile = File(...),
    leftEar:  UploadFile = File(...),
):
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    try:
        right_bytes = await rightEar.read()
        right_img   = cv2.imdecode(np.frombuffer(right_bytes, np.uint8), cv2.IMREAD_COLOR)
        if right_img is None:
            raise ValueError("Invalid right ear image")

        left_bytes = await leftEar.read()
        left_img   = cv2.imdecode(np.frombuffer(left_bytes, np.uint8), cv2.IMREAD_COLOR)
        if left_img is None:
            raise ValueError("Invalid left ear image")

        # Normalize both ears
        right_norm = segment_and_normalize(right_img)
        left_norm  = segment_and_normalize(left_img)

        # Blue point check on both ears
        right_pts, _ = detect_blue_points(right_norm, min_area=20)
        left_pts,  _ = detect_blue_points(left_norm,  min_area=20)

        # Annotate with detected blue points (if any)
        right_annotated = draw_detected_points(right_norm, right_pts) if right_pts else right_norm
        left_annotated  = draw_detected_points(left_norm,  left_pts)  if left_pts  else left_norm

        # Create session
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "right_ear": right_annotated,
            "left_ear":  left_annotated,
        }
        logger.info(f"Session {session_id}: segmented. Right blue pts: {len(right_pts)}, Left: {len(left_pts)}")

        return JSONResponse({
            "success": True,
            "data": {
                "session_id":              session_id,
                "image_size":              IMAGE_SIZE,
                "right_ear_image":         image_to_base64(right_annotated),
                "left_ear_image":          image_to_base64(left_annotated),
                "right_blue_points":       len(right_pts),
                "right_blue_points_coords": [{"x": float(p[0]), "y": float(p[1])} for p in right_pts],
                "left_blue_points":        len(left_pts),
                "has_blue_points":         len(right_pts) > 0 or len(left_pts) > 0,
            }
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Segment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------
# STEP 3: Mirror clicked right-ear points → left ear
# ----------------------------------------------------------
@app.post("/mirror-and-measure")
async def mirror_and_measure(request: MirrorRequest):
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload images first.")

    session   = sessions[session_id]
    right_ear = session["right_ear"]
    left_ear  = session["left_ear"]

    right_points = [(p.x, p.y) for p in request.right_ear_points]
    if len(right_points) < 1:
        raise HTTPException(status_code=400, detail="At least 1 point is required")

    # Mirror horizontally for left ear
    left_points = [
        (_round_float(float(IMAGE_SIZE - x), 1), _round_float(float(y), 1))
        for x, y in right_points
    ]

    # Draw on both ears
    is_triangle  = request.piercing_type == "triangle"
    is_snakebite = request.piercing_type == "snakebite"
    is_dashed    = request.piercing_type == "snake_curve"
    
    # Determine which side labels should go based on point positions (favor 'outer' side)
    avg_x = np.mean([p[0] for p in right_points])
    # If points are on the right (avg_x > 128), labels go to the RIGHT (outer edge)
    r_label_side = "right" if avg_x > IMAGE_SIZE / 2 else "left"
    # Left ear is mirrored, so labels also flip side
    l_label_side = "left" if avg_x > IMAGE_SIZE / 2 else "right"

    right_with_landmarks = draw_landmarks_with_lines(right_ear, right_points, 
                                                     is_closed=is_triangle, is_dashed=is_dashed, 
                                                     label_side=r_label_side, is_gold=is_snakebite)
    left_with_landmarks  = draw_landmarks_with_lines(left_ear,  left_points,  
                                                     is_closed=is_triangle, is_dashed=is_dashed, 
                                                     label_side=l_label_side, is_gold=is_snakebite)

    # Calculate distances between points
    num_pts = len(left_points)
    distances = []
    
    # 1 -> 2, 2 -> 3, etc.
    for i in range(num_pts - 1):
        p1 = np.array(left_points[i])
        p2 = np.array(left_points[i + 1])
        px  = float(np.linalg.norm(p1 - p2))
        cm  = px / PIXELS_PER_CM
        distances.append({
            "from_point":       i + 1,
            "to_point":         i + 2,
            "distance_pixels":  _round_float(px, 3),
            "distance_cm":      _round_float(cm, 3),
        })

    # For triangle, add 3 -> 1
    if is_triangle and num_pts == 3:
        p1 = np.array(left_points[2])
        p0 = np.array(left_points[0])
        px = float(np.linalg.norm(p1 - p0))
        cm = px / PIXELS_PER_CM
        distances.append({
            "from_point":       3,
            "to_point":         1,
            "distance_pixels":  _round_float(px, 3),
            "distance_cm":      _round_float(cm, 3),
        })

    total_px = sum(d["distance_pixels"] for d in distances)
    total_cm = sum(d["distance_cm"]     for d in distances)

    # Save in session for /validate-frame
    sessions[session_id]["right_points"]  = right_points
    sessions[session_id]["left_points"]   = left_points
    sessions[session_id]["piercing_type"] = request.piercing_type

    logger.info(f"Session {session_id}: {len(right_points)} landmarks mirrored. Piercing: {request.piercing_type}")

    return JSONResponse({
        "success": True,
        "data": {
            "session_id":   session_id,
            "image_size":   IMAGE_SIZE,
            "pixels_per_cm": PIXELS_PER_CM,
            "right_ear": {
                "points":          [{"x": x, "y": y} for x, y in right_points],
                "landmarks_image": image_to_base64(right_with_landmarks),
            },
            "left_ear": {
                "points":          [{"x": x, "y": y} for x, y in left_points],
                "landmarks_image": image_to_base64(left_with_landmarks),
            },
            "distances": distances,
            "total_distance": {
                "pixels": _round_float(total_px, 3),
                "cm":     _round_float(total_cm, 3),
            },
        }
    })


# ----------------------------------------------------------
# STEP 4: Live webcam frame → validate blue markers
# ----------------------------------------------------------
@app.post("/validate-frame")
async def validate_frame(
    file:       UploadFile = File(...),
    session_id: str        = Form(...),
    ear_side:   str        = Form("left"),
):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    if "right_points" not in session:
        raise HTTPException(status_code=400,
                            detail="No digital points saved. Complete Mirror & Measure first.")

    digital_pts = session["left_points"] if ear_side == "left" else session["right_points"]

    contents = await file.read()
    frame    = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image frame")

    # Normalize ear from live frame
    try:
        ear = segment_and_normalize(frame)
    except Exception:
        ear = None

    if ear is None:
        return JSONResponse({"success": True, "ear_detected": False,
                             "guidance": [], "annotated_image": None})

    # Detect physical blue markers on the live ear
    live_pts  = detect_blue_markers_live(ear)
    annotated = ear.copy()

    # Draw digital target rings (black ring + number)
    for i, dp in enumerate(digital_pts):
        cv2.circle(annotated, (int(dp[0]), int(dp[1])), 7, (0, 0, 0), 2)
        cv2.putText(annotated, str(i + 1),
                    (int(dp[0]) + 8, int(dp[1]) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)

    # Match each digital point to nearest un-used live marker
    used    = set()
    results: List[Dict[str, Any]] = []

    for i, dp in enumerate(digital_pts):
        best_d, best_j = 9999, -1
        for j, lp in enumerate(live_pts):
            if j in used:
                continue
            d = np.hypot(lp[0] - dp[0], lp[1] - dp[1])
            if d < best_d:
                best_d, best_j = d, j

        if best_j != -1 and best_d < IMAGE_SIZE * 0.4:
            used.add(best_j)
            lp = live_pts[best_j]
            msg, err_cm = get_point_guidance(dp, lp)
            correct = "CORRECT" in msg
            color   = (0, 220, 80) if correct else (255, 120, 0)

            # Draw live marker (solid blue + white outline)
            cv2.circle(annotated, (int(lp[0]), int(lp[1])), 6, (255, 50, 50), -1)
            cv2.circle(annotated, (int(lp[0]), int(lp[1])), 6, (255, 255, 255), 1)

            # Arrow: live marker → digital target
            cv2.arrowedLine(annotated,
                            (int(lp[0]), int(lp[1])),
                            (int(dp[0]), int(dp[1])),
                            (0, 255, 0), 1, tipLength=0.3)

            # Short guidance text on image
            parts         = msg.split("→")
            guidance_part = str(parts[-1]) if parts else ""
            short         = "OK" if correct else str(guidance_part).strip()[:14]
            cv2.putText(annotated, short,
                        (int(lp[0]) + 8, int(lp[1]) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            results.append({
                "point": i + 1, "message": msg,
                "error_cm": err_cm, "correct": correct,
                "live": [lp[0], lp[1]], "digital": [dp[0], dp[1]]
            })
        else:
            results.append({
                "point": i + 1, "message": "No marker detected",
                "error_cm": None, "correct": False,
                "live": None, "digital": [dp[0], dp[1]]
            })

    # Overall accuracy
    valid = [r for r in results if r["error_cm"] is not None]
    overall_accuracy = 0.0
    if valid:
        avg_err_mm = np.mean([r["error_cm"] * 10 for r in valid])
        overall_accuracy = max(0.0, _round_float(100.0 - float(avg_err_mm) * 10.0, 1))

    return JSONResponse({
        "success":      True,
        "ear_detected": True,
        "ear_side":     ear_side,
        "guidance":     results,
        "summary": {
            "overall_accuracy":   overall_accuracy,
            "detected_markers":   len(used),
            "total_points":       len(digital_pts),
            "status": ("Excellent" if overall_accuracy > 90
                       else "Good" if overall_accuracy > 70
                       else "Needs Work"),
        },
        "annotated_image": image_to_base64(annotated),
    })


# ----------------------------------------------------------
# Session management
# ----------------------------------------------------------
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id in sessions:
        return {"success": True, "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        sessions.pop(session_id)
        return {"success": True, "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


# ==================== RUN SERVER ====================
if __name__ == "__main__":
    print("Phase 3 Blue Point Validation API on http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

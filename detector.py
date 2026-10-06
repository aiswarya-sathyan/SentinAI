from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import librosa
    _LIBROSA_OK = True
except Exception:  # pragma: no cover
    librosa, _LIBROSA_OK = None, False

try:
    import cv2
    _CV_OK = True
except Exception:  # pragma: no cover
    cv2, _CV_OK = None, False

try:
    import mediapipe as mp
    _MP_OK = hasattr(mp, "solutions")
except Exception:  # pragma: no cover
    mp, _MP_OK = None, False

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
LATENCY_BUDGET_MS = 180
SR = 16000
N_FFT, HOP = 1024, 160            # 64 ms window, 10 ms hop (feature frames)
MEL_HOP, N_MELS = 256, 64         # spectrogram display
MIN_AUDIO_S, MAX_AUDIO_S = 0.75, 15.0
MAX_IMG_SIDE = 640
AUX_GAIN = 0.20                   # max share of the remaining headroom an aux cue group can add

THRESH_SYNTHETIC, THRESH_SUSPICIOUS, MIN_CONFIDENCE = 0.60, 0.35, 0.25

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
             378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]
R_EYE_OUT, R_EYE_IN, L_EYE_OUT, L_EYE_IN = 33, 133, 263, 362
NOSE_TIP, CHEEK_L, CHEEK_R = 1, 234, 454

AUDIO_WEIGHTS = {"pitch_flat": 0.20, "jitter_low": 0.15, "centroid_flat": 0.15,
                 "zcr_anomaly": 0.10, "breath_missing": 0.20, "hnr_high": 0.20}
AUDIO_AUX_WEIGHTS = {"pause_clean": 0.6, "pause_regular": 0.4}
AUDIO_FLAGS = {
    "pitch_flat": "Pitch contour unnaturally flat (low F0 deviation)",
    "jitter_low": "Cycle-to-cycle pitch jitter is too smooth",
    "centroid_flat": "Spectral centroid barely moves (monotone timbre)",
    "zcr_anomaly": "Zero-crossing rate outside natural speech range or splice-like jumps",
    "breath_missing": "No micro-breaths in a clip long enough to expect them",
    "hnr_high": "Voice is unnaturally clean (harmonics-to-noise ratio very high)",
    "pause_clean": "Pauses are near-digital silence (no room / mic noise floor)",
    "pause_regular": "Pauses are suspiciously evenly spaced (scripted rhythm)",
}
AUDIO_LABELS = {
    "pitch_flat": "Pitch flatness", "jitter_low": "Pitch jitter too smooth",
    "centroid_flat": "Monotone timbre", "zcr_anomaly": "ZCR anomaly",
    "breath_missing": "Missing breaths", "hnr_high": "Over-clean voice (HNR)",
    "pause_clean": "Digital-silence pauses", "pause_regular": "Regular pause rhythm",
    "tts": "Neural-TTS signature",
}
VIDEO_WEIGHTS = {"eye": 0.25, "boundary": 0.25, "edge": 0.35, "texture": 0.15}
VIDEO_AUX_WEIGHTS = {"color": 0.5, "freq": 0.5}
VIDEO_FLAGS = {
    "eye": "Eye-distance ratio / eye symmetry outside natural facial proportions",
    "boundary": "Face-contour mesh is jagged (boundary jitter)",
    "edge": "Face-mask edge looks blended: seam is smoother than its surroundings",
    "texture": "Skin texture is over-smoothed (can also be a noisy-webcam artefact)",
    "color": "Skin tone of the face differs from the neck / surrounding skin",
    "freq": "Face-region frequency spectrum has synthesis artefacts (over-smoothed or periodic)",
    "flicker": "Landmarks flicker frame-to-frame (unstable face swap)",
    "seam_flicker": "Mask-edge strength fluctuates between frames",
    "dropout": "Face tracking drops out intermittently",
}
VIDEO_LABELS = {
    "eye": "Eye geometry", "boundary": "Contour jitter", "edge": "Mask-edge blending",
    "texture": "Over-smooth skin", "color": "Skin-tone mismatch", "freq": "Spectral artefacts",
    "flicker": "Landmark flicker", "seam_flicker": "Seam flicker", "dropout": "Face dropout",
}
TEMPORAL_WEIGHTS = {"flicker": 0.4, "seam_flicker": 0.35, "dropout": 0.25}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def ramp(x: Optional[float], lo: float, hi: float) -> float:
    """Linear 0 -> 1 as x moves from lo to hi (works when lo > hi). Non-finite -> 0."""
    if x is None or not np.isfinite(x) or lo == hi:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _rr(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    """Like ramp() but returns None when the metric is unavailable."""
    return None if (x is None or not np.isfinite(x)) else ramp(x, lo, hi)


def _wavg(pairs) -> float:
    tot = sum(w for _, w in pairs)
    return float(sum(v * w for v, w in pairs) / tot) if tot > 0 else 0.0


def _py(v: Any) -> Any:
    """numpy -> plain python; NaN/inf -> None (JSON/UI friendly)."""
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _verdict(risk: float, conf: float) -> str:
    # Lower bound 0.40 so compressed real voice notes do not trigger 'SUSPICIOUS'
    if risk >= 0.60 and conf >= 0.5:
        return "SYNTHETIC"
    elif risk >= 0.40 and conf >= 0.3:
        return "SUSPICIOUS"
    return "AUTHENTIC"


def _empty(kind: str, error: Optional[str] = None) -> Dict[str, Any]:
    return {"ok": False, "kind": kind, "error": error, "verdict": "INCONCLUSIVE",
            "risk": 0.0, "confidence": 0.0, "latency_ms": 0.0, "metrics": {},
            "subscores": {}, "flags": [], "notes": [], "spectrogram": None,
            "annotated_png": None, "contributions": {}, "timeline": []}


def _combine(core, aux, gain: float = AUX_GAIN) -> Tuple[float, Dict[str, float]]:
    """core/aux: lists of (label, value|None, weight).
    -> (risk, contributions) where contributions sum exactly to risk."""
    core = [(l, v, w) for l, v, w in core if v is not None]
    tot = sum(w for _, _, w in core)
    contrib = {l: v * w / tot for l, v, w in core} if tot > 0 else {}
    base = float(sum(contrib.values()))
    aux = [(l, v, w) for l, v, w in aux if v is not None]
    atot = sum(w for _, _, w in aux)
    bonus = 0.0
    if atot > 0:
        aavg = sum(v * w for _, v, w in aux) / atot
        bonus = gain * aavg * (1.0 - base)
        s = sum(v * w for _, v, w in aux)
        for l, v, w in aux:
            contrib[l] = bonus * (v * w / s) if s > 0 else 0.0
    return float(base + bonus), contrib


def add_noise(y: np.ndarray, snr_db: float, seed: int = 7) -> np.ndarray:
    """Robustness test: add white noise at a given SNR."""
    rng = np.random.default_rng(seed)
    p = float(np.mean(y ** 2)) + 1e-12
    n = rng.standard_normal(len(y)).astype(np.float32)
    n *= np.sqrt(p / (10 ** (snr_db / 10.0)) / (float(np.mean(n ** 2)) + 1e-12))
    return (y + n).astype(np.float32)


def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    """Robustness test: re-compress a frame at a lower JPEG quality."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else None
    return out if out is not None else img


def engine_status() -> Dict[str, bool]:
    return {"librosa (audio)": _LIBROSA_OK, "opencv (video)": _CV_OK,
            "mediapipe (face mesh)": _MP_OK}


_FACE_MESH = None


def _get_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5)
    return _FACE_MESH


def _mesh_landmarks(img: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    if not _MP_OK:
        return None, "MediaPipe not installed."
    try:
        h, w = img.shape[:2]
        res = _get_mesh().process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            return np.array([(p.x * w, p.y * h, p.z * w)
                             for p in res.multi_face_landmarks[0].landmark], np.float32), ""
        return None, "FaceMesh found no face."
    except Exception as exc:
        return None, f"FaceMesh failed ({exc})."


def _fit(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMG_SIDE:
        s = MAX_IMG_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def warm_up() -> None:
    """Pre-load models / JIT so the first real call is not penalised in the latency card."""
    try:
        if _MP_OK:
            _get_mesh().process(np.zeros((96, 96, 3), np.uint8))
        if _LIBROSA_OK:
            analyze_audio_array(np.random.randn(SR) * 0.01, SR)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# AUDIO
# --------------------------------------------------------------------------- #
def analyze_audio_stream(file_path: str, noise_snr_db: Optional[float] = None) -> Dict[str, Any]:
    """Analyze a recorded/uploaded audio file. Never raises."""
    t0 = time.perf_counter()
    if not _LIBROSA_OK:
        res = _empty("audio", "librosa is not installed (pip install librosa).")
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True, duration=MAX_AUDIO_S)
    except Exception as exc:
        res = _empty("audio", f"Could not read audio file: {exc}")
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res
    return analyze_audio_array(y, sr, _t0=t0, noise_snr_db=noise_snr_db)


def analyze_audio_array(y: np.ndarray, sr: int, _t0: Optional[float] = None,
                        noise_snr_db: Optional[float] = None) -> Dict[str, Any]:
    """Same analysis on an in-memory buffer (use this for WebSocket / streaming chunks)."""
    t0 = _t0 if _t0 is not None else time.perf_counter()

    def done(r: Dict[str, Any]) -> Dict[str, Any]:
        r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return r

    if not _LIBROSA_OK:
        return done(_empty("audio", "librosa is not installed (pip install librosa)."))
    try:
        y = np.nan_to_num(np.asarray(y, dtype=np.float32))
        if y.ndim > 1:
            y = y.mean(axis=-1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        dur = len(y) / SR
        if dur < MIN_AUDIO_S:
            return done(_empty("audio", f"Clip too short ({dur:.2f}s). Record 2+ seconds of speech."))
        if float(np.max(np.abs(y))) < 1e-4:
            return done(_empty("audio", "Audio is silent."))
        y = y / (np.max(np.abs(y)) + 1e-9)
        if noise_snr_db is not None:                     # robustness test
            y = add_noise(y, float(noise_snr_db))
            y = y / (np.max(np.abs(y)) + 1e-9)

        out = _empty("audio")
        notes: List[str] = []
        if noise_snr_db is not None:
            notes.append(f"Robustness test: white noise added at {noise_snr_db:.0f} dB SNR.")

        # --- mel-spectrogram payload for the UI -------------------------------- #
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=MEL_HOP,
                                             n_mels=N_MELS, fmax=8000)
        out["spectrogram"] = {"mel_db": librosa.power_to_db(mel, ref=np.max).astype(np.float32),
                              "sr": SR, "hop": MEL_HOP}

        # --- frame features -------------------------------------------------- #
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
        cent = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)[0]
        flat = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP)[0]
        try:
            f0 = librosa.yin(y, fmin=70, fmax=400, sr=SR, frame_length=N_FFT, hop_length=HOP)
        except Exception:
            f0 = None
            notes.append("Pitch tracker unavailable: spectral-only fallback scoring.")
        n = min(len(a) for a in (rms, zcr, cent, flat) + ((f0,) if f0 is not None else ()))
        rms, zcr, cent, flat = rms[:n], zcr[:n], cent[:n], flat[:n]
        f0 = f0[:n] if f0 is not None else None

        p95, p10 = np.percentile(rms, 95) + 1e-9, np.percentile(rms, 10) + 1e-9
        snr = float(np.clip(20 * np.log10(p95 / p10), 0, 60))      # crude SNR proxy
        voiced = (rms > 0.25 * p95) & (flat < 0.2)
        n_voiced = int(voiced.sum())
        if n_voiced < 10:
            out.update(error="Not enough voiced speech detected. Speak closer to the mic.",
                       metrics={"duration_s": dur, "snr_db": snr, "voiced_frames": n_voiced})
            return done(out)
        idx = np.where(voiced)[0]

        # pitch deviation + jitter proxy
        pitch_mean = pitch_dev = jitter = hnr = float("nan")
        if f0 is not None:
            f_v = f0[idx]
            pitch_mean = float(np.median(f_v))
            pitch_dev = float(np.std(12 * np.log2(f_v / pitch_mean)))      # semitones
            contiguous = np.diff(idx) == 1
            d = np.abs(np.diff(f_v))[contiguous]
            if d.size > 5:
                jitter = float(np.mean(d / f_v[:-1][contiguous]))
            hnr = _hnr_proxy(y, f0, idx)

        # spectral centroid distribution / transitions, ZCR continuity
        c = cent[idx]
        cent_mean = float(np.mean(c))
        cent_cv = float(np.std(c) / (cent_mean + 1e-9))
        cent_flux = float(np.mean(np.abs(np.diff(cent))))
        zcr_mean = float(np.mean(zcr[idx]))
        zcr_jump = float(np.percentile(np.abs(np.diff(zcr)), 99))

        # micro-breaths: low-energy, noise-like runs >= 80 ms
        quiet = (rms > 0.01 * p95) & (rms < 0.15 * p95) & (flat > 0.1)
        breaths = _count_runs(quiet, 8) / dur * 10

        # AUX evidence: pause noise floor + pause rhythm
        floor_gap_db = float(np.clip(20 * np.log10(p95 / (np.percentile(rms, 5) + 1e-7)), 0, 100))
        pauses = _run_lengths(rms < 0.15 * p95, 8)
        pause_cv = float(np.std(pauses) / (np.mean(pauses) + 1e-9)) if len(pauses) >= 4 else float("nan")

        sub = {
            "pitch_flat": ramp(pitch_dev, 1.8, 0.5) if np.isfinite(pitch_dev) else 0.0,
            "jitter_low": _rr(jitter, 0.008, 0.001) if np.isfinite(jitter) else 0.0,
            "centroid_flat": ramp(cent_flux, 150.0, 30.0) if np.isfinite(cent_flux) else 0.0,
            "zcr_anomaly": max(ramp(zcr_mean, 0.12, 0.02), ramp(zcr_jump, 0.15, 0.35)),
            "breath_missing": ramp(breaths, 0.5, 0.0) * ramp(dur, 3.0, 6.0),
            "hnr_high": _rr(hnr, 22.0, 32.0) if np.isfinite(hnr) else 0.0,
        }
        aux = {
            "pause_clean": ramp(floor_gap_db, 60.0, 80.0) * ramp(dur, 1.5, 3.0),
            "pause_regular": _rr(pause_cv, 0.35, 0.10),
        }

        weights = dict(AUDIO_WEIGHTS)
        if snr < 12:   # noise-aware weighting
            weights["breath_missing"] *= 0.3
            weights["hnr_high"] *= 0.3
            notes.append("Background noise is high: breath and HNR cues down-weighted.")
        aux_w = dict(AUDIO_AUX_WEIGHTS)
        if noise_snr_db is not None or snr < 12:
            aux_w["pause_clean"] *= 0.2                  # noisy input: floor cue is meaningless

        risk, contrib = _combine(
            [(AUDIO_LABELS[k], v, weights[k]) for k, v in sub.items()],
            [(AUDIO_LABELS[k], v, aux_w[k]) for k, v in aux.items()])
        sub.update(aux)

        conf = ramp(dur, 0.75, 3.0) * ramp(n_voiced, 10, 60) * (0.4 + 0.6 * ramp(snr, 3, 15))

        flags = [AUDIO_FLAGS[k] for k, v in sorted(sub.items(), key=lambda kv: -(kv[1] or 0))
                 if v is not None and v >= 0.5]

        # Targeted neural-TTS check (unnatural pitch regularity + stable centroid in clean audio)
        is_neural_tts = (
            np.isfinite(pitch_dev) and (3.5 <= pitch_dev <= 5.5) and
            np.isfinite(cent_cv) and (0.40 <= cent_cv <= 0.60) and
            snr > 18
        )
        if is_neural_tts and risk < 0.78:
            contrib[AUDIO_LABELS["tts"]] = 0.78 - risk
            risk = 0.78
            flags.append("Neural TTS Signature: Unnatural pitch/centroid stability detected")

        metrics = dict(pitch_mean_hz=pitch_mean, pitch_dev_st=pitch_dev, jitter=jitter,
                       centroid_mean_hz=cent_mean, centroid_cv=cent_cv, centroid_flux_hz=cent_flux,
                       zcr_mean=zcr_mean, zcr_jump=zcr_jump, breaths_per_10s=breaths, hnr_db=hnr,
                       snr_db=snr, duration_s=dur, voiced_frames=n_voiced,
                       pause_floor_gap_db=floor_gap_db, pause_cv=pause_cv)

        out.update(ok=True, verdict=_verdict(risk, conf), risk=float(risk), confidence=float(conf),
                   metrics={k: _py(v) for k, v in metrics.items()},
                   subscores={k: _py(v) for k, v in sub.items()}, flags=flags, notes=notes,
                   contributions={k: float(v) for k, v in contrib.items()})
        return done(out)
    except Exception as exc:  # never crash the UI
        return done(_empty("audio", f"Audio analysis failed: {exc}"))


def _run_lengths(mask: np.ndarray, min_len: int) -> List[int]:
    runs, run = [], 0
    for m in mask:
        if m:
            run += 1
        else:
            if run >= min_len:
                runs.append(run)
            run = 0
    if run >= min_len:
        runs.append(run)
    return runs


def _count_runs(mask: np.ndarray, min_len: int) -> int:
    return len(_run_lengths(mask, min_len))


def _hnr_proxy(y: np.ndarray, f0: np.ndarray, idx: np.ndarray) -> float:
    vals = []
    for i in idx[::4]:
        seg = y[i * HOP: i * HOP + 1024]
        if len(seg) < 1024:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        lag = int(SR / f0[i])
        if ac[0] <= 1e-9 or not (0 < lag < len(ac)):
            continue
        r = float(np.clip(ac[lag] / ac[0], 1e-3, 0.999))
        vals.append(10 * np.log10(r / (1 - r)))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# VIDEO: shared geometry
# --------------------------------------------------------------------------- #
def yaw_signed(lm: np.ndarray) -> float:
    """Signed head-yaw proxy. POSITIVE = nose displaced toward the image-RIGHT eye (landmark 263),
    NEGATIVE = toward the image-LEFT eye (landmark 33). ~0 when frontal."""
    eye_dist = float(np.linalg.norm(lm[R_EYE_OUT, :2] - lm[L_EYE_OUT, :2]))
    d_r = float(np.linalg.norm(lm[NOSE_TIP, :2] - lm[R_EYE_OUT, :2]))
    d_l = float(np.linalg.norm(lm[NOSE_TIP, :2] - lm[L_EYE_OUT, :2]))
    return (d_r - d_l) / (eye_dist + 1e-6)


def face_pose(image_bytes: bytes) -> Dict[str, Any]:
    """Landmarks + signed yaw for one image. Never raises. Used by liveness.py."""
    res: Dict[str, Any] = {"ok": False, "error": None}
    if not (_CV_OK and _MP_OK):
        res["error"] = "Liveness needs OpenCV and MediaPipe FaceMesh."
        return res
    try:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            res["error"] = "Could not decode the image."
            return res
        img = _fit(img)
        lm, reason = _mesh_landmarks(img)
        if lm is None:
            res["error"] = reason
            return res
        face_w = float(np.linalg.norm(lm[CHEEK_L, :2] - lm[CHEEK_R, :2]))
        res.update(ok=True, img=img, gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), lm=lm,
                   yaw_signed=yaw_signed(lm), face_w=face_w)
        return res
    except Exception as exc:
        res["error"] = f"Pose estimation failed: {exc}"
        return res


def _seam_metrics(gray: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """(seam_ratio, inside_texture_variance). Blended masks give a low seam ratio."""
    h, w = gray.shape
    k = max(3, int(0.03 * max(h, w))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner, outer = cv2.erode(mask, ker), cv2.dilate(mask, ker)
    ring = cv2.subtract(outer, inner)
    inside = cv2.erode(mask, ker, iterations=2)
    outside = cv2.subtract(cv2.dilate(mask, ker, iterations=2), outer)
    lap = cv2.Laplacian(gray, cv2.CV_32F)

    def var(m: np.ndarray) -> float:
        v = lap[m > 0]
        return float(v.var()) if v.size > 50 else float("nan")

    v_r, v_i, v_o = var(ring), var(inside), var(outside)
    ratio = v_r / (0.5 * (v_i + v_o) + 1e-6) if np.isfinite([v_r, v_i, v_o]).all() else float("nan")
    return ratio, v_i


def _color_mismatch(img: np.ndarray, mask: np.ndarray) -> float:
    """Chroma distance (Lab a/b) between skin inside the face oval and skin just outside it
    (neck / jaw line). Pasted faces rarely match the host's skin tone. NaN if too little skin."""
    h, w = mask.shape
    k = max(5, int(0.04 * max(h, w))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner = cv2.erode(mask, ker) > 0
    ring = (cv2.dilate(mask, ker, iterations=2) > 0) & ~(cv2.dilate(mask, ker) > 0)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycc, (0, 133, 77), (255, 173, 127)) > 0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_in, a_out = inner & skin, ring & skin
    if a_in.sum() < 300 or a_out.sum() < 150:
        return float("nan")
    d = np.median(lab[a_in][:, 1:3], axis=0) - np.median(lab[a_out][:, 1:3], axis=0)
    return float(np.hypot(d[0], d[1]))


def _freq_artifacts(gray: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """(hf_drop, bump_std) from the radially averaged log power spectrum of the face crop.
    Fits a 1/f power law; synthetic faces tend to under-shoot it at high frequency (over-smoothing)
    or show bumps (periodic up-sampling artefacts). NaN when the crop is too small to be reliable."""
    x, y, w, h = cv2.boundingRect(mask)
    crop = gray[y:y + h, x:x + w]
    if min(crop.shape) < 128:                      # never up-sample: it would fake an HF drop
        return float("nan"), float("nan")
    crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    crop -= crop.mean()
    crop *= np.outer(np.hanning(128), np.hanning(128)).astype(np.float32)
    p = np.log(np.abs(np.fft.fftshift(np.fft.fft2(crop))) ** 2 + 1e-6)
    yy, xx = np.indices(p.shape)
    r = np.hypot(yy - 64, xx - 64).astype(int)
    prof = np.bincount(r.ravel(), p.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    rr = np.arange(4, 64)
    A = np.vstack([np.log(rr), np.ones_like(rr, dtype=float)]).T
    coef, *_ = np.linalg.lstsq(A, prof[4:64], rcond=None)
    resid = prof[4:64] - A @ coef
    hf = resid[rr >= 40]
    return float(-hf.mean()), float(hf.std())


def _oval_mask(gray: np.ndarray, lm: np.ndarray) -> np.ndarray:
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(lm[FACE_OVAL, :2].astype(np.int32)), 255)
    return mask


def _encode_png(img: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None


# --------------------------------------------------------------------------- #
# VIDEO: single frame
# --------------------------------------------------------------------------- #
def analyze_video_frame(image_bytes: bytes, jpeg_quality: Optional[int] = None) -> Dict[str, Any]:
    """Analyze one captured frame (JPEG/PNG bytes). Never raises."""
    t0 = time.perf_counter()
    if not _CV_OK:
        r = _empty("video", "OpenCV is not installed (pip install opencv-python).")
    else:
        try:
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            r = _empty("video", "Could not decode the image.") if img is None else _analyze_bgr(img, jpeg_quality)
        except Exception as exc:
            r = _empty("video", f"Video analysis failed: {exc}")
    r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return r


def _analyze_bgr(img: np.ndarray, jpeg_quality: Optional[int] = None) -> Dict[str, Any]:
    img = _fit(img)
    if jpeg_quality is not None:                      # robustness test
        img = jpeg_roundtrip(img, jpeg_quality)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lm, reason = _mesh_landmarks(img)
    res = _video_fallback(img, gray, reason) if lm is None else _video_mesh(img, gray, lm)
    if jpeg_quality is not None and res.get("ok"):
        res["notes"].append(f"Robustness test: frame re-compressed at JPEG quality {int(jpeg_quality)}.")
    return res


def _video_mesh(img: np.ndarray, gray: np.ndarray, lm: np.ndarray) -> Dict[str, Any]:
    out = _empty("video")
    notes: List[str] = []
    face_w = float(np.linalg.norm(lm[CHEEK_L, :2] - lm[CHEEK_R, :2])) + 1e-6

    # eye-distance ratio, eye symmetry, yaw proxy (to skip the check on turned heads)
    eye_dist = float(np.linalg.norm(lm[R_EYE_OUT, :2] - lm[L_EYE_OUT, :2]))
    eye_ratio = eye_dist / face_w
    w_r = np.linalg.norm(lm[R_EYE_OUT, :2] - lm[R_EYE_IN, :2])
    w_l = np.linalg.norm(lm[L_EYE_OUT, :2] - lm[L_EYE_IN, :2])
    eye_sym = float(abs(w_r - w_l) / ((w_r + w_l) / 2 + 1e-6))
    yaw_s = yaw_signed(lm)
    yaw = abs(yaw_s)
    frontal = 1.0 - ramp(yaw, 0.15, 0.45)
    eye_risk = None
    if frontal >= 0.3:
        eye_risk = max(ramp(abs(eye_ratio - 0.62), 0.08, 0.18), ramp(eye_sym, 0.12, 0.35))
    else:
        notes.append("Head is turned: eye-geometry check skipped.")

    # mesh boundary jitter: residual of the face-oval contour vs. its smoothed version
    oval = lm[FACE_OVAL, :2]
    pad = np.concatenate([oval[-2:], oval, oval[:2]])
    smooth = np.stack([np.convolve(pad[:, c], np.ones(5) / 5, mode="valid") for c in (0, 1)], axis=1)
    jitter = float(np.mean(np.linalg.norm(oval - smooth, axis=1)) / face_w)

    # facial mask edge anomaly + texture + colour + spectrum
    mask = _oval_mask(gray, lm)
    seam_ratio, v_in = _seam_metrics(gray, mask)
    col = _color_mismatch(img, mask)
    hf_drop, bump = _freq_artifacts(gray, mask)

    sub = {"eye": eye_risk, "boundary": ramp(jitter, 0.010, 0.030),
           "edge": _rr(seam_ratio, 1.0, 0.35), "texture": _rr(v_in, 30.0, 6.0)}
    freq = None
    if np.isfinite(hf_drop) and np.isfinite(bump):
        freq = max(ramp(hf_drop, 1.5, 3.5), ramp(bump, 0.6, 1.2))
    aux = {"color": _rr(col, 8.0, 20.0), "freq": freq}

    risk, contrib = _combine(
        [(VIDEO_LABELS[k], v, VIDEO_WEIGHTS[k]) for k, v in sub.items()],
        [(VIDEO_LABELS[k], v, VIDEO_AUX_WEIGHTS[k]) for k, v in aux.items()])
    sub.update(aux)

    brightness = float(gray.mean())
    conf = ramp(face_w, 60, 160)
    if brightness < 45:
        conf *= 0.6
        notes.append("Low light: confidence reduced.")
    if face_w < 90:
        notes.append("Face is small in frame: move closer for a reliable read.")

    # annotated overlay (landmarks + contour)
    vis = img.copy()
    for x, y, _ in lm[:468]:
        cv2.circle(vis, (int(x), int(y)), 1, (129, 185, 16), -1)
    cv2.polylines(vis, [oval.astype(np.int32)], True, (153, 243, 52), 1, cv2.LINE_AA)

    metrics = dict(landmark_count=min(len(lm), 468), eye_distance_ratio=eye_ratio,
                   eye_symmetry=eye_sym, yaw_proxy=yaw, yaw_signed=yaw_s, boundary_jitter=jitter,
                   seam_ratio=seam_ratio, texture_var=v_in, skin_tone_delta=col,
                   spectrum_hf_drop=hf_drop, face_px=face_w, brightness=brightness,
                   engine="mediapipe-facemesh")
    flags = [VIDEO_FLAGS[k] for k, v in sorted(sub.items(), key=lambda kv: -(kv[1] or 0))
             if v is not None and v >= 0.5]
    out.update(ok=True, verdict=_verdict(risk, conf), risk=float(risk), confidence=float(conf),
               metrics={k: _py(v) for k, v in metrics.items()},
               subscores={k: _py(v) for k, v in sub.items()}, flags=flags, notes=notes,
               contributions={k: float(v) for k, v in contrib.items()},
               annotated_png=_encode_png(vis))
    return out


def _video_fallback(img: np.ndarray, gray: np.ndarray, reason: str) -> Dict[str, Any]:
    """Haar-cascade + mask-edge/texture heuristics when FaceMesh is unavailable or fails."""
    out = _empty("video")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = [] if cascade.empty() else cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        out["error"] = f"No face detected. {reason} Centre your face and improve lighting.".strip()
        return out
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mask = np.zeros(gray.shape, np.uint8)
    cv2.ellipse(mask, (x + w // 2, y + h // 2), (w // 2, int(h * 0.55)), 0, 0, 360, 255, -1)
    seam_ratio, v_in = _seam_metrics(gray, mask)
    col = _color_mismatch(img, mask)
    sub = {"edge": _rr(seam_ratio, 1.0, 0.35), "texture": _rr(v_in, 30.0, 6.0)}
    aux = {"color": _rr(col, 8.0, 20.0)}
    risk, contrib = _combine(
        [(VIDEO_LABELS[k], v, VIDEO_WEIGHTS[k]) for k, v in sub.items()],
        [(VIDEO_LABELS[k], v, VIDEO_AUX_WEIGHTS[k]) for k, v in aux.items()])
    sub.update(aux)
    conf = 0.5 * ramp(w, 60, 160)       # fallback evidence is thinner: cap confidence
    vis = img.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (129, 185, 16), 2)
    flags = [VIDEO_FLAGS[k] for k, v in sub.items() if v is not None and v >= 0.5]
    out.update(ok=True, verdict=_verdict(risk, conf), risk=float(risk), confidence=float(conf),
               metrics={"landmark_count": 0, "seam_ratio": _py(seam_ratio), "texture_var": _py(v_in),
                        "skin_tone_delta": _py(col), "face_px": int(w), "engine": "haar-fallback"},
               subscores={k: _py(v) for k, v in sub.items()}, flags=flags,
               contributions={k: float(v) for k, v in contrib.items()},
               notes=[f"Fallback heuristics in use. {reason}".strip()],
               annotated_png=_encode_png(vis))
    return out


# --------------------------------------------------------------------------- #
# VIDEO: file (keyframes + temporal window)
# --------------------------------------------------------------------------- #
def analyze_video_file(path: str, jpeg_quality: Optional[int] = None) -> Dict[str, Any]:
    """Analyze a video: 5 spread keyframes (full single-frame analysis) plus one second of
    consecutive frames in the middle for temporal cues (landmark flicker, seam flicker, dropout).
    Face swaps are much easier to catch over time than in one still. Never raises."""
    t0 = time.perf_counter()

    def done(r: Dict[str, Any]) -> Dict[str, Any]:
        r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return r

    if not _CV_OK:
        return done(_empty("video", "OpenCV is not installed (pip install opencv-python)."))
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if n <= 0:
            return done(_empty("video", "Could not read any frames from the video."))

        def read_at(i: int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            return f if ok and f is not None else None

        idxs = list(range(n)) if n <= 5 else \
            [int(v) for v in np.linspace(0.1 * (n - 1), 0.9 * (n - 1), 5)]
        keys = []
        for i in idxs:
            f = read_at(i)
            if f is not None:
                keys.append(_analyze_bgr(f, jpeg_quality))
        valid = [r for r in keys if r["ok"]]
        if not valid:
            err = next((r["error"] for r in keys if r.get("error")), "No usable frames.")
            return done(_empty("video", err))

        # ---- temporal window ------------------------------------------------ #
        win = min(30, n)
        start = max(0, n // 2 - win // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        shapes, seams, tried = [], [], 0
        for _ in range(win):
            ok, f = cap.read()
            if not ok or f is None:
                break
            tried += 1
            f = _fit(f)
            lm, _why = _mesh_landmarks(f)
            if lm is None:
                continue
            fw = float(np.linalg.norm(lm[CHEEK_L, :2] - lm[CHEEK_R, :2])) + 1e-6
            pts = lm[:468, :2]
            shapes.append((pts - pts.mean(axis=0)) / fw)
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            sr_, _ = _seam_metrics(g, _oval_mask(g, lm))
            if np.isfinite(sr_):
                seams.append(sr_)

        tsub: Dict[str, Optional[float]] = {}
        tmet: Dict[str, Any] = {"window_frames": tried, "window_s": tried / fps}
        if tried >= 10:
            if len(shapes) >= 5:
                X = np.stack(shapes)
                acc = X[2:] - 2 * X[1:-1] + X[:-2]
                flick = float(np.mean(np.linalg.norm(acc, axis=2)))
                tmet["landmark_flicker"] = flick
                tsub["flicker"] = ramp(flick, 0.0035, 0.010)
            if len(seams) >= 5:
                cv = float(np.std(seams) / (np.mean(seams) + 1e-6))
                tmet["seam_cv"] = cv
                tsub["seam_flicker"] = ramp(cv, 0.15, 0.45)
            drop = 1.0 - len(shapes) / max(tried, 1)
            tmet["face_dropout"] = drop
            tsub["dropout"] = ramp(drop, 0.10, 0.40)

        # ---- aggregate -------------------------------------------------------- #
        risks = [r["risk"] for r in valid]
        agg = 0.5 * float(np.median(risks)) + 0.5 * float(np.percentile(risks, 75))
        conf = float(np.mean([r["confidence"] for r in valid])) * (len(valid) / max(len(keys), 1))

        labels = {l for r in valid for l in r["contributions"]}
        mean_c = {l: float(np.mean([r["contributions"].get(l, 0.0) for r in valid])) for l in labels}
        s = sum(mean_c.values())
        contrib = {l: v * agg / s for l, v in mean_c.items()} if s > 1e-9 else {}

        tpairs = [(VIDEO_LABELS[k], v, TEMPORAL_WEIGHTS[k]) for k, v in tsub.items() if v is not None]
        risk = agg
        if tpairs:
            tavg = sum(v * w for _, v, w in tpairs) / sum(w for _, _, w in tpairs)
            bonus = 0.25 * tavg * (1.0 - agg)
            ssum = sum(v * w for _, v, w in tpairs)
            for l, v, w in tpairs:
                contrib[l] = bonus * (v * w / ssum) if ssum > 0 else 0.0
            risk = agg + bonus

        rep = max(valid, key=lambda r: r["risk"])
        keysub: Dict[str, Optional[float]] = {}
        for r in valid:
            for k, v in r["subscores"].items():
                if v is not None:
                    keysub[k] = max(keysub.get(k, 0.0) or 0.0, v)
        keysub.update({k: v for k, v in tsub.items()})
        flags: List[str] = []
        for k, v in sorted(keysub.items(), key=lambda kv: -(kv[1] or 0)):
            if v is not None and v >= 0.5 and k in VIDEO_FLAGS:
                flags.append(VIDEO_FLAGS[k])
        notes = [f"Analysed {len(valid)} keyframes + {tried}-frame temporal window."]
        notes += [x for x in rep["notes"] if x not in notes][:2]

        out = _empty("video")
        m = dict(rep["metrics"])
        m.update({k: _py(v) for k, v in tmet.items()})
        m["frames_analyzed"] = len(valid)
        out.update(ok=True, verdict=_verdict(risk, conf), risk=float(risk), confidence=float(conf),
                   metrics=m, subscores={k: _py(v) for k, v in keysub.items()}, flags=flags,
                   notes=notes, contributions={k: float(v) for k, v in contrib.items()},
                   annotated_png=rep["annotated_png"], timeline=[round(float(x), 3) for x in risks])
        return done(out)
    except Exception as exc:
        return done(_empty("video", f"Video analysis failed: {exc}"))
    finally:
        if cap is not None:
            cap.release()


# --------------------------------------------------------------------------- #
# FUSION
# --------------------------------------------------------------------------- #
def _fuse_media(parts, max_boost: float = 0.2, veto: float = 0.85) -> Tuple[float, float]:
    base = _wavg([(r["risk"], w * r["confidence"]) for r, w in parts])
    trusted = [r["risk"] for r, _ in parts if r["confidence"] >= 0.3]
    peak = max(trusted) if trusted else 0.0
    risk = (1 - max_boost) * base + max_boost * peak
    if peak >= veto:
        risk = max(risk, THRESH_SYNTHETIC + 0.05)
    conf = _wavg([(r["confidence"], w) for r, w in parts])
    return float(risk), float(conf)


def fuse_risk(audio: Optional[Dict[str, Any]] = None, video: Optional[Dict[str, Any]] = None,
              w_audio: float = 0.5, w_video: float = 0.5,
              max_boost: float = 0.2, veto: float = 0.85) -> Dict[str, Any]:
    """
    Baseline 0.5*A + 0.5*V, made confidence-aware:
      base = sum(w_i * c_i * R_i) / sum(w_i * c_i)
      risk = (1 - b) * base + b * max(R_i of modalities with c_i >= 0.3)
      a trusted modality >= veto forces at least the SYNTHETIC band.
    """
    parts = [(r, w) for r, w in ((audio, w_audio), (video, w_video))
             if r and r.get("confidence", 0) > 0]
    if not parts:
        return {"risk": 0.0, "confidence": 0.0, "verdict": "INCONCLUSIVE", "modalities": 0}
    risk, conf = _fuse_media(parts, max_boost, veto)
    return {"risk": risk, "confidence": conf, "verdict": _verdict(risk, conf), "modalities": len(parts)}


def fuse_all(audio: Optional[Dict[str, Any]] = None, video: Optional[Dict[str, Any]] = None,
             liveness: Optional[Dict[str, Any]] = None, nlp: Optional[Dict[str, Any]] = None
             ) -> Dict[str, Any]:
    """
    synthetic-media risk : confidence-aware fusion of voice, video and the liveness challenge
                           (a failed challenge is a strong, near-veto signal)
    intent amplifier     : a scam-like transcript raises the overall risk; on its own it can reach
                           SUSPICIOUS (step-up verification) but never SYNTHETIC, because a script
                           alone does not prove the media is fake.
    """
    media = [(r, w) for r, w in ((audio, 0.4), (video, 0.4), (liveness, 0.5))
             if r and r.get("ok") and r.get("confidence", 0) > 0]
    nlp_ok = bool(nlp and nlp.get("ok"))
    if not media and not nlp_ok:
        return {"risk": 0.0, "confidence": 0.0, "verdict": "INCONCLUSIVE", "modalities": 0,
                "synthetic_risk": 0.0, "intent_risk": 0.0, "reasons": []}
    reasons: List[str] = []
    synth, conf = 0.0, 0.0
    if media:
        synth, conf = _fuse_media(media)
        if liveness and liveness.get("ok") and liveness.get("confidence", 0) >= 0.4 \
                and (liveness["subscores"].get("direction") or 0) >= 0.9:
            if synth < 0.72:
                reasons.append("Liveness challenge failed: the requested head turn was not performed.")
            synth = max(synth, 0.72)
        for name, r in (("Voice", audio), ("Video", video), ("Liveness", liveness)):
            if r and r.get("ok") and r["verdict"] in ("SYNTHETIC", "SUSPICIOUS"):
                reasons.append(f"{name} analysis: {r['verdict'].lower()} ({round(r['risk'] * 100)}% risk).")
    intent = float(nlp["risk"] * nlp["confidence"]) if nlp_ok else 0.0
    risk = synth + (1.0 - synth) * 0.35 * intent if media else min(float(nlp["risk"]), 0.58)
    if nlp_ok and nlp["risk"] >= 0.35:
        reasons.append(f"Transcript shows social-engineering tactics ({nlp['verdict'].lower()}).")
        if media and synth >= 0.35:
            reasons.append("Suspicious media combined with a scripted pressure request.")
    conf = conf if media else float(nlp["confidence"]) * 0.8
    return {"risk": float(risk), "confidence": float(conf), "verdict": _verdict(risk, conf),
            "modalities": len(media) + (1 if nlp_ok else 0), "synthetic_risk": float(synth),
            "intent_risk": float(nlp["risk"]) if nlp_ok else 0.0, "reasons": reasons}

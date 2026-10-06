
from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

import detector

DIRECTIONS = ("LEFT", "RIGHT")
WEIGHTS = {"direction": 0.45, "timing": 0.15, "replay": 0.15, "identity": 0.10, "artifact": 0.15}
LABELS = {"direction": "Head-turn direction", "timing": "Response time", "replay": "Replay check",
          "identity": "Geometry continuity", "artifact": "Turned-face artefacts"}
FLAGS = {
    "direction": "Requested head turn was NOT performed (wrong direction or too small)",
    "timing": "Response time is outside the human window (too fast / too slow)",
    "replay": "Response frame is (nearly) identical to the reference frame: possible replay",
    "identity": "Face proportions changed between the reference and response frames",
    "artifact": "Artefacts found on the turned face",
}
MIN_TURN, GOOD_TURN = 0.03, 0.12           # change in signed yaw
HUMAN_MIN_S, HUMAN_FAST_S, SLOW_S, STALE_S = 0.3, 0.9, 20.0, 45.0


def new_challenge(rng: Optional[random.Random] = None) -> Dict[str, Any]:
    rng = rng or random.Random()
    return {"direction": rng.choice(DIRECTIONS), "delay_s": round(rng.uniform(2.0, 5.0), 2)}


def expected_sign(direction: str, mirrored: bool) -> int:
    """Sign of detector.yaw_signed (positive = nose toward image-right) expected for a turn to the
    USER's left/right. Un-mirrored image: user's left = image right (+1). Mirrored: flipped."""
    s = 1 if direction == "LEFT" else -1
    return -s if mirrored else s


def _vertical_profile(lm: np.ndarray) -> np.ndarray:
    """Vertical proportions barely change with yaw: eyes->mouth, nose->eye line, mouth->chin."""
    top, chin = lm[10, 1], lm[152, 1]
    h = abs(chin - top) + 1e-6
    eye_y = 0.5 * (lm[R_EYE, 1] + lm[L_EYE, 1])
    return np.array([(lm[13, 1] - eye_y) / h, (lm[1, 1] - eye_y) / h, (chin - lm[13, 1]) / h])


R_EYE, L_EYE = detector.R_EYE_OUT, detector.L_EYE_OUT


def evaluate(baseline: bytes, response: bytes, direction: str, response_time_s: float,
             mirrored: bool = True) -> Dict[str, Any]:
    """Score one challenge. Never raises."""
    t0 = time.perf_counter()
    out = detector._empty("liveness")
    out["timeline"] = []
    try:
        b = detector.face_pose(baseline)
        r = detector.face_pose(response)
        if not r["ok"]:
            out["error"] = ("No face found in the response photo. Turn less than ~45 degrees and keep "
                            f"your whole face visible, then retry. ({r['error']})")
            return out
        notes = []
        base_yaw = b["yaw_signed"] if b["ok"] else 0.0
        if not b["ok"]:
            notes.append("Reference frame had no face: assuming a frontal start.")
        sign = expected_sign(direction, mirrored)
        turn = sign * (r["yaw_signed"] - base_yaw)

        sub: Dict[str, Optional[float]] = {}
        sub["direction"] = detector.ramp(turn, GOOD_TURN, MIN_TURN)
        rt = float(response_time_s)
        sub["timing"] = max(detector.ramp(rt, HUMAN_FAST_S, HUMAN_MIN_S),
                            detector.ramp(rt, SLOW_S, STALE_S))
        same_bytes = baseline == response
        mad = float("nan")
        if b["ok"]:
            a = cv2.resize(b["gray"], (64, 64)).astype(np.float32)
            c = cv2.resize(r["gray"], (64, 64)).astype(np.float32)
            mad = float(np.mean(np.abs(a - c)))
        sub["replay"] = 1.0 if same_bytes else (detector.ramp(mad, 2.0, 0.5) if np.isfinite(mad) else 0.0)
        ident = None
        if b["ok"]:
            diff = float(np.mean(np.abs(_vertical_profile(b["lm"]) - _vertical_profile(r["lm"]))))
            ident = detector.ramp(diff, 0.05, 0.13)
        sub["identity"] = ident
        art = detector._video_mesh(r["img"], r["gray"], r["lm"])
        sub["artifact"] = art["risk"] if art["ok"] else None

        pairs = [(v, WEIGHTS[k]) for k, v in sub.items() if v is not None]
        risk = detector._wavg(pairs)
        if sub["direction"] >= 0.9:
            risk = max(risk, 0.75)
        if sub["replay"] >= 0.9:
            risk = max(risk, 0.80)
        if sub["timing"] >= 0.9:                       # answered before a human could have reacted
            risk = max(risk, 0.50)
        tot = sum(w for _, w in pairs)
        contrib = {LABELS[k]: v * WEIGHTS[k] / tot for k, v in sub.items() if v is not None}
        s = sum(contrib.values())
        if s > 1e-9:                                   # keep contributions summing to the risk
            contrib = {k: v * risk / s for k, v in contrib.items()}

        conf = detector.ramp(r["face_w"], 60, 160) * (1.0 if b["ok"] else 0.7)
        flags = [FLAGS[k] for k, v in sorted(sub.items(), key=lambda kv: -(kv[1] or 0))
                 if v is not None and v >= 0.5]
        if sub["direction"] < 0.5:
            notes.append(f"Head turn confirmed ({direction.lower()}, signed yaw change {turn:+.2f}).")
        if sub["direction"] >= 0.9:
            notes.append("If you turned the right way but it failed, flip 'Camera image is mirrored' "
                         "in the sidebar.")
        out.update(
            ok=True, verdict=detector._verdict(risk, conf), risk=float(risk), confidence=float(conf),
            subscores={k: detector._py(v) for k, v in sub.items()}, flags=flags,
            notes=notes, contributions={k: float(v) for k, v in contrib.items()},
            annotated_png=art.get("annotated_png"),
            metrics={"challenge": direction, "yaw_reference": detector._py(base_yaw),
                     "yaw_response": detector._py(r["yaw_signed"]), "turn_toward_target": detector._py(turn),
                     "response_time_s": round(rt, 2), "frame_difference": detector._py(mad),
                     "mirrored_camera": mirrored})
        return out
    except Exception as exc:
        out["error"] = f"Liveness check failed: {exc}"
        return out
    finally:
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

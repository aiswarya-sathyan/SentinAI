from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# category -> (weight, label, regex patterns)
_CATS: Dict[str, Tuple[float, str, List[str]]] = {
    "urgency": (0.30, "Artificial urgency", [
        r"\burgent(ly)?\b", r"\bimmediately\b", r"\bright (now|away)\b", r"\basap\b",
        r"\bwithin (the next )?\d+ ?(minutes?|mins?|hours?)\b", r"\bdon'?t (wait|delay)\b",
        r"\blast chance\b", r"\bbefore it'?s too late\b", r"\bdeadline\b", r"\bhurry\b",
        r"\bturant\b", r"\bjaldi\b", r"\budane\b", r"तुरंत", r"जल्दी", r"உடனே"]),
    "authority": (0.30, "Authority / identity claim", [
        r"\bthis is (the |your )?(ceo|cfo|director|manager|officer|inspector|bank|police|customs)\b",
        r"\b(fraud|security|compliance|risk) (department|team|desk)\b", r"\bcustomer (care|support)\b",
        r"\b(cyber ?crime|income tax|cbi|rbi|narcotics|customs) (department|officer|police|cell)?\b",
        r"\bi am calling (from|on behalf of)\b", r"\bsenior (management|executive)\b"]),
    "secrecy": (0.35, "Secrecy request", [
        r"\bdon'?t tell (anyone|anybody|your)\b", r"\bkeep (this|it) (between us|confidential|secret|quiet)\b",
        r"\bdo not (inform|tell|discuss|share this)\b", r"\bdon'?t mention\b", r"\bconfidential\b",
        r"\bkisi ko (mat|nahi) ?(batana|bolna)?\b"]),
    "financial_request": (0.45, "Request to move money", [
        r"\b(transfer|wire|send|remit|deposit)\b[^.?!]{0,40}\b(money|funds|amount|payment|rs\.?|inr|usd|rupees|dollars?|\$|₹)",
        r"\bgift ?cards?\b", r"\b(bitcoin|crypto(currency)?|usdt)\b", r"\bupi\b", r"\bwestern union\b",
        r"\b(account|ifsc|routing|swift) (number|code)\b", r"\bpay(ment)? (immediately|now|today|the fee)\b",
        r"\bprocessing fee\b", r"\bpaise bhej", r"पैसे भेज", r"\bpanam anuppu", r"பணம்"]),
    "credential_request": (0.55, "Request for credentials / OTP", [
        r"\b(otp|one[- ]time (password|passcode|code)|verification code|security code)\b",
        r"\b(pin|cvv|cvc|password|passcode)\b", r"\b(card|aadhaar|aadhar|pan|passport|social security|ssn) (number|details|no\.?)\b",
        r"\b(read|tell|share|give|say)\b[^.?!]{0,25}\b(code|otp|pin|password)\b", r"\botp (batao|bata do|sollunga)\b",
        r"ओटीपी", r"ஓடிபி"]),
    "threat": (0.35, "Threat / fear appeal", [
        r"\barrest(ed)?\b", r"\blegal action\b", r"\b(account|card|sim|number) (will be |has been |is )?(blocked|suspended|frozen|closed|deactivated)\b",
        r"\bwarrant\b", r"\bpenalty\b", r"\bcompromised\b", r"\bunauthori[sz]ed (transaction|access|activity)\b",
        r"\bkhata band\b", r"\bgiraftar\b", r"गिरफ्तार", r"கைது"]),
    "remote_access": (0.45, "Remote-access / link lure", [
        r"\b(anydesk|teamviewer|quicksupport|rustdesk)\b", r"\bremote (access|control|session)\b",
        r"\b(install|download) (this|the|an?) (app|software|apk|tool)\b", r"\bclick (on )?(the|this) link\b",
        r"\bscreen ?shar(e|ing)\b"]),
    "reward_lure": (0.25, "Prize / refund lure", [
        r"\byou (have )?won\b", r"\blottery\b", r"\bcongratulations\b", r"\brefund\b", r"\bcash ?back\b",
        r"\bclaim your (prize|reward|gift)\b"]),
    "emergency_kin": (0.30, "Family-emergency pretext", [
        r"\bi'?m in (trouble|jail|hospital)\b", r"\bnew number\b", r"\blost my phone\b", r"\bbail\b",
        r"\baccident\b", r"\b(mom|dad|grandma|grandpa)\b[^.?!]{0,40}\b(help|money|send)\b"]),
}
_COMPILED = {k: [re.compile(p, re.I) for p in pats] for k, (_, _, pats) in _CATS.items()}
_PRESSURE = ("urgency", "threat", "authority", "secrecy")
_ASK = ("financial_request", "credential_request", "remote_access")

_ASR_MODEL = None


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0)) if lo != hi else 0.0


def analyze_text(text: Optional[str]) -> Dict[str, Any]:
    """Never raises. Returns the standard analyzer dict plus `matches` = [(start, end, category)]."""
    t0 = time.perf_counter()
    out: Dict[str, Any] = {"ok": False, "kind": "nlp", "error": None, "verdict": "INCONCLUSIVE",
                           "risk": 0.0, "confidence": 0.0, "latency_ms": 0.0, "metrics": {},
                           "subscores": {}, "flags": [], "notes": [], "matches": [],
                           "contributions": {}, "spectrogram": None, "annotated_png": None}
    try:
        text = (text or "").strip()
        words = re.findall(r"\w+", text, flags=re.UNICODE)
        if len(words) < 3:
            out["error"] = "Transcript is too short. Paste or type what was said on the call."
            return out
        text = text[:20000]

        hits: Dict[str, List[Tuple[int, int]]] = {}
        for cat, regs in _COMPILED.items():
            for rg in regs:
                for m in rg.finditer(text):
                    hits.setdefault(cat, []).append((m.start(), m.end()))
        sub: Dict[str, float] = {}
        matches: List[Tuple[int, int, str]] = []
        for cat, spans in hits.items():
            uniq = sorted(set(spans))
            n = len(uniq)
            sub[cat] = min(1.0, 0.6 + 0.2 * (n - 1))     # first hit is already strong evidence
            matches += [(s, e, cat) for s, e in uniq]

        # noisy-OR over weighted categories
        keep = 1.0
        parts = {}
        for cat, s in sub.items():
            c = _CATS[cat][0] * s
            parts[cat] = c
            keep *= (1.0 - c)
        risk = 1.0 - keep
        combo = any(c in sub for c in _ASK) and any(c in sub for c in _PRESSURE)
        if combo:
            risk = risk + (1.0 - risk) * 0.35
            out["notes"].append("Combination detected: a request for money/credentials delivered "
                                "under pressure is the classic vishing pattern.")
        risk = float(min(risk, 0.99))

        nw = len(words)
        conf = (0.6 + 0.4 * _ramp(nw, 8, 60)) if sub else (0.3 + 0.7 * _ramp(nw, 8, 80))

        tot = sum(parts.values()) or 1.0
        contrib = {_CATS[c][1]: risk * v / tot for c, v in parts.items()}
        flags = []
        for cat in sorted(sub, key=lambda c: -parts[c]):
            s, e = hits[cat][0]
            phrase = text[s:e]
            flags.append(f"{_CATS[cat][1]}: “{phrase}”" + (f" (+{len(set(hits[cat])) - 1} more)"
                                                          if len(set(hits[cat])) > 1 else ""))
        verdict = "SCAM-LIKE" if (risk >= 0.6 and conf >= 0.5) else \
                  "SUSPICIOUS" if (risk >= 0.35 and conf >= 0.3) else "BENIGN"
        out.update(ok=True, verdict=verdict, risk=risk, confidence=float(conf),
                   subscores={_CATS[c][1]: sub.get(c, 0.0) for c in _CATS},
                   flags=flags, matches=sorted(matches), contributions=contrib,
                   metrics={"words": nw, "tactics_found": len(sub), "combo_ask_plus_pressure": combo})
        return out
    except Exception as exc:  # pragma: no cover
        out["error"] = f"NLP analysis failed: {exc}"
        return out
    finally:
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)


# --------------------------------------------------------------------------- #
# optional local speech-to-text
# --------------------------------------------------------------------------- #
def asr_available() -> bool:
    try:
        import faster_whisper
        return True
    except Exception:
        return False


def transcribe(path: str, model_size: str = "base") -> Tuple[str, Optional[str]]:
    """Local transcription with faster-whisper (pip install faster-whisper). First use downloads
    the model once; after that it is offline. Returns (text, error)."""
    global _ASR_MODEL
    try:
        from faster_whisper import WhisperModel
        if _ASR_MODEL is None:
            _ASR_MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
        segs, _info = _ASR_MODEL.transcribe(path, beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip(), None
    except Exception as exc:
        return "", f"Speech-to-text unavailable: {exc}"

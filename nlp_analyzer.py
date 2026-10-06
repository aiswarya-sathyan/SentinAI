from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# category -> (weight, label, regex patterns)
# category -> (weight, label, regex patterns)
_CATS: Dict[str, Tuple[float, str, List[str]]] = {
    "urgency": (0.30, "Artificial urgency", [
        r"\burgent(ly)?\b", r"\bimmediately\b", r"\bright (now|away)\b", r"\basap\b",
        r"\bwithin (the next )?\d+ ?(minutes?|mins?|hours?)\b", r"\bdon'?t (wait|delay)\b",
        r"\blast chance\b", r"\bbefore it'?s too late\b", r"\bdeadline\b", r"\bhurry\b",
        r"\bturant\b", r"\bjaldi\b", r"\budane\b", r"तुरंत", r"जल्दी", r"உடனே",
        # Malayalam Script & Manglish
        r"ഉടൻ", r"ഉടനെ", r"വൈകരുത്", r"\budan\b", r"\budane\b", r"\bvaikaruthu\b", r"\bippol thanne\b"
    ]),
    "authority": (0.30, "Authority / identity claim", [
        r"\bthis is (the |your )?(ceo|cfo|director|manager|officer|inspector|bank|police|customs)\b",
        r"\b(fraud|security|compliance|risk) (department|team|desk)\b", r"\bcustomer (care|support)\b",
        r"\b(cyber ?crime|income tax|cbi|rbi|narcotics|customs) (department|officer|police|cell)?\b",
        r"\bi am calling (from|on behalf of)\b", r"\bsenior (management|executive)\b",
        # Malayalam Script & Manglish
        r"പോലീസ്", r"ബാങ്ക് ഉദ്യോഗസ്ഥൻ", r"കസ്റ്റംസ്", r"ക്രെഡിറ്റ് കാർഡ് വിഭാഗം",
        r"\bpolice station\b", r"\bbank officer\b", r"\bcustoms officer\b"
    ]),
    "secrecy": (0.35, "Secrecy request", [
        r"\bdon'?t tell (anyone|anybody|your)\b", r"\bkeep (this|it) (between us|confidential|secret|quiet)\b",
        r"\bdo not (inform|tell|discuss|share this)\b", r"\bdon'?t mention\b", r"\bconfidential\b",
        r"\bkisi ko (mat|nahi) ?(batana|bolna)?\b",
        # Malayalam Script & Manglish
        r"ആരോടും പറയരുത്", r"രഹസ്യമായി വെക്കുക", r"രഹസ്യം",
        r"\baarodum parayaruthe?\b", r"\brahasyam\b", r"\barum ariyaruthe?\b"
    ]),
    "financial_request": (0.45, "Request to move money", [
        r"\b(transfer|wire|send|remit|deposit)\b[^.?!]{0,40}\b(money|funds|amount|payment|rs\.?|inr|usd|rupees|dollars?|\$|₹)",
        r"\bgift ?cards?\b", r"\b(bitcoin|crypto(currency)?|usdt)\b", r"\bupi\b", r"\bwestern union\b",
        r"\b(account|ifsc|routing|swift) (number|code)\b", r"\bpay(ment)? (immediately|now|today|the fee)\b",
        r"\bprocessing fee\b", r"\bpaise bhej", r"पैसे भेज", r"\bpanam anuppu", r"பணம்",
        # Malayalam Script & Manglish
        r"പണം", r"പണം അയക്കുക", r"അക്കൗണ്ട്", r"ബാങ്ക് അക്കൗണ്ട്", r"പൈസ",
        r"\bpanam\b", r"\bpaisa\b", r"\bpanam ayakkya\b", r"\baccountil idu\b", r"\bfee adapikkyu\b"
    ]),
    "credential_request": (0.55, "Request for credentials / OTP", [
        r"\b(otp|one[- ]time (password|passcode|code)|verification code|security code)\b",
        r"\b(pin|ATM Pin|cvv|cvc|password|passcode)\b", r"\b(card|aadhaar|aadhar|pan|passport|social security|ssn) (number|details|no\.?)\b",
        r"\b(read|tell|share|give|say)\b[^.?!]{0,25}\b(code|otp|pin|password)\b", r"\botp (batao|bata do|sollunga)\b",
        r"ओटीपी", r"ஓடிபி",
        # Malayalam Script & Manglish
        r"ഒടിപി", r"പിൻ നമ്പർ", r"പാസ്‌വേഡ്", r"ഓടിപി പറയൂ",
        r"\botp parayu\b", r"\bpin parayu\b", r"\botp paranjutharu\b", r"\botp ayakkoo\b",r"\bATM pin paranjutharu\b"
    ]),
    "threat": (0.35, "Threat / fear appeal", [
        r"\barrest(ed)?\b", r"\blegal action\b", r"\b(account|card|sim|number) (will be |has been |is )?(blocked|suspended|frozen|closed|deactivated)\b",
        r"\bwarrant\b", r"\bpenalty\b", r"\bcompromised\b", r"\bunauthori[sz]ed (transaction|access|activity)\b",
        r"\bkhata band\b", r"\bgiraftar\b", r"गिरफ्तार", r"கைது",
        # Malayalam Script & Manglish
        r"അറസ്റ്റ്", r"ഡിജിറ്റൽ അറസ്റ്റ്", r"കേസ്", r"അക്കൗണ്ട് ബ്ലോക്ക് ചെയ്യും", r"കേസ് എടുക്കും",
        r"\barrest\b", r"\bdigital arrest\b", r"\bcase edukkum\b", r"\bblock aakum\b", r"\baccount block cheyyum\b"
    ]),
    "remote_access": (0.45, "Remote-access / link lure", [
        r"\b(anydesk|teamviewer|quicksupport|rustdesk)\b", r"\bremote (access|control|session)\b",
        r"\b(install|download) (this|the|an?) (app|software|apk|tool)\b", r"\bclick (on )?(the|this) link\b",
        r"\bscreen ?shar(e|ing)\b",
        # Malayalam Script & Manglish
        r"ലിങ്കിൽ ക്ലിക്ക് ചെയ്യുക", r"ആപ്പ് ഡൗൺലോഡ് ചെയ്യുക",
        r"\blinkil click cheyyu\b", r"\bapp download cheyyu\b"
    ]),
    "reward_lure": (0.25, "Prize / refund lure", [
        r"\byou (have )?won\b", r"\blottery\b", r"\bcongratulations\b", r"\brefund\b", r"\bcash ?back\b",
        r"\bclaim your (prize|reward|gift)\b",
        # Malayalam Script & Manglish
        r"സമ്മാനം", r"ലോട്ടറി അടിച്ചു", r"റിഫണ്ട്",
        r"\bsammanam\b", r"\blottery adichu\b", r"\brefund tharam\b"
    ]),
    "emergency_kin": (0.30, "Family-emergency pretext", [
        r"\bi'?m in (trouble|jail|hospital)\b", r"\bnew number\b", r"\blost my phone\b", r"\bbail\b",
        r"\baccident\b", r"\b(mom|dad|grandma|grandpa)\b[^.?!]{0,40}\b(help|money|send)\b",
        # Malayalam Script & Manglish
        r"അപകടം പറ്റിയതാ", r"ആശുപത്രിയിലാണ്", r"അടിയന്തര സഹായം",
        r"\baccident pati\b", r"\bhospitalil aane\b", r"\bpanam sahayam\b"
    ]),
}
_COMPILED = {k: [re.compile(p, re.I) for p in pats] for k, (_, _, pats) in _CATS.items()}
# Phrases that are scam indicators on their own, however short the utterance ("digital arrest" is not
# something any real agency says on a phone call).
_HARD = re.compile(r"\bdigital arrest\b|ഡിജിറ്റൽ അറസ്റ്റ്|\bdigital arrest cheyyum\b", re.I)
_PRESSURE = ("urgency", "threat", "authority", "secrecy")
_ASK = ("financial_request", "credential_request", "remote_access")

_ASR_MODELS: Dict[str, Any] = {}      # model_size -> loaded WhisperModel (loaded once per process)
ASR_SR = 16000                        # Whisper's native sample rate
SILENCE_RMS = 1e-4                    # below this a buffer is dead air / static floor, not speech


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0)) if lo != hi else 0.0


def analyze_text(text: Optional[str]) -> Dict[str, Any]:
    """Never raises. Returns the standard analyzer dict plus `matches` = [(start, end, category)]."""
    t0 = time.perf_counter()
    out: Dict[str, Any] = {
        "ok": False, "kind": "nlp", "error": None, "verdict": "INCONCLUSIVE",
        "risk": 0.0, "confidence": 0.0, "latency_ms": 0.0, "metrics": {},
        "subscores": {}, "flags": [], "notes": [], "matches": [],
        "contributions": {}, "spectrogram": None, "annotated_png": None
    }
    try:
        text = (text or "").strip()
        # Whitespace tokens that contain at least one letter/digit. `\w+` on its own is NOT safe for
        # Malayalam/Hindi/Tamil: virama & vowel signs are combining marks (not \w), so one word such as
        # "ഡിജിറ്റൽ" would be split into 5+ fragments and inflate the word count.
        words = [w for w in text.split() if re.search(r"\w", w, flags=re.UNICODE)]
        if not words:
            out["error"] = "No speech detected in transcript."
            return out

        text = text[:20000]

        hits: Dict[str, List[Tuple[int, int]]] = {}
        for cat, regs in _COMPILED.items():
            for rg in regs:
                for m in rg.finditer(text):
                    hits.setdefault(cat, []).append((m.start(), m.end()))

        # 1-2 word utterances ("ATM PIN", "OTP", "ഡിജിറ്റൽ അറസ്റ്റ്") are only analysable when a
        # high-risk pattern matched. Short AND clean -> too little evidence to say anything.
        if len(words) < 3 and not hits:
            out["error"] = "Transcript is too short to evaluate general intent."
            return out

        sub: Dict[str, float] = {}
        matches: List[Tuple[int, int, str]] = []
        for cat, spans in hits.items():
            uniq = sorted(set(spans))
            n = len(uniq)
            sub[cat] = min(1.0, 0.6 + 0.2 * (n - 1))     # first hit is strong evidence
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
        
        # If credentials/money are requested explicitly, force verdict to SUSPICIOUS/SCAM-LIKE even for short text
        if "credential_request" in sub or "financial_request" in sub:
            verdict = "SCAM-LIKE" if risk >= 0.45 else "SUSPICIOUS"
        if _HARD.search(text):
            risk = max(risk, 0.45)
            if verdict == "BENIGN":
                verdict = "SUSPICIOUS"

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
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# local speech-to-text (faster-whisper)
# --------------------------------------------------------------------------- #


def _load_asr_audio(path: str) -> Optional[np.ndarray]:
    """Decode `path` to a clean 16 kHz mono float32 array in [-1, 1] for Whisper."""
    y = sr = None
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(path, sr=None, mono=True)
        except Exception:
            return None
    try:
        y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if y.size == 0:
            return y
        y = y - float(np.mean(y))
        if int(sr) != ASR_SR:
            try:
                import librosa
                y = librosa.resample(y, orig_sr=int(sr), target_sr=ASR_SR)
            except Exception:
                from scipy.signal import resample_poly
                g = math.gcd(ASR_SR, int(sr))
                y = resample_poly(y, ASR_SR // g, int(sr) // g).astype(np.float32)
        if float(np.sqrt(np.mean(y ** 2))) < SILENCE_RMS:
            return np.zeros(0, dtype=np.float32)
        peak = float(np.max(np.abs(y)))
        return (y / (peak + 1e-9) * 0.95).astype(np.float32)
    except Exception:
        return None


def transcribe(path: str, model_size: str = "base") -> Tuple[str, Optional[str]]:
    """Local transcription with faster-whisper, tuned so clean synthetic voices are not discarded."""
    if not asr_available():
        return "", "faster-whisper is not installed. Run 'pip install faster-whisper' in terminal."

    try:
        from faster_whisper import WhisperModel
        model = _ASR_MODELS.get(model_size)
        if model is None:
            model = _ASR_MODELS[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")

        audio = _load_asr_audio(path)
        if audio is not None and audio.size == 0:
            return "", None                      # Silent buffer: nothing to transcribe
        source = audio if audio is not None else path

        segs, _info = model.transcribe(
            source,
            beam_size=1,
            vad_filter=False,
            temperature=0.0,
            no_speech_threshold=0.8,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segs if seg.text and seg.text.strip()).strip()
        return text, None
    except Exception as exc:
        return "", f"Speech-to-text unavailable: {exc}"
    from datasets import load_dataset

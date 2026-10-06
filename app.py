import copy
import hashlib
import html
import os
import re
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

import audit
import detector
import liveness
import nlp_analyzer
import policy

try:
    import librosa.display  # noqa: F401
    SPEC_OK = True
except Exception:
    SPEC_OK = False

st.set_page_config(page_title="SentinAI | Deepfake Defense", page_icon="🛡️", layout="wide")
SS = st.session_state
HERE = Path(__file__).parent
SAMPLES_DIR = HERE / "samples"

# --------------------------------------------------------------------------- #
# theme
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--bg:#070b0f;--panel:#0c1319;--line:#17303a;--em:#10b981;--em2:#34d399;--txt:#e2e8f0;--mut:#8aa0ae;}
.stApp{
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(16,185,129,.13), transparent 60%),
    radial-gradient(rgba(16,185,129,.07) 1px, transparent 1px) 0 0/24px 24px,
    var(--bg);
  color:var(--txt);font-family:'Space Grotesk',system-ui,sans-serif;
}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#080d12;border-right:1px solid var(--line)}
#MainMenu, footer{visibility:hidden}
.block-container{padding-top:1.6rem;max-width:1200px}

.hero{padding:6px 0 18px;border-bottom:1px solid var(--line);margin-bottom:18px}
.logo{font-size:2.1rem;font-weight:700;letter-spacing:.28em}
.logo span{color:var(--em);text-shadow:0 0 18px rgba(16,185,129,.7)}
.tag{color:var(--mut);margin-top:2px}
.pills{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}
.pills span{font:600 .68rem 'JetBrains Mono',monospace;letter-spacing:.12em;color:var(--em2);
  border:1px solid rgba(16,185,129,.35);background:rgba(16,185,129,.07);padding:4px 10px;border-radius:999px}

.ptitle{font:600 .72rem 'JetBrains Mono',monospace;letter-spacing:.2em;color:var(--em);
  text-transform:uppercase;margin:4px 0 10px}
.sect{font:700 1.05rem 'Space Grotesk',sans-serif;letter-spacing:.08em;margin:26px 0 6px;
  padding-bottom:6px;border-bottom:1px solid var(--line)}
[data-testid="stFileUploaderDropzone"]{background:var(--panel);border:1px dashed var(--line)}

/* threat radar banner */
.banner{--c:#64748b;--bg2:rgba(100,116,139,.10);display:flex;align-items:center;gap:18px;
  padding:16px 20px;border:1px solid var(--c);border-radius:14px;background:var(--bg2);
  box-shadow:0 0 28px -8px var(--c)}
.banner.ok{--c:#10b981;--bg2:rgba(16,185,129,.10)}
.banner.warn{--c:#f59e0b;--bg2:rgba(245,158,11,.10)}
.banner.bad{--c:#ef4444;--bg2:rgba(239,68,68,.12)}
.radar{position:relative;width:54px;height:54px;border-radius:50%;border:2px solid var(--c);flex:none}
.radar::before,.radar::after{content:"";position:absolute;inset:-2px;border-radius:50%;
  border:2px solid var(--c);animation:ping 2s ease-out infinite}
.radar::after{animation-delay:1s}
.radar i{position:absolute;inset:0;border-radius:50%;opacity:.55;
  background:conic-gradient(from 0deg,transparent 0 300deg,var(--c) 360deg);animation:sweep 2.4s linear infinite}
@keyframes ping{0%{transform:scale(1);opacity:.7}100%{transform:scale(1.9);opacity:0}}
@keyframes sweep{to{transform:rotate(360deg)}}
.b-main{flex:1}
.b-title{font-size:1.45rem;font-weight:700;letter-spacing:.08em;color:var(--c)}
.b-sub{color:var(--mut);font-size:.9rem;margin-top:2px}
.b-score{text-align:right}
.b-score b{display:block;font:600 2rem 'JetBrains Mono',monospace;color:var(--c)}
.b-score span{font:600 .62rem 'JetBrains Mono',monospace;letter-spacing:.14em;color:var(--mut)}
.bar{position:relative;height:6px;border-radius:99px;margin:10px 0 4px;overflow:hidden;
  background:linear-gradient(90deg,#10b981,#f59e0b 60%,#ef4444)}
.bar .mask{position:absolute;top:0;bottom:0;right:0;background:#0b1117}

/* recommended action */
.policy{--c:#64748b;margin:12px 0 4px;padding:12px 16px;border:1px solid var(--line);
  border-left:4px solid var(--c);border-radius:10px;background:rgba(12,19,25,.9)}
.policy.ok{--c:#10b981}.policy.warn{--c:#f59e0b}.policy.bad{--c:#ef4444}
.p-head{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.p-level{font:600 .66rem 'JetBrains Mono',monospace;letter-spacing:.16em;color:var(--c);
  border:1px solid var(--c);border-radius:99px;padding:2px 9px}
.p-action{font-weight:700;letter-spacing:.08em;color:var(--c)}
.p-head2{color:var(--mut);font-size:.88rem;margin:6px 0 2px}
.policy ul{margin:4px 0 0 18px;padding:0;font-size:.86rem;color:var(--txt)}
.policy .why{margin-top:6px;font-size:.78rem;color:var(--mut)}

/* metric cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0}
.card{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:12px;padding:12px 14px;
  background:linear-gradient(180deg,rgba(16,185,129,.07),rgba(12,19,25,.92))}
.card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--em)}
.card.warn::before{background:#f59e0b}
.c-label{font:600 .62rem 'JetBrains Mono',monospace;letter-spacing:.16em;color:var(--mut);text-transform:uppercase}
.c-val{font:600 1.7rem 'JetBrains Mono',monospace;margin-top:4px}
.c-val small{font-size:.8rem;color:var(--mut);margin-left:4px}
.c-sub{font-size:.74rem;color:var(--mut);margin-top:2px}

.flags{list-style:none;padding:0;margin:8px 0}
.flags li{padding:8px 12px;margin-bottom:6px;border:1px solid var(--line);border-left:3px solid #f59e0b;
  border-radius:8px;background:rgba(245,158,11,.05);font-size:.88rem}
.flags li.good{border-left-color:var(--em);background:rgba(16,185,129,.05)}
.flags li.note{border-left-color:#64748b;background:rgba(100,116,139,.06);color:var(--mut)}
.empty{border:1px dashed var(--line);border-radius:12px;padding:46px 20px;text-align:center;color:var(--mut)}
.challenge{border:1px solid #f59e0b;border-radius:14px;padding:18px;text-align:center;margin:8px 0 12px;
  background:rgba(245,158,11,.08)}
.challenge b{display:block;font:700 1.5rem 'Space Grotesk',sans-serif;letter-spacing:.1em;color:#fbbf24}
.challenge span{color:var(--mut);font-size:.85rem}
.transcript{border:1px solid var(--line);border-radius:10px;padding:10px 14px;background:var(--panel);
  line-height:1.6;font-size:.92rem}
.transcript mark{background:rgba(239,68,68,.25);color:#fecaca;border-radius:4px;padding:0 3px}
.foot{margin-top:28px;padding-top:12px;border-top:1px solid var(--line);color:var(--mut);
  font:400 .72rem 'JetBrains Mono',monospace;letter-spacing:.06em}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EMERALD = LinearSegmentedColormap.from_list(
    "sentin", ["#04110d", "#0b3d2e", "#10b981", "#a7f3d0", "#ecfdf5"])

BANNERS = {
    "AUTHENTIC": ("ok", "AUTHENTIC", "No synthetic signature detected"),
    "SUSPICIOUS": ("warn", "SUSPICIOUS", "Anomalies found: step-up verification advised"),
    "SYNTHETIC": ("bad", "SYNTHETIC THREAT", "High-risk synthetic indicators: block or escalate"),
    "INCONCLUSIVE": ("idle", "INCONCLUSIVE", "Not enough signal to decide"),
    "BENIGN": ("ok", "BENIGN", "No social-engineering tactics found"),
    "SCAM-LIKE": ("bad", "SCAM-LIKE SCRIPT", "Pressure combined with a request for money or credentials"),
}
MODES = ["🎙 Voice defense", "🎥 Video & liveness", "🧬 Full session"]
AUDIO_EXT = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
IMG_EXT = [".jpg", ".jpeg", ".png"]
VID_EXT = [".mp4", ".mov", ".avi"]
EXAMPLE_SCAM = ("This is the fraud department of your bank. Your account will be blocked in 30 minutes. "
                "To stop it, read me the OTP you just received and don't tell anyone about this call.")
EXAMPLE_OK = ("Hi, it's me. Can we move the project review to Thursday afternoon? "
              "I'll share the slides before the meeting.")

# --------------------------------------------------------------------------- #
# session state: nonces, resets, history
# --------------------------------------------------------------------------- #
for _k, _v in (("timeline", []), ("audit", []), ("_logged", set()), ("fused_sig", None),
               ("lv", {"phase": "idle"})):
    SS.setdefault(_k, _v)

PANELS = ("voice", "video", "text")
PANEL_STATE = {"voice": ("res_audio_v3", "res_audio", "res_audio_clean", "sample_voice"),
               "video": ("res_video", "res_video_clean", "sample_video"),
               "text": ("res_nlp",)}


def nonce(panel):
    return SS.get(f"nonce_{panel}", 0)


def bump(*panels):
    for p in panels:
        SS[f"nonce_{p}"] = nonce(p) + 1


def clear_panel(panel):
    """Fresh, empty widgets + forget cached results for one panel."""
    bump(panel)
    for k in PANEL_STATE[panel]:
        SS.pop(k, None)
    if panel == "video":
        SS["lv"] = {"phase": "idle"}


def reset_all():
    for p in PANELS:
        clear_panel(p)
    SS["fused_sig"] = None


def clear_history():
    SS["timeline"], SS["audit"], SS["_logged"], SS["fused_sig"] = [], [], set(), None


# --------------------------------------------------------------------------- #
# render helpers
# --------------------------------------------------------------------------- #
def fmt(v, spec=".1f"):
    return "—" if v is None else format(v, spec)


def banner(res, sub=None, title=None):
    cls, default_title, default_sub = BANNERS[res["verdict"]]
    risk, conf = round(res["risk"] * 100), round(res["confidence"] * 100)
    st.markdown(
        f'<div class="banner {cls}"><div class="radar"><i></i></div>'
        f'<div class="b-main"><div class="b-title">{html.escape(title or default_title)}</div>'
        f'<div class="b-sub">{html.escape(sub or default_sub)}</div></div>'
        f'<div class="b-score"><b>{risk}%</b><span>RISK · CONF {conf}%</span></div></div>'
        f'<div class="bar"><div class="mask" style="left:{risk}%"></div></div>',
        unsafe_allow_html=True)


def policy_card(rec):
    steps = "".join(f"<li>{html.escape(s)}</li>" for s in rec["steps"])
    why = ""
    if rec["reasons"]:
        why = '<div class="why">Why: ' + html.escape(" ".join(rec["reasons"])) + "</div>"
    st.markdown(
        f'<div class="policy {rec["css"]}"><div class="p-head">'
        f'<span class="p-level">{rec["level"]}</span><span class="p-action">{rec["action"]}</span></div>'
        f'<div class="p-head2">{html.escape(rec["headline"])}</div><ul>{steps}</ul>{why}</div>',
        unsafe_allow_html=True)


def card(label, value, unit="", sub="", warn=False):
    u = f"<small>{unit}</small>" if unit else ""
    return (f'<div class="{"card warn" if warn else "card"}"><div class="c-label">{label}</div>'
            f'<div class="c-val">{value}{u}</div><div class="c-sub">{sub}</div></div>')


def cards(items):
    st.markdown('<div class="cards">' + "".join(items) + "</div>", unsafe_allow_html=True)


def findings(res):
    items = [f"<li>{html.escape(f)}</li>" for f in res["flags"]]
    items += [f'<li class="note">{html.escape(n)}</li>' for n in res["notes"]]
    if res["ok"] and not res["flags"]:
        items.insert(0, '<li class="good">No strong synthetic cues in this sample.</li>')
    if items:
        st.markdown('<div class="ptitle">Signal findings</div><ul class="flags">' + "".join(items) + "</ul>",
                    unsafe_allow_html=True)


def latency_card(res, budget=detector.LATENCY_BUDGET_MS, sub=None):
    lat = res["latency_ms"]
    return card("Latency", f"{lat:.0f}", "ms", sub or f"budget {budget} ms", warn=lat > budget)


def empty_state(text):
    st.markdown(f'<div class="empty">{text}</div>', unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="sect">{text}</div>', unsafe_allow_html=True)


def mel_figure(spec):
    import librosa.display as ld
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=130)
    fig.patch.set_facecolor("#070b0f")
    ax.set_facecolor("#070b0f")
    img = ld.specshow(spec["mel_db"], sr=spec["sr"], hop_length=spec["hop"], x_axis="time",
                      y_axis="mel", fmax=8000, cmap=EMERALD, ax=ax)
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    for a in (ax, cbar.ax):
        a.tick_params(colors="#8aa0ae", labelsize=8)
    ax.xaxis.label.set_color("#8aa0ae")
    ax.yaxis.label.set_color("#8aa0ae")
    for s in ax.spines.values():
        s.set_color("#17303a")
    ax.set_title("Mel-spectrogram", color="#a7f3d0", fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def highlight(text, matches):
    out, pos = [], 0
    for s, e, cat in sorted(matches):
        if e <= pos:
            continue
        s = max(s, pos)
        out.append(html.escape(text[pos:s]))
        out.append(f"<mark>{html.escape(text[s:e])}</mark>")
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


# --------------------------------------------------------------------------- #
# analysis plumbing
# --------------------------------------------------------------------------- #
def _cached(key, token: bytes, fn):
    """Re-use the last result while the same input is still in the widget."""
    h = hashlib.md5(token).hexdigest()
    slot = SS.get(key)
    if slot and slot[0] == h:
        return slot[1]
    res = fn()
    SS[key] = (h, res)
    return res


def _md_safe(text: str) -> str:
    """Escape markdown/LaTeX control chars so ASR output (e.g. "$500", "*", "_") renders literally."""
    return re.sub(r"([\\`*_$\[\]~#>])", r"\\\1", text)


_SEVERITY = {"INCONCLUSIVE": 0, "AUTHENTIC": 1, "SUSPICIOUS": 2, "SYNTHETIC": 3}


def _tmp_run(data: bytes, suffix: str, fn):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    try:
        return fn(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def register(kind, sha, res, tag=""):
    """Log a NEW verification once: audit chain entry. Never stores media or transcript text."""
    key = (kind, sha, tag)
    if key in SS["_logged"] or res is None:
        return
    SS["_logged"].add(key)
    action = policy.recommend({"verdict": res["verdict"], "reasons": []})["action"]
    audit.append(SS["audit"], mode=SS.get("mode", ""), kind=kind, input_sha256=sha,
                 input_tag=tag, action=action, **audit.summarize(res))


def list_samples(exts):
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in exts)


def expected_label(name):
    n = name.lower()
    if n.startswith(("real", "genuine", "authentic", "human")):
        return "AUTHENTIC"
    if n.startswith(("clone", "fake", "deepfake", "synth", "tts", "swap")):
        return "SYNTHETIC"
    return None


def sample_buttons(panel, exts, ns):
    files = list_samples(exts)
    if not files:
        st.caption("No sample clips found. Put your own real / cloned files in the `samples/` folder "
                   "(see samples/README.txt) and they appear here.")
        return
    st.caption("Or run a bundled sample (a genuine analysis of the real file):")
    cols = st.columns(min(len(files), 3))
    for i, f in enumerate(files[:6]):
        if cols[i % len(cols)].button(f.stem[:20], key=f"{ns}_smp_{panel}_{f.name}_{nonce(panel)}"):
            clear_panel(panel)
            SS[f"sample_{panel}"] = str(f)
            st.rerun()


def sample_check(path, res):
    exp = expected_label(Path(path).name)
    if exp and res and res.get("ok"):
        got = res["verdict"]
        hit = (exp == "AUTHENTIC" and got == "AUTHENTIC") or (exp == "SYNTHETIC" and got != "AUTHENTIC")
        st.caption(f"Sample label: expected **{exp}** · detector says **{got}** {'✅' if hit else '❌'}")


# results shared between panels: R["audio"] = (sha, res, tag) ...
R = {}

# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def voice_panel(ns, snr_db):
    n = nonce("voice")
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.markdown('<div class="ptitle">Capture voice</div>', unsafe_allow_html=True)
        rec = st.audio_input("Record a live voice sample", key=f"{ns}_rec_{n}")
        up = st.file_uploader("…or upload a sample", key=f"{ns}_aup_{n}", type=[e[1:] for e in AUDIO_EXT])
        sample_buttons("voice", AUDIO_EXT, ns)
        if st.button("🗑 Clear voice input", key=f"{ns}_clr_voice_{n}"):
            clear_panel("voice")
            st.rerun()
        st.caption("Tip: 3–6 s of natural speech in a quiet room gives the most reliable read.")

    sample = SS.get("sample_voice")
    if rec is not None:
        data, suffix, label = rec.getvalue(), ".wav", "live recording"
    elif up is not None:
        data, suffix, label = up.getvalue(), os.path.splitext(up.name)[1] or ".wav", f"upload: {up.name}"
    elif sample:
        data, suffix, label = Path(sample).read_bytes(), Path(sample).suffix, f"sample: {Path(sample).name}"
    else:
        data = None

    with right:
        if data is None:
            R.pop("audio", None)
            SS["last_audio"] = None
            empty_state("Awaiting voice signal. Record a sample or upload a file.")
            return

        sha = audit.sha256_bytes(data)
        SS["last_audio"] = (data, suffix)

        # ------------------------------------------------------------------ #
        # DUAL-ENGINE PIPELINE: ACOUSTIC DSP + LOCAL STT & NLP               #
        # ------------------------------------------------------------------ #
        def _full_pipeline(path):
            # 1. Physical / acoustic checks: how does it SOUND? (never raises)
            res_d = detector.analyze_audio_stream(path, noise_snr_db=snr_db)
            acoustic = float(res_d.get("risk") or 0.0)

            # 2. Local speech-to-text: what was SAID? (always attempted, even if step 1 rejected the clip)
            transcript, asr_err = nlp_analyzer.transcribe(path)
            res_d["transcript"], res_d["asr_error"] = transcript, asr_err
            res_d["acoustic_risk"], res_d["semantic_risk"], res_d["nlp"] = acoustic, 0.0, None
            if not transcript:
                return res_d

            # 3. Semantic intent on the transcript
            nlp_res = nlp_analyzer.analyze_text(transcript)
            res_d["nlp"] = nlp_res
            if not nlp_res.get("ok"):
                return res_d
            semantic = float(nlp_res.get("risk") or 0.0)
            res_d["semantic_risk"] = semantic
            intent = nlp_res.get("verdict") in ("SCAM-LIKE", "SUSPICIOUS")   # e.g. "ATM PIN", "OTP"

            # 4. Fusion: 70 % acoustic + 30 % semantic; scam intent can never score below 0.45.
            #    If the acoustic engine rejected the clip, only the semantic evidence is available.
            fused = (0.70 * acoustic + 0.30 * semantic) if res_d.get("ok") else semantic
            if intent:
                fused = max(fused, 0.45)
            fused = float(min(fused, 1.0))

            conf = float(res_d.get("confidence") or 0.0)
            if intent:
                conf = max(conf, float(nlp_res.get("confidence") or 0.0))
            verdict = detector._verdict(fused, conf)
            if intent and verdict in ("AUTHENTIC", "INCONCLUSIVE"):
                verdict = "SUSPICIOUS"
            # fusion may raise the verdict but must never hide what the acoustic engine found
            if _SEVERITY.get(res_d.get("verdict"), 0) > _SEVERITY.get(verdict, 0):
                verdict = res_d["verdict"]

            if not res_d.get("ok") and intent:            # intent alone is enough to show a result
                res_d["notes"].append(f"Acoustic check skipped: {res_d.get('error')}")
                res_d["ok"], res_d["error"] = True, None
            res_d.update(risk=fused, confidence=conf, verdict=verdict)
            # `findings()` renders res["flags"], so NLP evidence must be appended THERE
            res_d["flags"] = list(res_d.get("flags", [])) + [f"Intent: {f}" for f in nlp_res.get("flags", [])]
            return res_d

        # Cache key: audio bytes + settings + whether ASR is installed. Prefix `_v3` retires results cached
        # by older builds (which had no transcript) so stale transcript-less entries are never re-served.
        res = _cached(
            "res_audio_v3",
            data + suffix.encode() + str(snr_db).encode() + (b"|asr1" if nlp_analyzer.asr_available() else b"|asr0"),
            lambda: _tmp_run(data, suffix, _full_pipeline),
        )

        clean = None
        if snr_db is not None:
            clean = _cached(
                "res_audio_clean", 
                data + suffix.encode(),
                lambda: _tmp_run(data, suffix, lambda p: detector.analyze_audio_stream(p))
            )

        tag = f"noise{snr_db}" if snr_db is not None else ""
        register("audio", sha, res, tag)
        R["audio"] = (sha, res, tag)

        st.caption(f"Source: {label}")
        if res.get("error"):
            st.warning(res["error"])

        # Transcript is rendered explicitly so an empty / failed STT is visible, never silent
        transcript = res.get("transcript") or ""
        if transcript:
            st.info(f"🎙️ Transcribed Audio: “{_md_safe(transcript)}”")
        elif res.get("asr_error"):
            st.warning(f"⚠️ Speech-to-text: {res['asr_error']}")
        elif not res.get("ok") and res.get("error"):
            pass                                          # already shown as the warning above
        else:
            st.caption("🎙️ No speech recognised in this clip (silence, music or noise only).")

        banner(res)

        if clean is not None and clean.get("ok") and res.get("ok"):
            st.info(
                f"Robustness: clean {round(clean['risk'] * 100)}% → with {snr_db} dB noise "
                f"{round(res.get('acoustic_risk', res['risk']) * 100)}%  "
                f"(Δ {round((res.get('acoustic_risk', res['risk']) - clean['risk']) * 100):+d} pts, acoustic engine only)"
            )

        m = res.get("metrics", {})
        cards([
            latency_card(res),
            card("Pitch Deviation", fmt(m.get("pitch_dev_st"), ".2f"), "st",
                 f"mean F0 {fmt(m.get('pitch_mean_hz'), '.0f')} Hz"),
            card("Spectral Centroid", fmt(m.get("centroid_mean_hz"), ".0f"), "Hz",
                 f"variation CV {fmt(m.get('centroid_cv'), '.2f')}"),
            card("Zero-Crossing Rate", fmt(m.get("zcr_mean"), ".3f"), "",
                 f"breaths/10s {fmt(m.get('breaths_per_10s'), '.1f')}"),
            card("Pause Noise Floor", fmt(m.get("pause_floor_gap_db"), ".0f"), "dB",
                 "very high = digital silence"),
        ])

        findings(res)

        if sample and rec is None and up is None:
            sample_check(sample, res)

    if res.get("spectrogram") is not None and SPEC_OK:
        st.markdown('<div class="ptitle">Live spectral view</div>', unsafe_allow_html=True)
        fig = mel_figure(res["spectrogram"])
        st.pyplot(fig)
        plt.close(fig)


def text_panel(ns):
    n = nonce("text")
    key = f"{ns}_txt_{n}"
    st.markdown('<div class="ptitle">What is being asked? · transcript NLP</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        def _set(k, v):
            SS[k] = v

        b1, b2 = st.columns(2)
        b1.button("Example: scam call", key=f"{ns}_ex1_{n}", on_click=_set, args=(key, EXAMPLE_SCAM))
        b2.button("Example: normal call", key=f"{ns}_ex2_{n}", on_click=_set, args=(key, EXAMPLE_OK))
        if nlp_analyzer.asr_available() and SS.get("last_audio"):
            def _asr():
                d, sfx = SS["last_audio"]
                txt, err = _tmp_run(d, sfx, nlp_analyzer.transcribe)
                SS[key] = txt or ""
                SS["asr_err"] = err or (None if txt else "No speech recognised in the voice sample.")
            st.button("🎧 Transcribe the voice sample (local)", key=f"{ns}_asr_{n}", on_click=_asr)
            if SS.get("asr_err"):
                st.warning(SS["asr_err"])
        txt = st.text_area("Paste or type what was said on the call", key=key, height=130,
                           placeholder="e.g. “Your account will be blocked. Share the OTP right now…”")
        if st.button("🗑 Clear transcript", key=f"{ns}_clr_text_{n}"):
            clear_panel("text")
            st.rerun()
        st.caption("Only a SHA-256 of the text is logged. The transcript itself is never stored.")
    with right:
        clean_txt = (txt or "").strip()[:20000]
        if not clean_txt:
            R.pop("nlp", None)
            empty_state("Awaiting transcript. Paste what the caller said (or try an example).")
            return
        res = _cached("res_nlp", clean_txt.encode(), lambda: nlp_analyzer.analyze_text(clean_txt))
        sha = audit.sha256_bytes(clean_txt.encode())
        register("nlp", sha, res)
        R["nlp"] = (sha, res, "")
        if res["error"]:
            st.warning(res["error"])
            return
        banner(res, title=BANNERS[res["verdict"]][1])
        cards([latency_card(res, 50),
               card("Tactics found", str(res["metrics"].get("tactics_found", 0)), "", "of 9 categories"),
               card("Words", str(res["metrics"].get("words", 0)), "", "analysed locally")])
        st.markdown(f'<div class="transcript">{highlight(clean_txt, res["matches"])}</div>',
                    unsafe_allow_html=True)
        findings(res)


def video_panel(ns, jpeg_q):
    n = nonce("video")
    lv = SS["lv"]
    left, right = st.columns([1, 1.5], gap="large")
    snap = up_img = up_vid = None
    with left:
        st.markdown('<div class="ptitle">Capture face</div>', unsafe_allow_html=True)
        if lv["phase"] == "idle":
            snap = st.camera_input("Capture a live frame", key=f"{ns}_cam_{n}")
            up_img = st.file_uploader("…or upload a face image", key=f"{ns}_iup_{n}", type=["jpg", "jpeg", "png"])
            up_vid = st.file_uploader("…or upload a video call recording (.mp4, .mov)", key=f"{ns}_vup_{n}",
                                      type=["mp4", "mov", "avi"])
            sample_buttons("video", IMG_EXT + VID_EXT, ns)
            if snap is not None:
                if st.button("🎯 Start liveness challenge", key=f"{ns}_lvstart_{n}", type="primary"):
                    SS["lv"] = {"phase": "challenge", "baseline": snap.getvalue(),
                                "issued_at": None, **liveness.new_challenge()}
                    bump("video")
                    st.rerun()
                st.caption("Uses this frame as the neutral reference, then asks for a head turn at a random moment.")
            if st.button("🗑 Clear video input", key=f"{ns}_clr_video_{n}"):
                clear_panel("video")
                st.rerun()
            st.caption("Tip: face the camera, even lighting, face at least ~1/3 of the frame.")
        else:
            challenge_ui(ns, n)

    # ---- choose the source of the video analysis ------------------------------ #
    sample = SS.get("sample_video")
    kind, data, suffix, label = None, None, "", ""
    if lv["phase"] != "idle":
        kind, data, label = "frame", lv["baseline"], "reference frame of the liveness challenge"
    elif snap is not None:
        kind, data, label = "frame", snap.getvalue(), "live camera frame"
    elif up_img is not None:
        kind, data, label = "frame", up_img.getvalue(), f"upload: {up_img.name}"
    elif up_vid is not None:
        kind, data, suffix, label = "file", up_vid.getvalue(), os.path.splitext(up_vid.name)[1] or ".mp4", \
            f"upload: {up_vid.name}"
    elif sample:
        p = Path(sample)
        data, label = p.read_bytes(), f"sample: {p.name}"
        kind, suffix = ("file", p.suffix) if p.suffix.lower() in VID_EXT else ("frame", "")

    with right:
        if data is None:
            R.pop("video", None)
            empty_state("Awaiting video signal. Capture a frame or upload an image / video.")
            return
        sha = audit.sha256_bytes(data)
        qtok = str(jpeg_q).encode()
        if kind == "file":
            st.video(data)
            res = _cached("res_video", data + suffix.encode() + qtok,
                          lambda: _tmp_run(data, suffix, lambda p: detector.analyze_video_file(p, jpeg_quality=jpeg_q)))
        else:
            res = _cached("res_video", data + qtok, lambda: detector.analyze_video_frame(data, jpeg_quality=jpeg_q))
        clean = None
        if jpeg_q is not None:
            if kind == "file":
                clean = _cached("res_video_clean", data + suffix.encode(),
                                lambda: _tmp_run(data, suffix, lambda p: detector.analyze_video_file(p)))
            else:
                clean = _cached("res_video_clean", data, lambda: detector.analyze_video_frame(data))
        tag = f"jpeg{jpeg_q}" if jpeg_q is not None else ""
        register("video", sha, res, tag)
        R["video"] = (sha, res, tag)
        st.caption(f"Source: {label}")
        if res["error"]:
            st.warning(res["error"])
        banner(res)
        if clean is not None and clean["ok"] and res["ok"]:
            st.info(f"Robustness: clean {round(clean['risk'] * 100)}% → JPEG quality {jpeg_q} "
                    f"{round(res['risk'] * 100)}%  (Δ {round((res['risk'] - clean['risk']) * 100):+d} pts)")
        m = res["metrics"]
        is_file = kind == "file"
        cards([
            latency_card(res, 4000 if is_file else detector.LATENCY_BUDGET_MS,
                         "file: keyframes + 1 s window" if is_file else None),
            card("Landmarks", fmt(m.get("landmark_count"), "d"), "", "of 468 face-mesh points"),
            card("Eye-Distance Ratio", fmt(m.get("eye_distance_ratio"), ".2f"), "",
                 f"symmetry Δ {fmt(m.get('eye_symmetry'), '.2f')}"),
            card("Boundary Jitter", fmt(m.get("boundary_jitter"), ".3f"), "", "face-contour roughness"),
            card("Mask-Edge Ratio", fmt(m.get("seam_ratio"), ".2f"), "", "low = blended seam"),
            card("Skin-Tone Δ", fmt(m.get("skin_tone_delta"), ".1f"), "", "face vs neck chroma"),
        ])
        if res["annotated_png"]:
            st.image(res["annotated_png"], caption="Face-mesh overlay (processed locally)")
        findings(res)
        if is_file and res.get("timeline"):
            st.markdown('<div class="ptitle">Per-keyframe risk</div>', unsafe_allow_html=True)
            st.line_chart(pd.DataFrame({"risk %": [v * 100 for v in res["timeline"]]}))
        if sample and lv["phase"] == "idle" and snap is None and up_img is None and up_vid is None:
            sample_check(sample, res)


def challenge_ui(ns, n):
    """Liveness challenge state machine: challenge -> done. Uses ONE camera widget at a time."""
    lv = SS["lv"]
    if lv["phase"] == "challenge":
        if lv.get("issued_at") is None:
            ph = st.empty()
            ph.markdown('<div class="challenge"><span>Get ready: keep facing the camera.<br>'
                        'An instruction will appear at a random moment…</span></div>', unsafe_allow_html=True)
            time.sleep(lv["delay_s"])                       # random delay: cannot be pre-recorded
            lv["issued_at"] = time.time()
            ph.empty()
        st.markdown(f'<div class="challenge"><b>TURN YOUR HEAD TO YOUR {lv["direction"]}</b>'
                    f'<span>then take the photo straight away</span></div>', unsafe_allow_html=True)
        shot = st.camera_input("Take the photo once you have turned", key=f"{ns}_chal_{n}")
        if shot is not None:
            resp = shot.getvalue()
            rt = time.time() - lv["issued_at"]
            lv["resp_sha"] = audit.sha256_bytes(resp)
            lv["result"] = liveness.evaluate(lv["baseline"], resp, lv["direction"], rt,
                                             mirrored=SS.get("mirrored", True))
            lv["phase"] = "done"
            st.rerun()
        if st.button("Cancel challenge", key=f"{ns}_lvcancel_{n}"):
            SS["lv"] = {"phase": "idle"}
            bump("video")
            st.rerun()
    else:  # done
        res = lv["result"]
        if res["error"]:
            st.warning(res["error"])
        else:
            passed = res["verdict"] == "AUTHENTIC"
            banner(res, title="LIVENESS PASSED" if passed else "LIVENESS FAILED",
                   sub=("Head turn performed correctly within a human response time" if passed
                        else "The challenge response looks wrong or synthetic"))
            m = res["metrics"]
            cards([card("Response time", fmt(m.get("response_time_s"), ".1f"), "s", "human window ≈ 1–20 s"),
                   card("Yaw change", fmt(m.get("turn_toward_target"), "+.2f"), "", f"toward {m.get('challenge','').lower()}"),
                   latency_card(res)])
            findings(res)
        c1, c2 = st.columns(2)
        if c1.button("🔁 New challenge", key=f"{ns}_lvagain_{n}"):
            SS["lv"] = {"phase": "challenge", "baseline": lv["baseline"], "issued_at": None,
                        **liveness.new_challenge()}
            bump("video")
            st.rerun()
        if c2.button("← Back to capture", key=f"{ns}_lvback_{n}"):
            SS["lv"] = {"phase": "idle"}
            bump("video")
            st.rerun()
        if not res["error"]:
            register("liveness", lv["resp_sha"], res)
            R["liveness"] = (lv["resp_sha"], res, "")


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Warming up local models…")
def _warm():
    detector.warm_up()
    return True


_warm()

with st.sidebar:
    st.markdown('<div class="ptitle">Engine status</div>', unsafe_allow_html=True)
    for name, ok in detector.engine_status().items():
        st.markdown(f"{'🟢' if ok else '🔴'} `{name}`")
    st.markdown(f"{'🟢' if nlp_analyzer.asr_available() else '⚪'} `speech-to-text (optional)`")

    st.markdown('<div class="ptitle" style="margin-top:18px">Robustness test</div>', unsafe_allow_html=True)
    rob_on = st.toggle("Degrade inputs before analysis", key="rob_on",
                       help="Adds noise to audio and lowers JPEG quality, then shows how the score moves.")
    snr_val = st.slider("Audio noise (SNR dB)", 5, 40, 15, disabled=not rob_on)
    jpg_val = st.slider("JPEG quality", 15, 90, 35, disabled=not rob_on)
    SNR_DB = float(snr_val) if rob_on else None
    JPEG_Q = int(jpg_val) if rob_on else None

    st.markdown('<div class="ptitle" style="margin-top:18px">Camera</div>', unsafe_allow_html=True)
    st.toggle("Camera image is mirrored", value=True, key="mirrored",
              help="Flip this if the liveness check rejects a head turn you did correctly.")

    st.markdown('<div class="ptitle" style="margin-top:18px">Session</div>', unsafe_allow_html=True)
    if st.button("↺ Reset all inputs"):
        reset_all()
        st.rerun()
    if st.button("Clear timeline & audit log"):
        clear_history()
        st.rerun()
    st.caption("All inference runs locally. No audio or video leaves this machine.")
    st.caption(f"Latency budget: {detector.LATENCY_BUDGET_MS} ms per single analysis.")
    st.caption("Heuristic engine: measure it on your own clips with `python evaluate.py`.")

st.markdown(
    '<div class="hero"><div class="logo">SENTIN<span>AI</span></div>'
    '<div class="tag">Real-Time Multi-Modal Deepfake &amp; Synthetic Voice Defense Engine</div>'
    '<div class="pills"><span>EDGE-FIRST</span><span>100% LOCAL</span><span>ZERO-TRUST</span>'
    '<span>VOICE + VIDEO + LIVENESS + NLP</span></div></div>', unsafe_allow_html=True)

mode = st.radio("Mode", MODES, horizontal=True, key="mode", on_change=reset_all,
                label_visibility="collapsed")
fused_slot = st.empty()

if mode == MODES[0]:
    voice_panel("voice", SNR_DB)
    section("Conversation intent (NLP)")
    text_panel("voice")
elif mode == MODES[1]:
    video_panel("video", JPEG_Q)
else:
    section("1 · Voice")
    voice_panel("full", SNR_DB)
    section("2 · Transcript")
    text_panel("full")
    section("3 · Video & liveness")
    video_panel("full", JPEG_Q)

# ---- fused banner + recommended action (rendered at the top) --------------------- #
res_of = {k: v[1] for k, v in R.items()}
fused = detector.fuse_all(res_of.get("audio"), res_of.get("video"), res_of.get("liveness"), res_of.get("nlp"))
rec = policy.recommend(fused, res_of.get("nlp"))
with fused_slot.container():
    if not res_of or fused["modalities"] == 0:
        banner({"verdict": "INCONCLUSIVE", "risk": 0.0, "confidence": 0.0},
               sub="Awaiting signal: record a voice sample, capture a frame or paste a transcript")
        policy_card(policy.recommend(fused))
    else:
        names = {"audio": "voice", "video": "video", "liveness": "liveness", "nlp": "transcript"}
        banner(fused, sub="Fused threat level · " + " + ".join(names[k] for k in res_of))
        policy_card(rec)

# ---- timeline + audit entry when the combined picture changes -------------------- #
sig = tuple(sorted((k, v[0], v[2]) for k, v in R.items()))
if R and fused["modalities"] > 0 and sig != SS["fused_sig"]:
    SS["fused_sig"] = sig
    SS["timeline"].append({"risk": round(fused["risk"] * 100, 1), "confidence": round(fused["confidence"] * 100, 1)})
    audit.append(SS["audit"], mode=mode, kind="fused", inputs=[f"{k}:{v[0]}" for k, v in sorted(R.items())],
                 verdict=fused["verdict"], risk=fused["risk"], confidence=fused["confidence"],
                 synthetic_risk=fused["synthetic_risk"], intent_risk=fused["intent_risk"],
                 action=rec["action"], level=rec["level"], reasons=fused["reasons"])
elif not R:
    SS["fused_sig"] = None

# ---- explainability ---------------------------------------------------------------- #
section("Why was this flagged?")
rows = {}
labels = {"audio": "Voice", "video": "Video", "liveness": "Liveness", "nlp": "Text"}
for k, (_s, r, _t) in R.items():
    for cue, val in (r.get("contributions") or {}).items():
        if val > 0.0005:
            rows[f"{labels[k]} · {cue}"] = round(val * 100, 1)
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown('<div class="ptitle">Cue contribution (risk points)</div>', unsafe_allow_html=True)
    if rows:
        df = pd.DataFrame({"risk points": pd.Series(rows).sort_values(ascending=False)})
        st.bar_chart(df, horizontal=True)
    else:
        st.caption("No cue is contributing risk yet." if R else "Run an analysis to see which cues drive the score.")
with c2:
    st.markdown('<div class="ptitle">Risk timeline (this session)</div>', unsafe_allow_html=True)
    if SS["timeline"]:
        tl = pd.DataFrame(SS["timeline"])
        tl.index = range(1, len(tl) + 1)
        tl["suspicious ≥"] = 40
        tl["synthetic ≥"] = 60
        st.line_chart(tl[["risk", "suspicious ≥", "synthetic ≥"]])
    else:
        st.caption("Each new combined result adds a point here.")

# ---- tamper-evident audit ---------------------------------------------------------- #
section("Tamper-evident audit report")
chain = SS["audit"]
ok, bad, msg = audit.verify(chain)
if chain:
    (st.success if ok else st.error)(("✅ " if ok else "❌ ") + msg + f" · head {chain[-1]['hash'][:16]}…")
    st.dataframe(pd.DataFrame([{"#": e["seq"], "time (UTC)": e["ts"], "kind": e["kind"],
                                "verdict": e.get("verdict"), "risk %": round((e.get("risk") or 0) * 100),
                                "action": e.get("action"), "input SHA-256": (e.get("input_sha256") or "")[:16],
                                "hash": e["hash"][:12]} for e in chain]), hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download audit report (JSON)", data=audit.export_json(chain, {"mode": mode}),
                       file_name="sentinai_audit.json", mime="application/json")
    if d2.button("Demo: tamper with entry #0 and re-verify"):
        fake = copy.deepcopy(chain)
        fake[0]["risk"] = 0.0
        fake[0]["verdict"] = "AUTHENTIC"
        _ok, _bad, _msg = audit.verify(fake)
        st.error(f"Tampered copy → {'valid' if _ok else 'REJECTED'}: {_msg}")
    st.caption("Only SHA-256 digests of inputs are stored: no audio, video or transcript text. "
               "Verify an exported file offline with `python audit.py verify sentinai_audit.json`.")
else:
    st.caption("Every verification is logged here with a timestamp, scores, flags, recommended action and the "
               "SHA-256 of the input, each entry chained to the previous one.")

st.markdown('<div class="foot">SENTINAI · EDGE-FIRST · ZERO-TRUST · ALL ANALYSIS RUNS LOCALLY</div>',
            unsafe_allow_html=True)
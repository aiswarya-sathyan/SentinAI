
from __future__ import annotations

from typing import Any, Dict, List, Optional

# level -> response. `css` matches the banner colour classes in app.py.
POLICY: Dict[str, Dict[str, Any]] = {
    "LOW": {
        "action": "ALLOW",
        "css": "ok",
        "headline": "Proceed: no synthetic signature detected",
        "steps": [
            "Allow the call / session to continue.",
            "Keep passive monitoring on for the rest of the session.",
        ],
        "escalate": False,
    },
    "ELEVATED": {
        "action": "STEP-UP VERIFICATION",
        "css": "warn",
        "headline": "Verify out-of-band before trusting this identity",
        "steps": [
            "Send a one-time passcode (OTP) to the user's registered device.",
            "Or call back on the number already on file (never a number given in the call).",
            "Run the live head-turn challenge: replayed video and many real-time face swaps fail it.",
            "Hold any payment, credential change or data export until verified.",
        ],
        "escalate": False,
    },
    "CRITICAL": {
        "action": "BLOCK & ESCALATE",
        "css": "bad",
        "headline": "Block the action and hand over to a human analyst",
        "steps": [
            "Block the transaction / privileged action immediately.",
            "Escalate to a human fraud analyst with the audit report attached.",
            "Do not disclose OTPs, PINs or account details on this channel.",
            "Preserve the tamper-evident audit log as evidence.",
        ],
        "escalate": True,
    },
    "UNKNOWN": {
        "action": "COLLECT MORE SIGNAL",
        "css": "idle",
        "headline": "Not enough signal: do not treat this as a pass",
        "steps": [
            "Re-capture in better light / a quieter room, or record a longer clip.",
            "Until then, treat the identity as unverified.",
        ],
        "escalate": False,
    },
}

VERDICT_TO_LEVEL = {"AUTHENTIC": "LOW", "BENIGN": "LOW", "SUSPICIOUS": "ELEVATED",
                    "SCAM-LIKE": "ELEVATED", "SYNTHETIC": "CRITICAL", "INCONCLUSIVE": "UNKNOWN"}


def recommend(fused: Dict[str, Any], nlp: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map a fused result (detector.fuse_all) to a response. Pure function, no UI."""
    level = VERDICT_TO_LEVEL.get(fused.get("verdict", "INCONCLUSIVE"), "UNKNOWN")
    pol = POLICY[level]
    steps: List[str] = list(pol["steps"])
    reasons: List[str] = list(fused.get("reasons", []))

    # content-aware extras: a scripted request for secrets deserves an explicit instruction
    if nlp and nlp.get("ok") and nlp.get("risk", 0) >= 0.5 and level in ("ELEVATED", "CRITICAL"):
        steps.append("The conversation asks for money/credentials under pressure: "
                     "refuse and verify through an official channel.")
    return {"level": level, "action": pol["action"], "css": pol["css"],
            "headline": pol["headline"], "steps": steps, "escalate": pol["escalate"],
            "reasons": reasons}

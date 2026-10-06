from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

GENESIS = "0" * 64
KEEP = ("verdict", "risk", "confidence", "latency_ms", "subscores", "flags", "notes")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def entry_hash(entry: Dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(_canon(body).encode("utf-8")).hexdigest()


def _round(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_round(x) for x in v]
    return v


def summarize(res: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only scores/flags of an analyzer result (drops spectrograms and images)."""
    return _round({k: res.get(k) for k in KEEP if k in res})


def append(chain: List[Dict[str, Any]], **fields: Any) -> Dict[str, Any]:
    """Add one entry to `chain` (a plain list, e.g. kept in st.session_state)."""
    entry: Dict[str, Any] = {
        "seq": len(chain),
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    }
    entry.update(_round(fields))
    entry["prev_hash"] = chain[-1]["hash"] if chain else GENESIS
    entry["hash"] = entry_hash(entry)
    chain.append(entry)
    return entry


def verify(chain: List[Dict[str, Any]]) -> Tuple[bool, Optional[int], str]:
    """-> (valid, index_of_first_bad_entry, message)"""
    prev = GENESIS
    for i, e in enumerate(chain):
        if e.get("seq") != i:
            return False, i, f"entry {i}: sequence number altered or entry removed"
        if e.get("prev_hash") != prev:
            return False, i, f"entry {i}: link to the previous entry is broken"
        if entry_hash(e) != e.get("hash"):
            return False, i, f"entry {i}: content no longer matches its hash"
        prev = e["hash"]
    return True, None, f"chain intact ({len(chain)} entries)"


def export_json(chain: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> str:
    ok, bad, msg = verify(chain)
    doc = {"report": "SentinAI verification audit log", "version": 1,
           "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
           "privacy": "Only SHA-256 digests of inputs are stored; no audio, video or transcript text.",
           "chain_valid_at_export": ok, "head_hash": chain[-1]["hash"] if chain else GENESIS,
           "entry_count": len(chain), "meta": meta or {}, "entries": chain}
    return json.dumps(doc, indent=2, ensure_ascii=False, default=str)


def verify_export(text: str) -> Tuple[bool, Optional[int], str]:
    doc = json.loads(text)
    chain = doc["entries"] if isinstance(doc, dict) else doc
    return verify(chain)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "verify":
        with open(sys.argv[2], encoding="utf-8") as fh:
            ok, bad, msg = verify_export(fh.read())
        print(("VALID: " if ok else "TAMPERED: ") + msg)
        sys.exit(0 if ok else 1)
    print("usage: python audit.py verify <report.json>")

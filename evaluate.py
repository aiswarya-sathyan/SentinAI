import argparse
import os
import sys

import numpy as np

import detector

AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
IMG = {".jpg", ".jpeg", ".png"}
VID = {".mp4", ".mov", ".avi"}


def score(path, noise, jpeg):
    ext = os.path.splitext(path)[1].lower()
    if ext in AUDIO:
        return "audio", detector.analyze_audio_stream(path, noise_snr_db=noise)
    if ext in IMG:
        with open(path, "rb") as fh:
            return "image", detector.analyze_video_frame(fh.read(), jpeg_quality=jpeg)
    if ext in VID:
        return "video", detector.analyze_video_file(path, jpeg_quality=jpeg)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_data")
    ap.add_argument("--noise", type=float, default=None, help="audio SNR dB for the robustness run")
    ap.add_argument("--jpeg", type=int, default=None, help="JPEG quality for the robustness run")
    a = ap.parse_args()

    rows = []                                    # (kind, label(1=fake), risk, verdict, name)
    for label, folder in ((0, "real"), (1, "fake")):
        d = os.path.join(a.root, folder)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            kind, res = score(os.path.join(d, fn), a.noise, a.jpeg)
            if res is None:
                continue
            if not res["ok"]:
                print(f"  skipped {fn}: {res['error']}")
                continue
            rows.append((kind, label, res["risk"], res["verdict"], fn))
            print(f"{folder:5s} {kind:6s} risk {res['risk']:.2f}  {res['verdict']:10s} {fn}")
    if not rows:
        sys.exit("No analysable files found in eval_data/real and eval_data/fake.")

    for kind in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == kind]
        y = np.array([r[1] for r in sub])
        s = np.array([r[2] for r in sub])
        if len(set(y)) < 2:
            print(f"\n[{kind}] need both real and fake files for metrics ({len(sub)} files)")
            continue
        pred = np.array([r[3] != "AUTHENTIC" for r in sub])        # SUSPICIOUS or worse = flagged
        acc = float((pred == y).mean())
        far = float(((~pred) & (y == 1)).sum() / (y == 1).sum())   # fakes let through
        frr = float((pred & (y == 0)).sum() / (y == 0).sum())      # real users flagged
        print(f"\n[{kind}] n={len(sub)} ({int(y.sum())} fake / {int((1 - y).sum())} real)")
        print(f"  at current verdict thresholds: accuracy {acc:.1%} | fakes missed {far:.1%} | real flagged {frr:.1%}")
        try:
            from sklearn.metrics import roc_auc_score
            print(f"  ROC-AUC {roc_auc_score(y, s):.3f}")
        except Exception:
            pass
        best = max(((float(((s >= t) == y).mean()), float(t)) for t in np.linspace(0.05, 0.95, 91)))
        print(f"  best single risk threshold on this data: {best[1]:.2f} (accuracy {best[0]:.1%})")


if __name__ == "__main__":
    main()

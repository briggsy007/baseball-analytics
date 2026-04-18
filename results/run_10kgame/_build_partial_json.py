"""
Reconstruct pitchgpt_ablation_metrics.json from the ablation log if the
run was killed before count_only completed.  Parses the human-readable
log lines written by scripts/pitchgpt_ablation.py.
"""
from __future__ import annotations
import json
import math
import re
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parent / "_ablation.log"
OUT = Path(__file__).resolve().parent / "pitchgpt_ablation_metrics.json"

# Mirror the baseline from the 1K run.
THRESH = 10.0

EPOCH_RE = re.compile(
    r"\[(?P<name>full|tokens-only|count-only)\] epoch (?P<ep>\d+)/(?P<total>\d+)\s+"
    r"train=(?P<tr>[0-9.]+) \(ppl (?P<tppl>[0-9.]+)\)\s+"
    r"val=(?P<vl>[0-9.]+) \(ppl (?P<vppl>[0-9.]+)\)\s+(?P<dt>[0-9.]+)s"
)
BEST_RE = re.compile(
    r"\[(?P<name>full|tokens-only|count-only)\] BEST epoch (?P<best>\d+)\s+"
    r"test_loss=(?P<tl>[0-9.]+)\s+test_ppl=(?P<tp>[0-9.]+)\s+\((?P<wall>[0-9.]+)s total\)"
)
DSBUILD_RE = re.compile(r"datasets built in (?P<dt>[0-9.]+)s \(train=(?P<tr>\d+) val=(?P<vl>\d+) test=(?P<te>\d+)\)")


def name_to_key(n: str) -> str:
    return {"full": "full", "tokens-only": "tokens_only", "count-only": "count_only"}[n]


def main() -> int:
    text = LOG.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    variants: dict[str, dict] = {}
    ds_info = {}
    for ln in lines:
        m = DSBUILD_RE.search(ln)
        if m:
            ds_info = {
                "build_sec": float(m.group("dt")),
                "n_train": int(m.group("tr")),
                "n_val": int(m.group("vl")),
                "n_test": int(m.group("te")),
            }
        m = EPOCH_RE.search(ln)
        if m:
            key = name_to_key(m.group("name"))
            v = variants.setdefault(key, {"history": [], "mode": key, "params": 1398562})
            v["history"].append({
                "epoch": int(m.group("ep")),
                "train_loss": float(m.group("tr")),
                "val_loss": float(m.group("vl")),
                "train_perplexity": float(m.group("tppl")),
                "val_perplexity": float(m.group("vppl")),
                "wall_clock_sec": float(m.group("dt")),
            })
            v["train_loss_final"] = float(m.group("tr"))
            v["val_loss_final"] = float(m.group("vl"))
        m = BEST_RE.search(ln)
        if m:
            key = name_to_key(m.group("name"))
            v = variants.setdefault(key, {"history": [], "mode": key, "params": 1398562})
            v["test_loss"] = float(m.group("tl"))
            v["test_perplexity"] = float(m.group("tp"))
            v["epoch_best"] = int(m.group("best"))
            v["wall_clock_sec"] = float(m.group("wall"))

    if "full" not in variants or "test_perplexity" not in variants.get("full", {}):
        print("ERROR: no full-variant completion found in log.", file=sys.stderr)
        return 2

    full_ppl = variants["full"]["test_perplexity"]
    drops = {}
    for name, m in variants.items():
        if name == "full" or "test_perplexity" not in m:
            continue
        ppl = m["test_perplexity"]
        drops[name] = round(100.0 * (ppl - full_ppl) / ppl, 2) if ppl > 0 else 0.0

    if drops:
        best_abl = min(drops, key=lambda k: drops[k])
        best_pct = drops[best_abl]
        passed = best_pct >= THRESH
    else:
        best_abl, best_pct, passed = None, 0.0, False

    # Comparison vs 1K run.
    prior = Path(__file__).resolve().parent.parent / "run_5epoch" / "pitchgpt_ablation_metrics.json"
    prior_gap = None
    if prior.exists():
        prior_json = json.loads(prior.read_text(encoding="utf-8"))
        prior_gap = prior_json.get("drops_vs_full_pct", {}).get("tokens_only")

    gap_tok = drops.get("tokens_only")
    gap_cnt = drops.get("count_only")

    if best_abl is None:
        verdict = "FAIL — no ablations completed"
    else:
        verdict = (
            f"{'PASS' if passed else 'FAIL'} — full model beats best ablation "
            f"({best_abl}) by {best_pct:.2f}% (spec requires ≥{THRESH:.0f}%)."
        )

    out = {
        "seed": 42,
        "device": "cpu",
        "epochs": 5,
        "batch_size": 32,
        "lr": 0.001,
        "train_range": "2015-2022",
        "val_range": "2023",
        "test_range": "2024",
        "max_train_games": 10000,
        "max_val_games": 3000,
        "max_test_games": 3000,
        "n_train_sequences": ds_info.get("n_train"),
        "n_val_sequences": ds_info.get("n_val"),
        "n_test_sequences": ds_info.get("n_test"),
        "dataset_build_sec": ds_info.get("build_sec"),
        **variants,
        "drops_vs_full_pct": drops,
        "full_vs_tokens_only_pct": gap_tok,
        "full_vs_count_only_pct": gap_cnt,
        "spec_threshold_full_vs_any_pct": THRESH,
        "best_ablation": best_abl,
        "best_ablation_gap_pct": best_pct,
        "pass": bool(passed),
        "verdict": verdict,
        "comparison_vs_1k": {
            "1k_full_vs_tokens_pct": prior_gap,
            "10k_full_vs_tokens_pct": gap_tok,
            "gap_widened": (gap_tok is not None and prior_gap is not None and gap_tok > prior_gap),
        },
    }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(json.dumps({"verdict": verdict, "drops": drops, "full_ppl": full_ppl}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

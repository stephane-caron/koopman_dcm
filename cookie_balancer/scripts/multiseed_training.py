# SPDX-License-Identifier: Apache-2.0

"""Multi-seed training of the DCM observable, for variance bars on ω̂.

Trains K replicates of the DCM observable at fixed ``nb_lambda_steps``
values, varying only the random seed. This answers two questions of the
paper:

- Section 4.4: if we train K times, do we get the same eigenvalue?
- Appendix B.1: how does the eigenvalue depend on the number of lambda
  steps, beyond seed variation?

Default matrix: K=5 seeds × nb_lambda_steps ∈ {1, 5, 10, 20} = 20 runs.
At ~3 min/run on GPU (calibrated on cookie.h5, nls=10), the full
sweep is ~60 min. Outer loop is over seed and inner loop over nls,
so each completed seed delivers one value per number of lambda steps,
which is useful if the script is interrupted.

Usage::

    pixi run python scripts/multiseed_training.py

Output: ``sweep_runs/multiseed/{logs,models}/`` and
``sweep_runs/multiseed/results.json``. Resumable: already-completed
runs are skipped on restart.
"""

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from cookie_balancer import settings

# Training dataset
DATASET: Path = settings.DATA_DIR / "cookie.h5"

# Directory for logs, models, and the JSON summary
RESULTS_DIR: Path = Path("sweep_runs/multiseed")

# Run matrix: every seed is trained at every number of lambda steps
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
NB_LAMBDA_STEPS: tuple[int, ...] = (1, 5, 10, 20)
EPOCHS: int = 100  # per run

RUN_TIMEOUT_MIN: float = 20.0  # per-run timeout, in minutes
USE_GPU: bool = True  # train on CPU if False, much slower

MEANINGFUL_OMEGA_MIN = 1.0

FLOOR_RE = re.compile(r"val_com.*?floor.*?=\s*([0-9.eE+\-]+)")
LAMBDA_RE = re.compile(
    r"Multi-step loss:\s*nb_lambda_steps=(\d+)\s*→\s*λ=([0-9.eE+\-]+),\s*"
    r"nb_horizon_steps=(\d+)"
)
EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)/\d+\s+"
    r"train=(?P<train>[0-9.eE+\-]+)\s+"
    r"\(dcm=(?P<train_dcm>[0-9.eE+\-]+)\s+"
    r"com=(?P<train_com>[0-9.eE+\-]+)\)\s+\|\s+"
    r"val=(?P<val>[0-9.eE+\-]+)\s+"
    r"\(dcm=(?P<val_dcm>[0-9.eE+\-]+)\s+"
    r"com=(?P<val_com>[0-9.eE+\-]+)\)\s+\|\s+"
    r"ω=(?P<omega>[0-9.eE+\-]+)"
)


def make_runs(
    seeds: Sequence[int], nls_values: Sequence[int], epochs: int
) -> List[Dict[str, Any]]:
    """Build the run matrix: outer=seed, inner=nls.

    Each completed seed delivers one value of ω̂ per number of lambda steps.

    Args:
        seeds: Random seeds.
        nls_values: Numbers of lambda steps.
        epochs: Number of epochs per run.

    Returns:
        Configurations of the runs.
    """
    runs: List[Dict[str, Any]] = []
    for seed in seeds:
        for nls in nls_values:
            runs.append(
                {
                    "seed": int(seed),
                    "nb_lambda_steps": int(nls),
                    "epochs": int(epochs),
                }
            )
    return runs


def run_label(cfg: Dict[str, Any]) -> str:
    """Label of a run, used to name its files.

    Args:
        cfg: Run configuration, from :func:`make_runs`.

    Returns:
        Label of the run.
    """
    return f"multiseed_s{cfg['seed']}_nls{cfg['nb_lambda_steps']}"


def parse_log(log_text: str) -> List[Dict[str, float]]:
    """Parse the per-epoch losses and ω from a training log.

    Args:
        log_text: Training log.

    Returns:
        One dictionary per epoch, with its losses and ω.
    """
    out: List[Dict[str, float]] = []
    for m in EPOCH_RE.finditer(log_text):
        out.append(
            {
                "epoch": int(m.group("epoch")),
                "train": float(m.group("train")),
                "train_dcm": float(m.group("train_dcm")),
                "train_com": float(m.group("train_com")),
                "val": float(m.group("val")),
                "val_dcm": float(m.group("val_dcm")),
                "val_com": float(m.group("val_com")),
                "omega": float(m.group("omega")),
            }
        )
    return out


def parse_floor(log_text: str) -> float | None:
    """Parse the trivial-prediction floor of the CoM loss from a log.

    Args:
        log_text: Training log.

    Returns:
        Floor of the validation CoM loss, or None if absent.
    """
    m = FLOOR_RE.search(log_text)
    return float(m.group(1)) if m else None


def parse_lambda_info(log_text: str) -> Dict[str, float] | None:
    """Parse the parameters of the multi-step loss from a training log.

    Args:
        log_text: Training log.

    Returns:
        Number of lambda steps, λ and truncation horizon, or None if
        absent.
    """
    m = LAMBDA_RE.search(log_text)
    if not m:
        return None
    return {
        "nb_lambda_steps": int(m.group(1)),
        "lambda": float(m.group(2)),
        "nb_horizon_steps": int(m.group(3)),
    }


def summarize(
    epochs: List[Dict[str, float]],
    floor: float | None,
    lambda_info: Dict[str, float] | None,
) -> Dict[str, Any]:
    """Summarize a training run.

    Epochs with ω above ``MEANINGFUL_OMEGA_MIN`` are meaningful; the tail
    statistics of ω are over the last quarter of meaningful epochs.

    Args:
        epochs: Per-epoch losses and ω, from :func:`parse_log`.
        floor: Floor of the validation CoM loss, from :func:`parse_floor`.
        lambda_info: Parameters of the multi-step loss.

    Returns:
        Summary of the run.
    """
    if not epochs:
        return {
            "nb_epochs": 0,
            "val_com_floor": floor,
            "lambda_info": lambda_info,
        }
    meaningful = [e for e in epochs if e["omega"] > MEANINGFUL_OMEGA_MIN]
    summary: Dict[str, Any] = {
        "nb_epochs": len(epochs),
        "nb_meaningful_epochs": len(meaningful),
        "val_com_floor": floor,
        "lambda_info": lambda_info,
        "final_omega": epochs[-1]["omega"],
        "final_val": epochs[-1]["val"],
        "final_val_dcm": epochs[-1]["val_dcm"],
        "final_val_com": epochs[-1]["val_com"],
        "best_by_val": min(epochs, key=lambda e: e["val"]),
        "best_by_val_dcm": min(epochs, key=lambda e: e["val_dcm"]),
    }
    if meaningful:
        omegas = [e["omega"] for e in meaningful]
        tail_start = max(0, len(meaningful) - max(1, len(meaningful) // 4))
        tail_omegas = omegas[tail_start:]
        summary["best_by_val_meaningful"] = min(
            meaningful, key=lambda e: e["val"]
        )
        summary["tail_omega_mean"] = sum(tail_omegas) / len(tail_omegas)
        summary["tail_omega_min"] = min(tail_omegas)
        summary["tail_omega_max"] = max(tail_omegas)
    else:
        summary["best_by_val_meaningful"] = None
        summary["tail_omega_mean"] = None
        summary["tail_omega_min"] = None
        summary["tail_omega_max"] = None
    return summary


def launch(
    cfg: Dict[str, Any],
    data_abs: Path,
    logs_dir: Path,
    models_dir: Path,
    timeout_s: int,
    env: Dict[str, str],
    use_gpu: bool,
) -> Dict[str, Any]:
    """Train one run in a subprocess, and summarize it from its log.

    Args:
        cfg: Run configuration, from :func:`make_runs`.
        data_abs: Absolute path of the training dataset.
        logs_dir: Directory of the training logs.
        models_dir: Directory of the trained models.
        timeout_s: Timeout of the run, in seconds.
        env: Environment variables of the subprocess.
        use_gpu: Whether to require a CUDA GPU.

    Returns:
        Result of the run, with its status, summary and per-epoch values.
    """
    label = run_label(cfg)
    log_path = logs_dir / f"{label}.log"
    # Explicit output path, to keep the sweep's models in its own results
    # directory rather than the project-wide models/ default.
    # Absolute, since the subprocess runs from models_dir.
    onnx_path = (models_dir / f"dcm-{data_abs.stem}-{label}.onnx").resolve()
    cmd = [
        sys.executable,
        "-c",
        "from cookie_balancer.train_dcm import train; "
        f"train(dataset={str(data_abs)!r}, output={str(onnx_path)!r}, "
        f"seed={cfg['seed']}, epochs={cfg['epochs']}, "
        f"nb_lambda_steps={cfg['nb_lambda_steps']}, overwrite=True, "
        f"require_gpu={use_gpu})",
    ]
    print(f"\n--- Run {label} ---", flush=True)
    print(f"  cmd:   {shlex.join(cmd)}", flush=True)
    print(f"  cwd:   {models_dir}", flush=True)
    t0 = time.time()
    status = "ok"
    err: str | None = None
    with open(log_path, "wb") as f:
        try:
            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
                env=env,
                cwd=models_dir,
            )
            if proc.returncode != 0:
                status = f"exit_{proc.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except Exception as exc:  # noqa: BLE001
            status = "error"
            err = repr(exc)
    wall = time.time() - t0
    log_text = log_path.read_text(errors="replace")
    epochs_data = parse_log(log_text)
    floor = parse_floor(log_text)
    lambda_info = parse_lambda_info(log_text)
    summary = summarize(epochs_data, floor, lambda_info)
    result = {
        "label": label,
        "config": cfg,
        "status": status,
        "error": err,
        "wall_time_s": wall,
        "log_path": str(log_path),
        "onnx_path": str(onnx_path),
        "summary": summary,
        "epochs": epochs_data,
    }
    bm = summary.get("best_by_val_meaningful")
    tail = summary.get("tail_omega_mean")
    print(
        f"  result: status={status} wall={wall:.0f}s "
        f"epochs_parsed={summary.get('nb_epochs', 0)}",
        flush=True,
    )
    if bm:
        tail_str = f"tailω̄={tail:.3f}" if tail is not None else ""
        print(
            f"  best:   val={bm['val']:.3e} "
            f"(dcm={bm['val_dcm']:.3e} com={bm['val_com']:.3e}) "
            f"ω={bm['omega']:.3f} @ ep{bm['epoch']}  {tail_str}",
            flush=True,
        )
    return result


def write_results_atomic(results: List[Dict[str, Any]], path: Path) -> None:
    """Write results to a JSON file, replacing it atomically.

    Args:
        results: Results of the runs.
        path: Path of the JSON file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(path)


def load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the results of previous runs, to resume a sweep.

    Args:
        path: Path of the JSON results file.

    Returns:
        Results keyed by run label, empty if the file is absent or invalid.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return {r["label"]: r for r in data}
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: could not read {path}: {exc!r}. Treating as empty.",
            flush=True,
        )
        return {}


def is_completed(result: Dict[str, Any]) -> bool:
    """Check whether a run completed with at least one epoch.

    Args:
        result: Result of the run.

    Returns:
        True if the run completed.
    """
    return (
        result.get("status") == "ok"
        and result.get("summary", {}).get("nb_epochs", 0) > 0
    )


def convergence_metrics(epochs: List[Dict[str, float]]) -> Dict[str, Any]:
    """End-of-training metrics: what basin training actually converged to.

    Distinct from best-by-val (a snapshot from along the trajectory).
    Tail means are over the last 25% of all epochs, with no ω filter —
    ω-collapse must be visible.

    Args:
        epochs: Per-epoch losses and ω, from :func:`parse_log`.

    Returns:
        Final and tail values of ω and of the validation DCM loss.
    """
    if not epochs:
        return {
            "final_omega": None,
            "final_val_dcm": None,
            "tail_omega_raw": None,
            "tail_val_dcm_raw": None,
            "collapsed": None,
            "nb_epochs": 0,
        }
    tail_start = max(0, len(epochs) - max(1, len(epochs) // 4))
    tail = epochs[tail_start:]
    final_omega = epochs[-1]["omega"]
    return {
        "final_omega": final_omega,
        "final_val_dcm": epochs[-1]["val_dcm"],
        "tail_omega_raw": sum(e["omega"] for e in tail) / len(tail),
        "tail_val_dcm_raw": sum(e["val_dcm"] for e in tail) / len(tail),
        "collapsed": final_omega <= MEANINGFUL_OMEGA_MIN,
        "nb_epochs": len(epochs),
    }


def print_final_table(results: List[Dict[str, Any]]) -> None:
    """Print a summary table of the runs.

    Args:
        results: Results of the runs.
    """
    rows = []
    for r in results:
        cfg = r.get("config", {})
        cm = convergence_metrics(r.get("epochs", []) or [])
        mark = " (!)" if cm.get("collapsed") else ""
        rows.append(
            (
                cfg.get("nb_lambda_steps", -1),
                cfg.get("seed", -1),
                r["label"] + mark,
                r.get("status", "?"),
                cm.get("tail_val_dcm_raw"),
                cm.get("final_omega"),
                cm.get("tail_omega_raw"),
                cm.get("nb_epochs", 0),
                r.get("wall_time_s"),
            )
        )
    rows.sort(key=lambda row: (row[0], row[1]))
    header = (
        f"{'nls':>3}  {'seed':>4}  {'label':<32}  {'status':<10}  "
        f"{'tail val_dcm':>12}  {'final ω':>7}  {'tail ω̄':>7}  "
        f"{'eps':>3}  {'wall(s)':>8}"
    )
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    print(
        "Convergence metrics: tail = mean of last 25% of all epochs "
        "(no ω filter)."
    )
    print(
        "Rows marked (!) collapsed: final ω ≤ "
        f"{MEANINGFUL_OMEGA_MIN} (wrong basin)."
    )
    print("-" * len(header))

    def fnum(x: Any, fmt: str, w: int) -> str:
        """Format a number, or a dash if it is missing.

        Args:
            x: Number to format, or None.
            fmt: Format specification, without width.
            w: Width of the field.

        Returns:
            Formatted field.
        """
        if x is None:
            return f"{'-':>{w}}"
        try:
            return f"{x:>{w}{fmt}}"
        except (TypeError, ValueError):
            return f"{'-':>{w}}"

    for r in rows:
        print(
            f"{r[0]:>3d}  {r[1]:>4d}  {r[2]:<32}  {r[3]:<10}  "
            f"{fnum(r[4], '.3e', 12)}  {fnum(r[5], '.3f', 7)}  "
            f"{fnum(r[6], '.3f', 7)}  {r[7]:>3d}  "
            f"{fnum(r[8], '.0f', 8)}"
        )

    # Per-nls aggregate across seeds
    print()
    print("Per-nls aggregate (across completed seeds, end-of-training):")
    by_nls: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        if not is_completed(r):
            continue
        nls = r["config"]["nb_lambda_steps"]
        by_nls.setdefault(nls, []).append(r)
    print(
        f"  {'nls':>3}  {'n':>2}  {'collapsed':>9}  "
        f"{'ω̄ mean':>8}  {'ω̄ min':>7}  {'ω̄ max':>7}  "
        f"{'val_dcm mean':>13}  {'val_dcm min':>12}  {'val_dcm max':>12}"
    )
    for nls in sorted(by_nls):
        rs = by_nls[nls]
        cms = [convergence_metrics(r.get("epochs", []) or []) for r in rs]
        tails = [
            c["tail_omega_raw"] for c in cms if c["tail_omega_raw"] is not None
        ]
        vdcms = [
            c["tail_val_dcm_raw"]
            for c in cms
            if c["tail_val_dcm_raw"] is not None
        ]
        n_collapsed = sum(1 for c in cms if c.get("collapsed"))
        if not tails or not vdcms:
            continue
        print(
            f"  {nls:>3d}  {len(rs):>2d}  {n_collapsed:>4d}/{len(rs):<4d}  "
            f"{sum(tails) / len(tails):>8.3f}  {min(tails):>7.3f}  "
            f"{max(tails):>7.3f}  "
            f"{sum(vdcms) / len(vdcms):>13.3e}  {min(vdcms):>12.3e}  "
            f"{max(vdcms):>12.3e}"
        )


def self_test() -> None:
    """Check the log parsers on a sample log."""
    sample = (
        "[upkie] [12:45:26] [info] Multi-step loss: nb_lambda_steps=10 → "
        "λ=0.900000, "
        "nb_horizon_steps=50 (truncation tail λ^nb_horizon_steps=5.154e-03) "
        "(train_dcm.py:418)\n"
        "[upkie] [12:45:26] [info] val_com_floor (1-step trivial baseline) "
        "= mean((Δc_line)²) = 4.4130e-05 (train_dcm.py:535)\n"
        "[upkie] [12:45:30] [info] epoch   1/100  train=7.391e-03 "
        "(dcm=5.710e-03 com=1.681e-03) | val=5.044e-03 "
        "(dcm=4.221e-03 com=8.226e-04) | ω=4.8078 rad/s "
        "(train_dcm.py:615)\n"
        "[upkie] [12:45:32] [info] epoch   2/100  train=4.876e-03 "
        "(dcm=4.046e-03 com=8.309e-04) | val=4.527e-03 "
        "(dcm=3.586e-03 com=9.409e-04) | ω=4.6466 rad/s "
        "(train_dcm.py:615)"
    )
    epochs = parse_log(sample)
    assert len(epochs) == 2, epochs
    assert abs(epochs[0]["omega"] - 4.8078) < 1e-4
    floor = parse_floor(sample)
    assert floor is not None and abs(floor - 4.4130e-05) < 1e-9, floor
    td = parse_lambda_info(sample)
    assert td is not None and td["nb_lambda_steps"] == 10
    assert td["nb_horizon_steps"] == 50
    summary = summarize(epochs, floor, td)
    assert summary["nb_meaningful_epochs"] == 2
    assert summary["tail_omega_mean"] is not None
    print("Self-test passed.")


def main() -> None:
    """Run the sweep, skipping runs that already completed."""
    self_test()

    data_abs = DATASET.resolve()
    if not data_abs.exists():
        sys.exit(f"Dataset not found: {data_abs}")

    seeds = tuple(SEEDS)
    nls_values = tuple(sorted(set(NB_LAMBDA_STEPS)))
    runs = make_runs(seeds, nls_values, EPOCHS)

    logs_dir = RESULTS_DIR / "logs"
    models_dir = RESULTS_DIR / "models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "results.json"

    print(
        f"Planned {len(runs)} runs "
        f"(K={len(seeds)} seeds × {len(nls_values)} nls values: "
        f"{list(nls_values)}, {EPOCHS} epochs each)."
    )
    print(f"Dataset: {data_abs}")
    print(f"Summary → {summary_path}")
    print(f"Logs    → {logs_dir}")
    print(f"Models  → {models_dir}")

    existing = load_existing(summary_path)
    env = dict(os.environ)
    results: List[Dict[str, Any]] = []
    try:
        for i, cfg in enumerate(runs):
            label = run_label(cfg)
            print(f"\n[{i + 1}/{len(runs)}] {label}", flush=True)
            prev = existing.get(label)
            if prev and is_completed(prev):
                print("  already completed, skipping", flush=True)
                results.append(prev)
                continue
            result = launch(
                cfg,
                data_abs=data_abs,
                logs_dir=logs_dir,
                models_dir=models_dir,
                timeout_s=int(RUN_TIMEOUT_MIN * 60),
                env=env,
                use_gpu=USE_GPU,
            )
            results.append(result)
            write_results_atomic(results, summary_path)
    except KeyboardInterrupt:
        print(
            f"\nInterrupted. Partial results preserved at {summary_path}.",
            flush=True,
        )
        if results:
            write_results_atomic(results, summary_path)
        print_final_table(results)
        sys.exit(130)

    print_final_table(results)
    print(f"\nDone. Full results in {summary_path}.")


if __name__ == "__main__":
    main()

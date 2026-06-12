"""
Build a mobile-friendly HTML dashboard from DiffTrack experiment results.

Usage:
    python build_dashboard.py --results_dir ./results --out_dir ./mobile_results
"""

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from collections import defaultdict
from html import escape

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_log(log_path: Path) -> dict | None:
    """Return metrics from the LAST complete Mean block in a log file."""
    text = log_path.read_text()

    # A complete block ends with a Mean delta_16 line
    blocks = re.findall(
        r"((?:Video \d+.*?\n)+Mean delta_avg:\s*([\d.]+).*?Mean delta_16:\s*([\d.]+))",
        text,
        re.DOTALL,
    )
    if not blocks:
        return None

    last_block = blocks[-1][0]
    metrics = {}
    for key in ("delta_avg", "delta_1", "delta_2", "delta_4", "delta_8", "delta_16"):
        m = re.search(rf"Mean {key}:\s*([\d.]+)", last_block)
        if m:
            metrics[key] = float(m.group(1))

    video_lines = re.findall(r"Video (\d+) \((\d+) frames\)", last_block)
    metrics["n_videos"] = len(video_lines)
    metrics["frame_counts"] = [int(f) for _, f in video_lines]
    return metrics


def parse_dir_name(name: str) -> dict:
    """Extract layer, timestep, noise from dir name like layer[17]_timestep[49]_noiseFalse."""
    layer  = re.search(r"layer\[([^\]]+)\]", name)
    ts     = re.search(r"timestep\[([^\]]+)\]", name)
    noise  = re.search(r"noise(\w+)", name)
    return {
        "layer":     layer.group(1)  if layer  else "?",
        "timestep":  ts.group(1)     if ts     else "?",
        "noise":     noise.group(1)  if noise  else "?",
    }


def collect_experiments(results_dir: Path) -> list[dict]:
    """Walk results_dir and return a list of parsed experiment records."""
    records = []
    for log_path in sorted(results_dir.rglob("log.txt")):
        metrics = parse_log(log_path)
        if not metrics:
            continue
        exp_dir   = log_path.parent
        group_dir = exp_dir.parent
        cfg       = parse_dir_name(exp_dir.name)
        records.append({
            "group":    group_dir.name,
            "cfg_name": exp_dir.name,
            "path":     exp_dir,
            "log":      log_path,
            "metrics":  metrics,
            **cfg,
        })
    return records


# --------------------------------------------------------------------------- #
# Chart generation
# --------------------------------------------------------------------------- #

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]


def _bar_chart(labels, values, title, ylabel, out_path, baseline_label=None):
    if plt is None:
        return False

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 4))
    bars = ax.bar(labels, values, color=COLORS[:len(labels)], edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    if baseline_label and baseline_label in labels:
        idx = labels.index(baseline_label)
        bars[idx].set_edgecolor("gold")
        bars[idx].set_linewidth(2.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def make_layer_chart(records, out_dir: Path) -> Path | None:
    rows = [r for r in records if r["group"] == "param_study" and r["timestep"] == "49"
            and r["metrics"]["n_videos"] >= 2]
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: int(r["layer"]))
    labels = [f"Layer {r['layer']}" for r in rows]
    values = [r["metrics"]["delta_avg"] for r in rows]
    out = out_dir / "layer_comparison.png"
    if _bar_chart(labels, values, "Layer Ablation (timestep=49)", "Mean δ_avg", out, baseline_label="Layer 17"):
        return out
    return None


def make_timestep_chart(records, out_dir: Path) -> Path | None:
    rows = [r for r in records if r["layer"] == "17"
            and r["metrics"]["n_videos"] >= 2]
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: int(r["timestep"]))
    labels = [f"t={r['timestep']}" for r in rows]
    values = [r["metrics"]["delta_avg"] for r in rows]
    out = out_dir / "timestep_comparison.png"
    if _bar_chart(labels, values, "Timestep Ablation (layer=17)\nt=1 clean → t=49 noisy",
                  "Mean δ_avg", out, baseline_label="t=1"):
        return out
    return None


def make_delta_breakdown_chart(records, out_dir: Path) -> Path | None:
    """Multi-threshold breakdown for the best run (layer=17 with most videos)."""
    candidates = [r for r in records if r["layer"] == "17" and r["metrics"]["n_videos"] >= 2]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r["metrics"]["delta_avg"])
    keys   = ["delta_1", "delta_2", "delta_4", "delta_8", "delta_16"]
    labels = ["δ₁", "δ₂", "δ₄", "δ₈", "δ₁₆"]
    values = [best["metrics"].get(k, 0) for k in keys]
    out = out_dir / "delta_breakdown.png"
    if _bar_chart(labels, values,
                  f"δ Threshold Breakdown\n(layer={best['layer']}, t={best['timestep']})",
                  "% correct (↑)", out):
        return out
    return None


def make_limitations_chart(records, out_dir: Path) -> Path | None:
    rows = [r for r in records if r["group"] == "limitations" and r["metrics"]["n_videos"] >= 2]
    if not rows:
        return None
    labels = [f"L{r['layer']} t={r['timestep']}" for r in rows]
    values = [r["metrics"]["delta_avg"] for r in rows]
    out = out_dir / "limitations.png"
    if _bar_chart(labels, values, "Limitations Comparison", "Mean δ_avg", out):
        return out
    return None


# --------------------------------------------------------------------------- #
# Video collection
# --------------------------------------------------------------------------- #

def collect_videos(results_dir: Path, out_dir: Path) -> list[dict]:
    """Copy all .mp4s to out_dir/videos/ with flat names; return metadata list."""
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for mp4 in sorted(results_dir.rglob("*.mp4")):
        rel  = mp4.relative_to(results_dir)
        flat = str(rel).replace("/", "__").replace("[", "").replace("]", "")
        dst  = vid_dir / flat
        shutil.copy2(mp4, dst)

        parts  = rel.parts
        group  = parts[0] if len(parts) > 1 else "misc"
        cfg    = parts[1] if len(parts) > 2 else ""
        kind   = parts[2] if len(parts) > 3 else ""   # gt / pred
        vid_id = parts[3] if len(parts) > 4 else mp4.stem
        entries.append({
            "src":    f"videos/{flat}",
            "source": rel.as_posix(),
            "group":  group,
            "cfg":    cfg,
            "kind":   kind,
            "vid_id": vid_id,
            "label":  f"{group} / {cfg} / {kind} / {vid_id}",
        })
    return entries


# --------------------------------------------------------------------------- #
# Lineage
# --------------------------------------------------------------------------- #

def _rel_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build_lineage(records, video_entries, chart_paths, results_dir: Path, out_dir: Path,
                  generated_at: str) -> dict:
    """Return a machine-readable map from dashboard rows back to source artifacts."""
    videos_by_run: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for video in video_entries:
        videos_by_run[(video["group"], video["cfg"])].append({
            "source": video["source"],
            "dashboard_copy": video["src"],
            "kind": video["kind"],
            "video_id": video["vid_id"],
        })

    experiment_records = []
    for record in sorted(records, key=lambda r: (r["group"], r["cfg_name"])):
        key = (record["group"], record["cfg_name"])
        metrics = record["metrics"]
        experiment_records.append({
            "run_id": f"{record['group']}::{record['cfg_name']}",
            "group": record["group"],
            "config_name": record["cfg_name"],
            "parameters": {
                "layer": record["layer"],
                "timestep": record["timestep"],
                "noise": record["noise"],
            },
            "inputs": {
                "result_dir": _rel_path(record["path"], results_dir.parent),
                "log": _rel_path(record["log"], results_dir.parent),
                "videos": [video["source"] for video in videos_by_run.get(key, [])],
            },
            "metrics": metrics,
            "outputs": {
                "dashboard": _rel_path(out_dir / "index.html", out_dir),
                "copied_videos": [video["dashboard_copy"] for video in videos_by_run.get(key, [])],
            },
        })

    return {
        "generated_at": generated_at,
        "generated_by": "build_dashboard.py",
        "results_dir": results_dir.as_posix(),
        "dashboard_dir": out_dir.as_posix(),
        "global_outputs": {
            "dashboard": "index.html",
            "lineage_json": "lineage.json",
            "lineage_csv": "lineage.csv",
            "charts": [f"plots/{p.name}" for p in chart_paths if p is not None],
        },
        "experiments": experiment_records,
    }


def write_lineage_files(lineage: dict, out_dir: Path) -> tuple[Path, Path]:
    json_path = out_dir / "lineage.json"
    csv_path = out_dir / "lineage.csv"

    json_path.write_text(json.dumps(lineage, indent=2), encoding="utf-8")

    fields = [
        "run_id", "group", "config_name", "layer", "timestep", "noise",
        "n_videos", "delta_avg", "source_log", "source_result_dir",
        "source_video_count", "copied_video_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment in lineage["experiments"]:
            metrics = experiment["metrics"]
            writer.writerow({
                "run_id": experiment["run_id"],
                "group": experiment["group"],
                "config_name": experiment["config_name"],
                "layer": experiment["parameters"]["layer"],
                "timestep": experiment["parameters"]["timestep"],
                "noise": experiment["parameters"]["noise"],
                "n_videos": metrics.get("n_videos", ""),
                "delta_avg": metrics.get("delta_avg", ""),
                "source_log": experiment["inputs"]["log"],
                "source_result_dir": experiment["inputs"]["result_dir"],
                "source_video_count": len(experiment["inputs"]["videos"]),
                "copied_video_count": len(experiment["outputs"]["copied_videos"]),
            })

    return json_path, csv_path


# --------------------------------------------------------------------------- #
# HTML generation
# --------------------------------------------------------------------------- #

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #111; color: #e8e8e8; padding: 12px; }
h1 { font-size: 1.4rem; margin-bottom: 4px; color: #fff; }
h2 { font-size: 1.1rem; margin: 20px 0 8px; color: #adf; border-bottom: 1px solid #333; padding-bottom: 4px; }
h3 { font-size: 0.9rem; color: #ccc; margin: 12px 0 4px; }
.subtitle { font-size: 0.8rem; color: #888; margin-bottom: 16px; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-bottom: 12px; }
th { background: #222; color: #adf; text-align: left; padding: 6px 8px; }
td { padding: 5px 8px; border-bottom: 1px solid #222; }
tr:nth-child(even) td { background: #161616; }
.best td { color: #7ef07e; font-weight: bold; }
code { color: #cce7ff; font-size: 0.74rem; word-break: break-all; }
.lineage-table td { vertical-align: top; }
.muted { color: #888; }
.plots { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 8px; }
.plots img { max-width: 100%; border-radius: 6px; background: #1a1a1a; flex: 1 1 280px; }
.video-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }
.video-card { background: #1a1a1a; border-radius: 8px; padding: 8px; }
.video-card p { font-size: 0.72rem; color: #999; margin-top: 4px; word-break: break-all; }
video { width: 100%; border-radius: 4px; display: block; }
.tag { display: inline-block; background: #2a3a50; color: #7bc; border-radius: 4px;
       font-size: 0.7rem; padding: 1px 6px; margin-right: 4px; }
.tag.pred { background: #2a4030; color: #7ec; }
.tag.gt { background: #3a2a20; color: #e97; }
footer { margin-top: 24px; font-size: 0.72rem; color: #555; text-align: center; }
"""


def _table_row(r: dict) -> str:
    m = r["metrics"]
    return (
        f"<tr>"
        f"<td>{r['group']}</td>"
        f"<td>L{r['layer']}</td>"
        f"<td>t={r['timestep']}</td>"
        f"<td>{m['n_videos']}</td>"
        f"<td><b>{m['delta_avg']:.1f}</b></td>"
        f"<td>{m.get('delta_1', 0):.1f}</td>"
        f"<td>{m.get('delta_2', 0):.1f}</td>"
        f"<td>{m.get('delta_4', 0):.1f}</td>"
        f"<td>{m.get('delta_8', 0):.1f}</td>"
        f"<td>{m.get('delta_16', 0):.1f}</td>"
        f"</tr>\n"
    )


def _video_card(v: dict) -> str:
    kind_tag = f'<span class="tag {v["kind"]}">{v["kind"] or "video"}</span>'
    return (
        f'<div class="video-card">'
        f'<video controls playsinline preload="metadata" src="{v["src"]}"></video>'
        f'<p>{kind_tag} {v["cfg"]} — {v["vid_id"]}</p>'
        f'</div>\n'
    )


def _plot_section(chart_paths: list[Path | None], out_dir: Path) -> str:
    imgs = [p for p in chart_paths if p is not None]
    if not imgs:
        return "<p style='color:#666'>No charts generated yet — run more experiments.</p>"
    return '<div class="plots">' + "".join(
        f'<img src="plots/{p.name}" alt="{p.stem}">' for p in imgs
    ) + "</div>"


def _lineage_table(lineage: dict) -> str:
    rows = []
    for experiment in lineage["experiments"]:
        metrics = experiment["metrics"]
        inputs = experiment["inputs"]
        outputs = experiment["outputs"]
        rows.append(
            "<tr>"
            f"<td>{escape(experiment['group'])}<br><code>{escape(experiment['config_name'])}</code></td>"
            f"<td><code>{escape(inputs['log'])}</code><br>"
            f"<span class=\"muted\">{len(inputs['videos'])} source video(s)</span></td>"
            f"<td>L{escape(str(experiment['parameters']['layer']))}, "
            f"t={escape(str(experiment['parameters']['timestep']))}, "
            f"noise={escape(str(experiment['parameters']['noise']))}</td>"
            f"<td>{metrics.get('n_videos', 0)} video(s)<br>"
            f"delta_avg={metrics.get('delta_avg', 0):.1f}</td>"
            f"<td>{len(outputs['copied_videos'])} dashboard video(s)</td>"
            "</tr>\n"
        )

    return f"""
<table class="lineage-table">
  <thead><tr>
    <th>Run</th><th>Source</th><th>Parameters</th><th>Parsed Metrics</th><th>Dashboard Outputs</th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def build_html(records, video_entries, chart_paths, out_dir: Path, gen_date: str, lineage: dict) -> str:
    # Summary table — sort by delta_avg desc
    sorted_records = sorted(records, key=lambda r: r["metrics"]["delta_avg"], reverse=True)
    best_cfg = sorted_records[0]["cfg_name"] if sorted_records else ""
    table_rows = "".join(
        f'<tr class="best">' + _table_row(r)[4:] if r["cfg_name"] == best_cfg
        else _table_row(r)
        for r in sorted_records
    )
    table_html = f"""
<table>
  <thead><tr>
    <th>Group</th><th>Layer</th><th>Timestep</th><th>Videos</th>
    <th>δ_avg</th><th>δ₁</th><th>δ₂</th><th>δ₄</th><th>δ₈</th><th>δ₁₆</th>
  </tr></thead>
  <tbody>{table_rows}</tbody>
</table>"""

    # Videos grouped by group+cfg
    video_by_group: dict[str, list] = defaultdict(list)
    for v in video_entries:
        video_by_group[v["group"]].append(v)

    video_sections = ""
    for group, vids in sorted(video_by_group.items()):
        video_sections += f'<h3>{group}</h3><div class="video-grid">'
        video_sections += "".join(_video_card(v) for v in vids)
        video_sections += "</div>\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DiffTrack Results</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>DiffTrack — Experiment Dashboard</h1>
  <p class="subtitle">Generated: {gen_date} &nbsp;|&nbsp;
    Serve locally: <code>cd mobile_results && python -m http.server 8080</code></p>

  <h2>All Results (sorted by δ_avg)</h2>
  {table_html}

  <h2>Charts</h2>
  {_plot_section(chart_paths, out_dir)}

  <h2>Lineage</h2>
  <p class="subtitle">Machine-readable files: <code>lineage.json</code> and <code>lineage.csv</code></p>
  {_lineage_table(lineage)}

  <h2>Tracking Videos</h2>
  {video_sections if video_sections else "<p style='color:#666'>No videos found.</p>"}

  <footer>DiffTrack · CogVideoX-2B · DAVIS eval · t=1 clean, t=49 noisy</footer>
</body>
</html>"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--out_dir",     default="./mobile_results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    plots_dir   = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {results_dir} ...")
    records = collect_experiments(results_dir)
    print(f"  Found {len(records)} completed experiment(s)")
    for r in records:
        print(f"  [{r['group']}] layer={r['layer']} ts={r['timestep']} "
              f"-> delta_avg={r['metrics']['delta_avg']:.1f} ({r['metrics']['n_videos']} videos)")

    print("Generating charts ...")
    chart_paths = [
        make_layer_chart(records,      plots_dir),
        make_timestep_chart(records,   plots_dir),
        make_delta_breakdown_chart(records, plots_dir),
        make_limitations_chart(records, plots_dir),
    ]
    generated = [p for p in chart_paths if p]
    print(f"  {len(generated)} chart(s) saved to {plots_dir}")

    print("Collecting videos ...")
    video_entries = collect_videos(results_dir, out_dir)
    print(f"  {len(video_entries)} video(s) copied")

    from datetime import datetime
    gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    lineage = build_lineage(records, video_entries, chart_paths, results_dir, out_dir, gen_date)
    lineage_json, lineage_csv = write_lineage_files(lineage, out_dir)
    print(f"  Lineage saved to {lineage_json} and {lineage_csv}")

    html = build_html(records, video_entries, chart_paths, out_dir, gen_date, lineage)
    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard ready -> {index_path}")
    print(f"Serve with:  cd {out_dir} && python -m http.server 8080")


if __name__ == "__main__":
    main()

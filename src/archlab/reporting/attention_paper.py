"""Build the audited attention manuscript and its figures from compact evidence.

Run with the project's evaluation dependencies (matplotlib, numpy, Jinja2).
This is a deterministic report build: it does not evaluate models or query jobs.
Raw-run receipts in the compact inputs document the historical extraction; the
build needs only the checked-in inputs and manuscript template.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import re
import shutil
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape  # noqa: E402
from markupsafe import Markup  # noqa: E402
from matplotlib.mathtext import math_to_image  # noqa: E402

NORMAL = "#28618b"
SIMPLEX = "#ba632e"
GRAY = "#68717b"
INK = "#242b34"
INPUTS = {
    "learning": "docs/site/zh/assets/learning-curves.json",
    "warmup": "docs/site/zh/assets/warmup.json",
    "endpoints": "docs/recorded-results/limite-comparison-20261007.json",
    "aligned": "docs/paper/data/aligned-scratch.json",
    "controls": "docs/paper/data/crossstudy.json",
    "methods-audit": "docs/paper/data/methods-audit.json",
}


def _receipt(path: Path, label: str) -> dict:
    payload = path.read_bytes()
    return {"path": label, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _csv(path: Path, columns: list[str], rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def render_math(tex: str) -> Markup:
    """Typeset TeX math into self-contained vector paths, including every glyph.

    Mathtext supplies the TeX math layout; it does not require a system TeX
    installation or browser font downloads. Keep the source in an accessible
    label and prefix SVG identifiers so repeated glyphs cannot collide.
    """
    tex = tex.strip()
    target = io.BytesIO()
    with matplotlib.rc_context({"mathtext.fontset": "stix", "svg.fonttype": "path"}):
        math_to_image(
            f"${tex}$",
            target,
            format="svg",
            dpi=144,
            color=INK,
            prop=matplotlib.font_manager.FontProperties(size=17),
        )
    svg = target.getvalue().decode("utf-8")
    svg = svg[svg.index("<svg") :]
    svg = re.sub(r"<dc:date>.*?</dc:date>", "", svg)
    prefix = "math-" + hashlib.sha256(tex.encode()).hexdigest()[:12]
    identifiers = re.findall(r'\bid="([^"]+)"', svg)
    for identifier in sorted(identifiers, key=len, reverse=True):
        svg = svg.replace(f'id="{identifier}"', f'id="{prefix}-{identifier}"')
        svg = svg.replace(f'="#{identifier}"', f'="#{prefix}-{identifier}"')
        svg = svg.replace(f"url(#{identifier})", f"url(#{prefix}-{identifier})")
    svg = svg.replace("<svg ", '<svg role="img" ', 1)
    svg = svg.replace(">", f"><title>{escape(tex)}</title>", 1)
    return Markup(svg)


def summarize(data: dict) -> dict:
    """Recompute manuscript quantities; reject mismatched or incomplete inputs."""
    learning = data["learning"]
    points = learning["series"]["points"]
    assert learning["series"]["columns"] == ["tokens", "normal", "simplicial"]
    assert len(points) == 66
    assert all(a[0] < b[0] for a, b in zip(points, points[1:], strict=False))
    assert points[-1][0] == 10_000_000_000
    saved = learning["series"]["all_saved_points"]
    assert all(len(saved[arm]) == 70 for arm in ("normal", "simplicial"))
    for arm, index in (("normal", 1), ("simplicial", 2)):
        lookup = dict(saved[arm])
        assert all(lookup[p[0]] == p[index] for p in points)
    endpoint = points[-1]
    thresholds = []
    for ce in (0.82, 0.81, endpoint[1]):
        first = {
            arm: next(tokens for tokens, loss in saved[arm] if loss <= ce)
            for arm in ("normal", "simplicial")
        }
        thresholds.append(
            {
                "ce": ce,
                "normal_tokens": first["normal"],
                "simplicial_tokens": first["simplicial"],
                "fewer_tokens_percent": 100 * (1 - first["simplicial"] / first["normal"]),
                "token_ratio": first["normal"] / first["simplicial"],
            }
        )
    assert thresholds[0]["normal_tokens"] == 5_242_880_000
    assert thresholds[0]["simplicial_tokens"] == 3_801_088_000
    assert thresholds[1]["normal_tokens"] == 9_043_968_000
    assert thresholds[1]["simplicial_tokens"] == 5_767_168_000
    assert thresholds[2]["normal_tokens"] == 9_568_256_000
    assert thresholds[2]["simplicial_tokens"] == 5_898_240_000

    aligned = []
    for cohort in data["aligned"]["cohorts"]:
        rows = cohort["points"]
        assert rows[0][1] == 0 and rows[0][2] == rows[0][3]
        assert all(a[1] < b[1] for a, b in zip(rows, rows[1:], strict=False))
        assert not cohort["completed_planned_budget"]
        final = rows[-1]
        aligned.append(
            {
                "width": cohort["width"],
                "matched_tokens": final[1],
                "matched_step": final[0],
                "final_normal": final[2],
                "final_simplicial": final[3],
                "delta": final[3] - final[2],
                "relative_reduction_percent": 100 * (final[2] - final[3]) / final[2],
                "nonzero_paired_points": len(rows) - 1,
                "simplicial_lower_points": sum(r[3] < r[2] for r in rows[1:]),
                "planned_budget_fraction": final[1] / cohort["planned_tokens"],
            }
        )
    for result in data["warmup"]["results"].values():
        assert result["responses"] == 120
        assert np.isclose(result["mean_pass_at_1"], result["correct_responses"] / 120)
        low, high = result["pass_at_1_cluster_bootstrap_95pct"]
        assert low <= result["mean_pass_at_1"] <= high
    assert data["endpoints"]["protocol"]["problems"] == 30
    assert data["endpoints"]["protocol"]["draws_per_problem"] == 4
    for result in data["endpoints"]["models"]:
        assert result["responses"] == 120
        assert 0 <= result["correct"] <= 120
        assert 0 <= result["native_context_exhausted"] <= 120

    return {
        "learning": {
            "paired_points": len(points),
            "simplicial_lower_points": sum(p[2] < p[1] for p in points),
            "final_normal": endpoint[1],
            "final_simplicial": endpoint[2],
            "final_delta": endpoint[2] - endpoint[1],
            "relative_reduction_percent": 100 * (endpoint[1] - endpoint[2]) / endpoint[1],
            "parameter_difference_percent": learning["comparison"]["parameter_difference_percent"],
        },
        "thresholds": thresholds,
        "aligned": aligned,
        "controls": [
            {
                **row,
                "delta": row["simplicial"] - row["normal"],
                "relative_delta_percent": 100 * (row["simplicial"] - row["normal"]) / row["normal"],
            }
            for row in data["controls"]["deepseek"]
        ],
        "warmup": data["warmup"]["results"],
        "endpoints": data["endpoints"]["models"],
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": "#9ca5af",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e5e8eb",
            "grid.linewidth": 0.6,
            "grid.alpha": 1,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "archlab-attention-paper-20261013",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, out: Path, name: str, description: str) -> Markup:
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        metadata = {"Date": None} if suffix == "svg" else None
        if suffix == "pdf":
            metadata = {"CreationDate": None, "ModDate": None}
        fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight", metadata=metadata, **kwargs)
    plt.close(fig)
    svg = (out / f"{name}.svg").read_text()
    svg = svg[svg.index("<svg") :]
    # Matplotlib reuses IDs such as figure_1; prefix every definition/reference.
    identifiers = re.findall(r'\bid="([^"]+)"', svg)
    for identifier in sorted(identifiers, key=len, reverse=True):
        svg = svg.replace(f'id="{identifier}"', f'id="{name}-{identifier}"')
        svg = svg.replace(f'="#{identifier}"', f'="#{name}-{identifier}"')
        svg = svg.replace(f"url(#{identifier})", f"url(#{name}-{identifier})")
    svg = svg.replace("<svg ", '<svg role="img" ', 1)
    svg = svg.replace(">", f"><title>{description}</title>", 1)
    return Markup(svg)


def plot_learning(data: dict, stats: dict, out: Path) -> Markup:
    rows = np.asarray(data["learning"]["series"]["points"])
    x = rows[:, 0] / 1e9
    fig, (ax, delta) = plt.subplots(
        2,
        1,
        figsize=(9.8, 6.3),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1], "hspace": 0.18},
    )
    ax.plot(x, rows[:, 1], color=NORMAL, lw=2, label="Ordinary attention")
    ax.plot(x, rows[:, 2], color=SIMPLEX, lw=2, label="2-simplicial attention")
    ax.scatter(x[-1:], rows[-1:, 1], color=NORMAL, s=30, zorder=4)
    ax.scatter(x[-1:], rows[-1:, 2], color=SIMPLEX, s=30, zorder=4)
    ax.annotate(
        f"{rows[-1, 1]:.6f}",
        (x[-1], rows[-1, 1]),
        xytext=(-5, 8),
        textcoords="offset points",
        ha="right",
        color=NORMAL,
    )
    ax.annotate(
        f"{rows[-1, 2]:.6f}",
        (x[-1], rows[-1, 2]),
        xytext=(-5, -16),
        textcoords="offset points",
        ha="right",
        color=SIMPLEX,
    )
    ax.set_ylabel("Validation CE (nats / target)")
    ax.set_title("a  Limite continuation: the same 66 saved training positions", loc="left", pad=12)
    ax.legend(loc="upper right")
    ax.set_ylim(0.795, 0.866)
    difference = rows[:, 2] - rows[:, 1]
    delta.axhline(0, color=GRAY, lw=0.8)
    delta.plot(x, difference, color=SIMPLEX, lw=1.8)
    delta.fill_between(x, difference, 0, color=SIMPLEX, alpha=0.09)
    delta.set_ylabel("CE difference\n(simplicial − ordinary)")
    delta.set_xlabel("Total supervised training targets (billions; includes adapter warmup)")
    delta.set_title("b  Paired validation difference", loc="left", fontsize=11, pad=8)
    delta.set_xlim(1.9, 10.15)
    delta.set_ylim(-0.017, 0.0015)
    delta.text(
        0.98,
        0.10,
        "Lower CE at 66 / 66 shared positions",
        transform=delta.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=GRAY,
    )
    _csv(
        out / "learning.csv",
        ["training_tokens", "normal_ce", "simplicial_ce", "simplicial_minus_normal_ce"],
        [[int(t), n, s, s - n] for t, n, s in rows],
    )
    return _save(
        fig,
        out,
        "learning",
        "Limite validation learning curves and their paired cross-entropy differences.",
    )


def plot_thresholds(stats: dict, out: Path) -> Markup:
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    labels = ["CE ≤ 0.820", "CE ≤ 0.810", "CE ≤ 0.809566\n(ordinary endpoint CE)"]
    for i, row in enumerate(stats["thresholds"]):
        y = 2 - i
        n, s = row["normal_tokens"] / 1e9, row["simplicial_tokens"] / 1e9
        ax.plot([s, n], [y, y], color="#b7c0c8", lw=2, zorder=1)
        ax.scatter(
            [n], [y], s=70, color=NORMAL, zorder=3, label="Ordinary attention" if i == 0 else None
        )
        ax.scatter(
            [s],
            [y],
            s=70,
            color=SIMPLEX,
            zorder=3,
            label="2-simplicial attention" if i == 0 else None,
        )
        for value, color in ((n, NORMAL), (s, SIMPLEX)):
            ax.annotate(
                f"{value:.3f}B",
                (value, y),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=10,
            )
        ax.text(10.4, y, f"{row['fewer_tokens_percent']:.1f}% fewer", va="center", fontsize=10)
    ax.set_yticks([2, 1, 0], labels)
    ax.set_xlim(3.0, 12.2)
    ax.set_ylim(-0.5, 2.65)
    ax.set_xticks([4, 6, 8, 10])
    ax.set_xlabel(
        "First observed saved position at or below the CE target (billion training targets)"
    )
    ax.set_title("Both arms use the first observed crossing; no interpolation", loc="left", pad=14)
    ax.legend(loc="lower right", ncols=2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    _csv(
        out / "thresholds.csv",
        list(stats["thresholds"][0]),
        [list(r.values()) for r in stats["thresholds"]],
    )
    return _save(
        fig,
        out,
        "thresholds",
        "Observed training targets needed for both arms to reach three fixed validation cross-entropy thresholds.",
    )


def plot_aime(data: dict, out: Path) -> Markup:
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.5), gridspec_kw={"wspace": 0.5})
    ax = axes[0]
    warmup = data["warmup"]["results"]
    warmup_rows = []
    for i, (arm, color) in enumerate((("base", GRAY), ("normal", NORMAL), ("simplicial", SIMPLEX))):
        row = warmup[arm]
        value = row["mean_pass_at_1"] * 100
        low, high = [v * 100 for v in row["pass_at_1_cluster_bootstrap_95pct"]]
        ax.errorbar(
            i,
            value,
            yerr=[[value - low], [high - value]],
            fmt="o",
            markersize=7,
            color=color,
            capsize=4,
            lw=1.5,
        )
        ax.annotate(
            f"{row['correct_responses']}/120",
            (i, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
        warmup_rows.append([arm, row["correct_responses"], 120, value, low, high])
    ax.set_xticks([0, 1, 2], ["Base", "Ordinary", "Simplicial"], rotation=20)
    ax.set_ylim(0, 40)
    ax.set_ylabel("Mean response correctness (%)")
    ax.set_title("a  Adapter warmup (~2B)", loc="left")
    ax.text(
        0.5,
        -0.30,
        "Marginal 95% problem-cluster intervals",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )
    endpoints = {r["model"]: r for r in data["endpoints"]["models"]}
    rows = []
    for col, (key, title) in enumerate(
        (("correct", "b  Later correctness"), ("native_context_exhausted", "c  Context exhausted")),
        start=1,
    ):
        ax = axes[col]
        for i, arm in enumerate(("normal", "simplicial")):
            color = NORMAL if arm == "normal" else SIMPLEX
            vals = [endpoints[f"{arm}_{stage}"][key] for stage in ("sft10b", "rl400")]
            bars = ax.bar(
                np.arange(2) + (i - 0.5) * 0.32,
                np.array(vals) / 120 * 100,
                width=0.30,
                color=color,
                label="Ordinary" if i == 0 else "2-simplicial",
            )
            for bar, count in zip(bars, vals, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.1,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color,
                )
        ax.set_xticks([0, 1], ["SFT 10B", "+ RL 400"])
        ax.set_ylabel("Responses (%)")
        ax.set_ylim(0, 32 if key == "correct" else 90)
        ax.set_title(title, loc="left")
        ax.text(
            0.5,
            -0.30,
            "Counts / 120; marginal intervals not shown",
            transform=ax.transAxes,
            ha="center",
            fontsize=8.5,
            color=GRAY,
        )
    axes[2].legend(loc="upper right", fontsize=8)
    fig.subplots_adjust(bottom=0.28, top=0.87)
    for arm in ("normal", "simplicial"):
        for stage in ("sft10b", "rl400"):
            r = endpoints[f"{arm}_{stage}"]
            rows.append([arm, stage, r["correct"], 120, r["native_context_exhausted"]])
    _csv(
        out / "aime-warmup.csv",
        ["arm", "correct", "responses", "correct_percent", "ci_low_percent", "ci_high_percent"],
        warmup_rows,
    )
    _csv(
        out / "aime-endpoints.csv",
        ["arm", "stage", "correct", "responses", "native_context_exhausted"],
        rows,
    )
    return _save(
        fig,
        out,
        "aime",
        "AIME 2026 warmup correctness with cluster intervals and later endpoint correctness and context exhaustion counts.",
    )


def plot_scratch(data: dict, stats: dict, out: Path) -> Markup:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 6.7),
        sharex="col",
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.20, "wspace": 0.33},
    )
    csv_rows = []
    for col, (cohort, summary) in enumerate(
        zip(data["aligned"]["cohorts"], stats["aligned"], strict=True)
    ):
        rows = np.asarray(cohort["points"])[1:]
        x = rows[:, 1] / 1e9
        ax, delta = axes[0, col], axes[1, col]
        ax.plot(x, rows[:, 2], color=NORMAL, lw=1.8, label="Ordinary attention")
        ax.plot(x, rows[:, 3], color=SIMPLEX, lw=1.8, label="2-simplicial attention")
        for index, color in ((2, NORMAL), (3, SIMPLEX)):
            ax.scatter(x[-1], rows[-1, index], color=color, s=25, zorder=3)
        ax.set_title(
            f"{'a' if col == 0 else 'b'}  Hidden width {cohort['width']}, depth 20", loc="left"
        )
        ax.set_ylabel("Validation CE (nats / target)")
        ax.text(
            0.97,
            0.97,
            f"Shared prefix: {x[-1]:.1f}B / 10B planned\nOne seed per arm; budget incomplete",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8.5,
            color=GRAY,
        )
        if col == 0:
            ax.legend(loc="center right", fontsize=9)
        difference = rows[:, 3] - rows[:, 2]
        delta.axhline(0, color=GRAY, lw=0.8)
        delta.plot(x, difference, color=SIMPLEX, lw=1.5)
        delta.fill_between(x, difference, 0, color=SIMPLEX, alpha=0.09)
        delta.set_ylabel("CE difference\n(simplicial − ordinary)")
        delta.set_xlabel("Matched training targets (billions)")
        delta.set_xlim(0, x[-1] + 0.06)
        delta.annotate(
            f"Final Δ {summary['delta']:+.4f}",
            (x[-1], difference[-1]),
            xytext=(-4, -15),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
        )
        for step, tokens, n, s in cohort["points"]:
            csv_rows.append([cohort["width"], step, tokens, n, s, s - n])
    _csv(
        out / "scratch.csv",
        [
            "hidden_width",
            "step",
            "training_tokens",
            "normal_ce",
            "simplicial_ce",
            "simplicial_minus_normal_ce",
        ],
        csv_rows,
    )
    return _save(
        fig,
        out,
        "scratch",
        "Aligned scratch training prefixes at hidden widths 128 and 640; ordinary and simplicial validation curves and paired differences.",
    )


def plot_controls(data: dict, stats: dict, out: Path) -> Markup:
    fig, axes = plt.subplots(
        1, 2, figsize=(10.7, 4.1), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.60}
    )
    ax = axes[0]
    qwen = data["controls"]["qwen"]
    rows = []
    for y, (key, _label, color) in enumerate(
        (
            ("global", "Global ordinary", GRAY),
            ("local", "Local ordinary", NORMAL),
            ("simplicial", "2-simplicial", SIMPLEX),
        )
    ):
        value = qwen["arms"][key]["ce"]
        ax.scatter(value, 2 - y, color=color, s=55)
        ax.annotate(
            f"{value:.6f}",
            (value, 2 - y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=color,
        )
        rows.append(["qwen_w320", key, value, qwen["training_tokens"], "", ""])
    ax.set_yticks([2, 1, 0], ["Global ordinary", "Local ordinary", "2-simplicial"])
    ax.set_xlim(3.633, 3.676)
    ax.set_ylim(-0.45, 2.65)
    ax.set_xlabel("Validation CE (nats / target)")
    ax.set_title("a  Qwen-style pilot at 3.003B", loc="left", pad=14)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    for y, row in enumerate(stats["controls"]):
        value = row["relative_delta_percent"]
        ax.scatter(value, 2 - y, color=SIMPLEX, s=55)
        interval = row["ci_delta"]
        if interval:
            lo, hi = [100 * v / row["normal"] for v in interval]
            ax.errorbar(
                value,
                2 - y,
                xerr=[[value - lo], [hi - value]],
                color=SIMPLEX,
                fmt="none",
                lw=1.5,
                capsize=4,
            )
        ax.annotate(
            f"{value:+.3f}%",
            (value, 2 - y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=SIMPLEX,
        )
        for arm in ("normal", "simplicial"):
            rows.append(
                [
                    row["id"],
                    arm,
                    row[arm],
                    row["training_tokens"],
                    interval[0] if interval else "",
                    interval[1] if interval else "",
                ]
            )
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(
        [2, 1, 0],
        [
            "Frozen adapters\n50.120M targets",
            "Full continuation\n756.365M targets",
            "Original scratch\n240.875M targets",
        ],
    )
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(-0.45, 2.65)
    ax.set_xlabel("Within-study CE difference / ordinary CE (%)")
    ax.set_title("b  Earlier DeepSeek-style controls", loc="left", pad=14)
    ax.grid(axis="y", visible=False)
    fig.text(
        0.52,
        0.01,
        "Positive differences favor ordinary attention. Interval only for full continuation; no pooled estimate.",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    fig.subplots_adjust(bottom=0.21)
    _csv(
        out / "controls.csv",
        ["study", "arm", "ce", "training_tokens", "paired_delta_ci_low", "paired_delta_ci_high"],
        rows,
    )
    return _save(
        fig,
        out,
        "controls",
        "Earlier within-study controls: Qwen pilot cross-entropies and relative DeepSeek cross-entropy differences; no pooled effect.",
    )


def export_pdf(output: Path, executable: Path | None = None) -> None:
    """Print the already-typeset HTML; browser/font paths belong in the environment."""
    from playwright.sync_api import sync_playwright

    errors, external_requests = [], []
    with sync_playwright() as playwright:
        options = {
            "headless": True,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
        }
        if executable is not None:
            options["executable_path"] = str(executable.resolve())
        browser = playwright.chromium.launch(**options)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                external_requests.append(request.url)
                if request.url.startswith(("https://", "http://"))
                else None
            ),
        )
        page.goto((output / "index.html").as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        assert page.locator("figure svg").count() == 5
        assert page.locator(".equation svg").count() == 5
        assert page.locator(".equation svg text").count() == 0  # Glyphs are embedded paths.
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.pdf(
            path=str(output / "paper.pdf"),
            print_background=True,
            prefer_css_page_size=True,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template=(
                '<div style="font-size:8px;width:100%;text-align:center;color:#65707a">'
                '<span class="pageNumber"></span> / <span class="totalPages"></span></div>'
            ),
        )
        assert not errors, errors
        assert not external_requests, external_requests
        _json(
            output / "data/pdf-verification.json",
            {
                "browser": browser.version,
                "html_sha256": _receipt(output / "index.html", "index.html")["sha256"],
                "pdf_sha256": _receipt(output / "paper.pdf", "paper.pdf")["sha256"],
                "math_renderer": "Matplotlib Mathtext, STIX, embedded SVG glyph paths",
                "display_equations": 3,
                "typeset_math_rows": 5,
                "experiment_figures": 5,
                "javascript_errors": errors,
                "external_render_requests": external_requests,
            },
        )
        browser.close()


def build(
    repo: Path,
    output: Path,
    template: Path,
    *,
    pdf: bool = False,
    browser_executable: Path | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    data_dir = output / "data"
    data_dir.mkdir(exist_ok=True)
    inputs = dict(INPUTS)
    data = {key: json.loads((repo / relative).read_text()) for key, relative in inputs.items()}
    stats = summarize(data)
    _style()
    figures = {
        "learning": plot_learning(data, stats, figure_dir),
        "thresholds": plot_thresholds(stats, figure_dir),
        "aime": plot_aime(data, figure_dir),
        "scratch": plot_scratch(data, stats, figure_dir),
        "controls": plot_controls(data, stats, figure_dir),
    }
    for key, relative in inputs.items():
        shutil.copyfile(repo / relative, data_dir / f"{key}.json")
    _json(output / "statistics.json", stats)
    if template.exists():
        env = Environment(
            loader=FileSystemLoader(str(template.parent)),
            undefined=StrictUndefined,
            autoescape=select_autoescape(["html", "j2"]),
        )
        env.filters["mathsvg"] = render_math
        rendered = env.get_template(template.name).render(data=data, stats=stats, figures=figures)
        (output / "index.html").write_text(rendered, encoding="utf-8")
    if pdf:
        export_pdf(output, browser_executable)
    verification = {
        "passed": True,
        "scope": "Compact input consistency and recomputed figure statistics; no model execution.",
        "checks": {
            "strictly_increasing_shared_limite_tokens": True,
            "paired_limite_points": 66,
            "all_saved_points_per_arm": 70,
            "shared_points_match_each_saved_series": True,
            "first_observed_threshold_crossings_match_recorded_values": True,
            "warmup_correctness_matches_counts": True,
            "warmup_estimates_within_supplied_marginal_intervals": True,
            "endpoint_count_denominators": 120,
            "aligned_shared_prefixes_increasing": True,
            "aligned_initial_cross_entropies_equal": True,
            "aligned_planned_budgets_incomplete": True,
        },
        "statistics": stats,
        "limitations": [
            "Raw receipts establish provenance; their payloads are not required or rehashed by this portable build.",
            "No missing seed-level or endpoint uncertainty is inferred.",
            "No cross-study effect is pooled.",
        ],
    }
    _json(data_dir / "verification.json", verification)
    (output / "provenance.json").unlink(missing_ok=True)
    manifest_path = data_dir / "manifest.json"
    artifact_paths = [
        output / "index.html",
        output / "statistics.json",
        output / "paper.pdf",
        output / "browser-verification.json",
        *(p for p in figure_dir.iterdir() if p.suffix in {".svg", ".pdf", ".png", ".csv"}),
        *data_dir.glob("*.json"),
    ]
    provenance = {
        "description": "Audited compact evidence; figures regenerated without model execution.",
        "evidence_extraction_date": "2026-10-13",
        "software": {
            package: importlib.metadata.version(package)
            for package in ("matplotlib", "numpy", "jinja2")
        },
        "inputs": [_receipt(repo / relative, relative) for relative in inputs.values()],
        "raw_run_receipts": (
            data["aligned"]["sources"]
            + data["controls"]["sources"]
            + data["methods-audit"]["sources"]
        ),
        "builder": _receipt(Path(__file__), "src/archlab/reporting/attention_paper.py"),
        "template": _receipt(template, str(template.relative_to(repo)))
        if template.exists()
        else None,
        "artifacts": [
            _receipt(p, str(p.relative_to(output)))
            for p in sorted(artifact_paths)
            if p.is_file() and p != manifest_path
        ],
        "validation": {
            "paired_limite_points": 66,
            "all_saved_points_per_arm": 70,
            "threshold_rule": "first observed saved point at or below target for each arm",
            "seed_confidence_intervals_inferred": False,
            "cross_study_effects_pooled": False,
            "new_model_evaluations": False,
        },
    }
    _json(manifest_path, provenance)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path("results/attention-paper-20261013"))
    parser.add_argument("--template", type=Path, default=Path("docs/paper/manuscript.html.j2"))
    parser.add_argument(
        "--pdf", action="store_true", help="Recompile the PDF with Playwright Chromium"
    )
    parser.add_argument(
        "--browser-executable", type=Path, help="Optional installed Chromium executable"
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    template = args.template.resolve()
    stats = build(repo, output, template, pdf=args.pdf, browser_executable=args.browser_executable)
    print(
        json.dumps(
            {
                "output": str(output),
                "figures": 5,
                "html_rendered": (output / "index.html").exists(),
                "paired_limite_points": stats["learning"]["paired_points"],
            }
        )
    )


if __name__ == "__main__":
    main()

"""Render one self-contained HTML per run from the StepRecord stream.

Reads ONLY trace_log JSONL. Degrades gracefully on fields that are not wired up
yet, so it is usable from the first instrumented run rather than after Phase 4.

Offline-only and non-negotiable: all CSS/JS inline, no CDN, no external font, no
network request of any kind. The hypothesis text carries real Slack messages
from named employees and copyrighted transcript material, so the file must be
safe to open offline and must never phone home.
"""
import argparse
import html
import json
import math
import os
from typing import Dict, List, Optional

import trace_log
from trace_log import StepRecord

W, PAD_L, PAD_R = 980, 64, 24
FLOOR_KEYS = ["reversals_median", "argmax_churn", "merge_count", "split_count", "anchor_revision_count"]


# ---------------------------------------------------------------- colour
def _hash(s: str) -> int:
    import hashlib
    return int(hashlib.md5((s or "").encode()).hexdigest()[:8], 16)


def build_palette(steps) -> Dict[str, str]:
    """One stable colour per LINEAGE, collision-free within a run.

    Keyed on lineage_id, not particle_id: particle_id is fresh every step, so
    hashing it gave ~131 keys in a 4-particle run, birthday-collided down to 37
    hues, and made "same colour = same particle" false.

    Hues are seeded from a stable hash so a lineage keeps its colour across
    runs, then nudged by the golden angle on collision so two lineages in the
    same run are never the same hue. Lightness shifts ONLY for genuine split
    children, whose parent hue they inherit.
    """
    lineages, is_split, parent_of = [], {}, {}
    for st in steps:
        for p in st.particles:
            lid = p.lineage_id or p.particle_id
            if lid not in is_split:
                lineages.append(lid)
                is_split[lid] = bool(getattr(p, "split_child", False))
                parent_of[lid] = p.parent_id
            elif getattr(p, "split_child", False):
                is_split[lid] = True

    used, hue_of = set(), {}
    for lid in lineages:
        h = _hash(lid) % 360
        guard = 0
        while any(abs(h - u) < 18 or abs(h - u) > 342 for u in used) and guard < 40:
            h = int(h + 137.508) % 360
            guard += 1
        used.add(h)
        hue_of[lid] = h

    palette = {}
    for lid in lineages:
        light = 44 if is_split[lid] else 52
        palette[lid] = f"hsl({hue_of[lid]},64%,{light}%)"
    return palette


def lid_of(p) -> str:
    return p.lineage_id or p.particle_id


# ---------------------------------------------------------------- geometry
def _scale(vals, lo=None, hi=None):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0, 1.0
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return lo, hi


def _x(i, n, w):
    return PAD_L + (w - PAD_L - PAD_R) * (i / max(1, n - 1))


def axis(n_steps, h, y0, label):
    ticks = []
    step = max(1, n_steps // 10)
    for i in range(0, n_steps, step):
        x = _x(i, n_steps, W)
        ticks.append(f'<line x1="{x:.1f}" y1="{y0+h}" x2="{x:.1f}" y2="{y0+h+4}" class="ax"/>'
                     f'<text x="{x:.1f}" y="{y0+h+15}" class="tick" text-anchor="middle">{i}</text>')
    return (f'<line x1="{PAD_L}" y1="{y0+h}" x2="{W-PAD_R}" y2="{y0+h}" class="ax"/>'
            + "".join(ticks)
            + f'<text x="{PAD_L}" y="{y0-6}" class="ptitle">{html.escape(label)}</text>')


# ---------------------------------------------------------------- panels
def panel_a(steps: List[StepRecord], palette, deadband) -> str:
    n = len(steps)
    series: Dict[str, List[Optional[float]]] = {}
    wser: Dict[str, List[Optional[float]]] = {}
    dup: Dict[str, set] = {}
    for i, s in enumerate(steps):
        for p in s.particles:
            k = lid_of(p)
            series.setdefault(k, [None] * n)[i] = p.raw_accumulator
            wser.setdefault(k, [None] * n)[i] = p.weight
            if getattr(p, "resample_duplicate", False):
                dup.setdefault(k, set()).add(i)

    h1, y1 = 190, 28
    flat = [v for vs in series.values() for v in vs if v is not None]
    if len(set(v for v in flat)) <= 1:
        a1 = ('<text x="%d" y="%d" class="nodata">accumulator is constant &mdash; '
              'Phase 2 has not wired accumulation, so no reversal can exist yet</text>' % (PAD_L, y1 + 60))
    elif not flat:
        a1 = ('<text x="%d" y="%d" class="nodata">no raw_accumulator yet &mdash; '
              'reversals cannot be counted until Phase 2 wires accumulation</text>' % (PAD_L, y1 + 60))
    else:
        lo, hi = _scale(flat)
        parts = []
        for pid, vs in series.items():
            pts, dots = [], []
            prev_dir, prev_val = 0, None
            for i, v in enumerate(vs):
                if v is None:
                    continue
                x = _x(i, n, W)
                y = y1 + h1 - (v - lo) / (hi - lo) * h1
                pts.append(f"{x:.1f},{y:.1f}")
                if prev_val is not None:
                    d = v - prev_val
                    if abs(d) >= deadband:
                        nd = 1 if d > 0 else -1
                        if prev_dir and nd != prev_dir:
                            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" class="rev"/>')
                        prev_dir = nd
                prev_val = v
            if len(pts) > 1:
                c = palette.get(pid, "hsl(0,0%,50%)")
                parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="1.9"/>')
                parts.append("".join(dots))
        a1 = "".join(parts)

    h2, y2 = 150, y1 + h1 + 52
    bands = []
    order = list(wser)
    for i, s in enumerate(steps):
        pass
    cum = [0.0] * n
    for pid in order:
        vs = wser[pid]
        top, bot = [], []
        for i in range(n):
            v = vs[i] or 0.0
            x = _x(i, n, W)
            y_lo = y2 + h2 - cum[i] * h2
            y_hi = y_lo - v * h2
            bot.append(f"{x:.1f},{y_lo:.1f}")
            top.append(f"{x:.1f},{y_hi:.1f}")
            cum[i] += v
        pts = " ".join(top + list(reversed(bot)))
        bands.append(f'<polygon points="{pts}" fill="{palette.get(pid, "hsl(0,0%,50%)")}" opacity="0.85">'
                     f'<title>{html.escape(pid)}</title></polygon>')
        if pid in dup:
            for i in sorted(dup[pid]):
                x = _x(i, n, W)
                bands.append(f'<rect x="{x-3:.1f}" y="{y2}" width="6" height="{h2}" '
                             f'fill="url(#hatch)" opacity="0.9"><title>resample duplicate</title></rect>')

    defs = ('<defs><pattern id="hatch" width="5" height="5" patternTransform="rotate(45)" '
            'patternUnits="userSpaceOnUse"><line x1="0" y1="0" x2="0" y2="5" '
            'stroke="var(--fg)" stroke-width="1.6" opacity="0.55"/></pattern></defs>')
    return (f'<svg viewBox="0 0 {W} {y2+h2+34}" class="chart">' + defs
            + axis(n, h1, y1, "A1 · unnormalized accumulated log-weight (reversals counted here)")
            + a1
            + axis(n, h2, y2, "A2 · normalized weight (stacked — argmax churn = top band changing colour)")
            + "".join(bands) + "</svg>")


def panel_b(steps: List[StepRecord], n_particles: int) -> str:
    n = len(steps)
    h, y0 = 170, 28
    le = [s.likelihood_ess for s in steps]
    pe = [s.posterior_ess for s in steps]
    pop = [s.population_post or len(s.particles) for s in steps]
    allv = [v for v in le + pe + [float(x) for x in pop] if v is not None] + [n_particles]
    lo, hi = _scale(allv, lo=0)

    def line(vals, cls):
        pts = [f"{_x(i,n,W):.1f},{y0+h-(v-lo)/(hi-lo)*h:.1f}" for i, v in enumerate(vals) if v is not None]
        return f'<polyline points="{" ".join(pts)}" fill="none" class="{cls}"/>' if pts else ""

    refs = []
    for val, lab, cls in ((0.75 * n_particles, f"0.75·N = {0.75*n_particles:.2f} (Phase 1 gate)", "gate"),
                          (steps[0].ess_threshold if steps and steps[0].ess_threshold else n_particles / 2,
                           f"resample threshold {steps[0].ess_threshold_name or 'N/2'}", "thr")):
        y = y0 + h - (val - lo) / (hi - lo) * h
        refs.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" class="{cls}"/>'
                    f'<text x="{W-PAD_R}" y="{y-4:.1f}" class="reflab" text-anchor="end">{html.escape(lab)}</text>')

    ticks = "".join(
        f'<line x1="{_x(i,n,W):.1f}" y1="{y0}" x2="{_x(i,n,W):.1f}" y2="{y0+h}" class="rs"/>'
        for i, s in enumerate(steps) if "resample" in s.operators_fired)

    return (f'<svg viewBox="0 0 {W} {y0+h+34}" class="chart">'
            + axis(n, h, y0, "B · ESS and population (gap between the two lines is the story)")
            + ticks + "".join(refs) + line(le, "le") + line(pe, "pe")
            + line([float(p) for p in pop], "pop") + "</svg>")


def panel_c(steps: List[StepRecord]) -> str:
    n = len(steps)
    rows = ["resample", "perturb", "merge", "split", "anchor_revision"]
    rh, y0 = 22, 28
    out = []
    for r, name in enumerate(rows):
        y = y0 + r * rh
        out.append(f'<text x="8" y="{y+14}" class="rowlab">{name}</text>')
        out.append(f'<line x1="{PAD_L}" y1="{y+10}" x2="{W-PAD_R}" y2="{y+10}" class="rowline"/>')
        for i, s in enumerate(steps):
            if name in s.operators_fired:
                out.append(f'<circle cx="{_x(i,n,W):.1f}" cy="{y+10}" r="4.5" class="op op-{name}"/>')

    y1 = y0 + len(rows) * rh + 22
    out.append(f'<text x="8" y="{y1+10}" class="rowlab">mass cond</text>')
    out.append(f'<text x="8" y="{y1+30}" class="rowlab">rank cond</text>')
    any_cond = False
    for i, s in enumerate(steps):
        m = sum(1 for p in s.particles if p.mass_condition_met)
        rk = sum(1 for p in s.particles if p.rank_condition_met)
        if m or rk:
            any_cond = True
        x = _x(i, n, W)
        w = max(2.0, (W - PAD_L - PAD_R) / max(1, n) * 0.7)
        out.append(f'<rect x="{x-w/2:.1f}" y="{y1}" width="{w:.1f}" height="14" '
                   f'fill="hsl(28,80%,50%)" opacity="{0.12 + 0.88*min(1,m/max(1,len(s.particles))):.2f}"/>')
        out.append(f'<rect x="{x-w/2:.1f}" y="{y1+20}" width="{w:.1f}" height="14" '
                   f'fill="hsl(200,72%,48%)" opacity="{0.12 + 0.88*min(1,rk/max(1,len(s.particles))):.2f}"/>')
    note = ("" if any_cond else
            f'<text x="{PAD_L}" y="{y1+52}" class="nodata">split conditions not wired yet &mdash; '
            'until they are, a zero split count cannot be told apart from "split can never fire"</text>')

    y2 = y1 + 66
    out.append(f'<text x="8" y="{y2+12}" class="rowlab">perturb a/r</text>')
    seen_pr = False
    for i, s in enumerate(steps):
        acc = sum(1 for p in s.particles if p.perturb_accepted is True)
        rej = sum(1 for p in s.particles if p.perturb_accepted is False)
        if acc or rej:
            seen_pr = True
            tot = acc + rej
            x, w = _x(i, n, W), 6
            ha = 18 * acc / tot
            out.append(f'<rect x="{x-w/2}" y="{y2+18-ha:.1f}" width="{w}" height="{ha:.1f}" fill="hsl(150,60%,42%)"/>')
            out.append(f'<rect x="{x-w/2}" y="{y2}" width="{w}" height="{18-ha:.1f}" fill="hsl(0,62%,52%)"/>')
    if not seen_pr:
        out.append(f'<text x="{PAD_L}" y="{y2+14}" class="nodata">no perturbation accept/reject recorded yet</text>')

    return (f'<svg viewBox="0 0 {W} {y2+40}" class="chart">'
            + f'<text x="{PAD_L}" y="{y0-6}" class="ptitle">C · operators, split diagnostic, perturbation accept/reject</text>'
            + "".join(out) + note + "</svg>")


def panel_d(steps: List[StepRecord]) -> str:
    n = len(steps)
    h, y0 = 150, 28
    mc = [s.mean_pairwise_cosine for s in steps]
    nc = [s.min_pairwise_cosine for s in steps]
    mj = [s.mean_pairwise_jaccard for s in steps]
    if not any(v is not None for v in mc + nc + mj):
        return (f'<svg viewBox="0 0 {W} 90" class="chart">'
                f'<text x="{PAD_L}" y="22" class="ptitle">D · diversity</text>'
                f'<text x="{PAD_L}" y="52" class="nodata">no diversity fields recorded yet</text></svg>')
    lo, hi = 0.0, 1.0

    def line(vals, cls):
        pts = [f"{_x(i,n,W):.1f},{y0+h-(v-lo)/(hi-lo)*h:.1f}" for i, v in enumerate(vals) if v is not None]
        return f'<polyline points="{" ".join(pts)}" fill="none" class="{cls}"/>' if pts else ""

    refs = ""
    for val, lab, cls in ((0.90, "merge 0.90", "gate"), (0.25, "rejuvenate 0.25", "thr")):
        y = y0 + h - (val - lo) / (hi - lo) * h
        refs += (f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" class="{cls}"/>'
                 f'<text x="{W-PAD_R}" y="{y-4:.1f}" class="reflab" text-anchor="end">{lab}</text>')
    return (f'<svg viewBox="0 0 {W} {y0+h+34}" class="chart">'
            + axis(n, h, y0, "D · diversity (read against B: flat likelihood ESS + low diversity = particle problem)")
            + refs + line(mc, "le") + line(nc, "pe") + line(mj, "pop") + "</svg>")


def panel_e(summary: Dict, floor: Optional[Dict]) -> str:
    cards = []
    for k in FLOOR_KEYS:
        v = summary.get(k)
        f = (floor or {}).get(k)
        if v is None:
            cards.append(f'<div class="card na"><div class="k">{k}</div><div class="v">&mdash;</div>'
                         f'<div class="f">not wired up</div></div>')
            continue
        if f is None:
            cards.append(f'<div class="card na"><div class="k">{k}</div><div class="v">{v}</div>'
                         f'<div class="f">no floor yet</div></div>')
        else:
            good = v > f
            cards.append(f'<div class="card {"good" if good else "under"}"><div class="k">{k}</div>'
                         f'<div class="v">{v}</div><div class="f">silver p95 = {f}</div></div>')
    meta_keys = ["corpus", "role", "corpus_role", "context_id", "target_agent", "n_hypotheses",
                 "model", "tracer_type", "steps", "median_likelihood_ess", "median_posterior_ess",
                 "effective_temperatures", "total_llm_calls", "wall_time_s"]
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(summary.get(k)))}</td></tr>"
                   for k in meta_keys if k in summary)
    return f'<div class="scoreboard">{"".join(cards)}</div><table class="meta">{rows}</table>'


CSS = """
:root{--bg:#fbfbfd;--fg:#1b1d22;--mut:#6b7280;--line:#d9dce3;--card:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14161a;--fg:#e8eaee;--mut:#98a0ad;--line:#2c313a;--card:#1c1f25}}
:root[data-theme=dark]{--bg:#14161a;--fg:#e8eaee;--mut:#98a0ad;--line:#2c313a;--card:#1c1f25}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:0 16px 48px}
h1{font-size:19px;margin:22px 0 4px}.sub{color:var(--mut);margin:0 0 18px;font-size:13px}
.scoreboard{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px 13px;min-width:140px}
.card .k{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:23px;font-weight:600;margin:3px 0}
.card .f{font-size:11px;color:var(--mut)}
.card.good{border-left:4px solid hsl(150,58%,42%)}
.card.under{border-left:4px solid var(--line)}
.card.na{opacity:.62;border-left:4px dashed var(--line)}
table.meta{border-collapse:collapse;margin:8px 0 26px;font-size:12px}
table.meta td{border-bottom:1px solid var(--line);padding:3px 14px 3px 0;color:var(--mut)}
table.meta td:first-child{font-weight:600;color:var(--fg)}
.chart{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:9px;margin:0 0 16px;display:block}
.ax{stroke:var(--line);stroke-width:1}
.tick,.reflab,.rowlab{fill:var(--mut);font-size:10px}
.ptitle{fill:var(--fg);font-size:12px;font-weight:600}
.nodata{fill:var(--mut);font-size:12px;font-style:italic}
.rev{fill:hsl(350,76%,54%);stroke:var(--card);stroke-width:1}
.le{stroke:hsl(205,74%,50%);stroke-width:2}
.pe{stroke:hsl(28,80%,52%);stroke-width:2}
.pop{stroke:var(--mut);stroke-width:1.4;stroke-dasharray:4 3}
.gate{stroke:hsl(150,52%,44%);stroke-dasharray:6 4;stroke-width:1}
.thr{stroke:hsl(350,60%,56%);stroke-dasharray:3 3;stroke-width:1}
.rs{stroke:var(--line);stroke-width:1}
.rowline{stroke:var(--line);stroke-width:1}
.op{fill:hsl(262,58%,58%)}
.legend{color:var(--mut);font-size:12px;margin:-6px 0 16px}
.legend b{color:var(--fg)}
"""


def render(steps: List[StepRecord], summary: Dict, floor: Optional[Dict], deadband: float) -> str:
    palette = build_palette(steps)
    n_lineages = len(palette)
    n_particles = summary.get("n_hypotheses") or (len(steps[0].particles) if steps else 4)

    title = html.escape(str(summary.get("context_id", summary.get("run_id", "run"))))
    body = (f"<h1>{title}</h1>"
            f'<p class="sub">{html.escape(str(summary.get("corpus","")))} &middot; '
            f'{html.escape(str(summary.get("role","")))} &middot; '
            f'{len(steps)} steps &middot; N={n_particles}</p>'
            + panel_e(summary, floor)
            + '<p class="legend"><b>A1</b> each particle\'s own evidence &mdash; monotone lines mean a linear '
              'filter. <b>B</b> blue = likelihood ESS (did the evaluator discriminate), orange = posterior ESS '
              '(did accumulation concentrate). <b>C</b> read the split diagnostic first whenever split count is 0.</p>'
            + panel_a(steps, palette, deadband)
            + panel_b(steps, n_particles)
            + panel_d(steps)
            + panel_c(steps))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>')


MIN_SILVER_RUNS = 3


def silver_floor(runs_path: str, exclude_run_id: Optional[str] = None, pct: float = 95.0) -> Optional[Dict]:
    """p95 over the silver control runs.

    Two guards, both load-bearing:
      * the run being rendered is excluded -- grading a run against a floor
        derived partly from itself is circular and would always look reasonable;
      * fewer than MIN_SILVER_RUNS silver runs yields no floor at all rather
        than a floor of one sample, which would be noise presented as a bar.
    The silver set and the percentile are fixed in advance. Re-deriving the
    floor after seeing gold turns the control into a free parameter.
    """
    if not os.path.exists(runs_path):
        return None
    rows = []
    with open(runs_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("role") != "silver":
                continue
            if exclude_run_id and r.get("run_id") == exclude_run_id:
                continue
            rows.append(r)
    if len(rows) < MIN_SILVER_RUNS:
        return None
    out = {}
    for k in FLOOR_KEYS:
        vals = sorted(r[k] for r in rows if r.get(k) is not None)
        if vals:
            idx = min(len(vals) - 1, int(math.ceil(pct / 100.0 * len(vals))) - 1)
            out[k] = vals[max(0, idx)]
    out["_n_silver_runs"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps_jsonl")
    ap.add_argument("--runs", default=None, help="runs.jsonl for the silver floor")
    ap.add_argument("--out", default=None)
    ap.add_argument("--deadband", type=float, default=1e-3)
    a = ap.parse_args()

    steps = trace_log.read_steps(a.steps_jsonl)
    run_id = os.path.basename(a.steps_jsonl).replace(".steps.jsonl", "")
    runs_path = a.runs or os.path.join(os.path.dirname(a.steps_jsonl) or ".", "runs.jsonl")

    summary = {"run_id": run_id, "steps": len(steps)}
    if os.path.exists(runs_path):
        with open(runs_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("run_id") == run_id:
                        summary = r
    summary.update(trace_log.nonlinearity_summary(steps, a.deadband))

    out = a.out or os.path.join(os.path.dirname(a.steps_jsonl) or ".", f"{run_id}.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render(steps, summary, silver_floor(runs_path, exclude_run_id=run_id), a.deadband))
    print(f"wrote {out}  ({os.path.getsize(out)} bytes, {len(steps)} steps)")


if __name__ == "__main__":
    main()

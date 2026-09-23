"""One target, every generation of the filter, stacked on a shared scene axis.

`stacked.py` answers "how do two agents differ in one scene". This answers the
other question that kept coming up: "did the thing we changed actually change
the trace". Same panels, same legend, same tooltips -- reusing
`stacked.panel()` rather than reimplementing it, so a fork, a death and an
operator tag mean exactly what they mean on the other report.

The shared x axis is what makes it readable: every panel is anchored on the
real turn each step scored, so a commitment appearing at turn 157 in one
generation lines up with turn 157 in all the others. Vertically stacked rather
than side by side for the same reason -- the eye compares along a column.

Usage:
  evolution.py ep_Katara fix_katara v2_katara v3_katara \\
      --labels "G0 baseline,G1 parser,G2 form,G3 null+surprise" \\
      --corpus atla --align-set atla-0604,...,atla-0611 \\
      --title "Katara, four generations" --out musing_out/reports/evolution_katara.html
"""
import argparse
import html

import musing_layout as ml
import stacked as st
import weight_chart as wc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run ids, oldest generation first")
    ap.add_argument("--labels", default="", help="comma-separated panel labels")
    ap.add_argument("--title", default="How the trace changed")
    ap.add_argument("--blurb", default="")
    ap.add_argument("--scene", default="")
    ap.add_argument("--align-set", default=None)
    ap.add_argument("--corpus", default="atla")
    ap.add_argument("--out", default=None)
    # Weights sum to 1 over 5-8 particles, so the median line sits near 1/n while
    # a single leader can reach 0.9. Scaling to the global max therefore leaves
    # ~half of every panel empty and crushes the bulk of the data into the bottom
    # sliver (measured: median line at 13% of panel height). Taller panels and a
    # settable ceiling are the two knobs that fix it without rescaling each panel
    # separately, which would destroy the cross-generation comparison.
    ap.add_argument("--height", type=int, default=330, help="panel height in px")
    ap.add_argument("--ymax", type=float, default=None,
                    help="weight-axis ceiling shared by all panels; default is a "
                         "percentile of the data, not the maximum")
    ap.add_argument("--ymax-pct", type=float, default=100.0,
                    help="percentile for the ceiling; 100 (the default) uses the true maximum, so no point is ever clipped. Lower it to zoom into the low-weight traffic.")
    a = ap.parse_args()

    steps = [st.load(r) for r in a.runs]
    labels = [x.strip() for x in a.labels.split(",")] if a.labels else []
    labels += a.runs[len(labels):]

    # Every panel here is the SAME target on the SAME input, so the generations
    # share a trajectory step for step (verified: 17/17/17, 15/15/15, 41x5, 37x5).
    # Step index therefore aligns them exactly, and scene alignment is not only
    # unnecessary but actively wrong on this corpus: scene_positions matches a
    # step to a turn by (speaker, first 40 chars), which fails when speakers are
    # merged into a coalition or the text is reformatted. Measured on the
    # Bloomfield runs it matched 8/17 and 9/15, and the unmatched steps collapse
    # onto the monotonic fallback -- 17 steps crammed into the left 13% of a
    # 66-turn axis. ATLA matched 39/41 and was fine, which is why this hid.
    xs = [None] * len(steps)
    if a.align_set:
        turns = st.scene_turns(a.corpus, a.align_set)
        xs = [st.scene_positions(s, turns) for s in steps]
        matched = sum(1 for x in xs[0] if float(x).is_integer())
        if matched < 0.8 * len(xs[0]):
            print(f"   scene alignment matched only {matched}/{len(xs[0])} steps "
                  f"-- falling back to the step axis, which is exact here anyway")
            xs = [None] * len(steps)
            n = max(len(s) for s in steps)
            axis = "step (the target's own turns)"
        else:
            n = len(turns)
            axis = "turn in the scene"
    else:
        n = max(len(s) for s in steps)
        axis = "step (each run's own turns)"

    # ONE weight scale across every panel. Generations differ in how much mass
    # the leader takes, which is a big part of what changed -- rescaling each
    # panel to its own maximum would hide exactly that.
    ws = sorted(p.weight or 0 for s in steps for r in s for p in r.particles) or [1.0]
    if a.ymax:
        ymax = a.ymax
    elif a.ymax_pct >= 100:
        ymax = max(0.35, 1.03 * ws[-1])
    else:
        ymax = max(0.35, 1.05 * ws[min(len(ws) - 1, int(a.ymax_pct / 100 * (len(ws) - 1)))])
    st.H = a.height
    over = sum(1 for w in ws if w > ymax)

    body, lines = [], []
    for i, (s, lab, x) in enumerate(zip(steps, labels, xs)):
        y0 = st.TOP + i * (st.H + st.GAP)
        p, nl = st.panel(s, y0, ymax, n, lab, f"{len(s)} turns", x)
        # panel() bakes the note before it knows the line count; patch it back
        p = p.replace(f"{len(s)} turns<", f"{len(s)} turns, {nl} hypotheses<")
        body.append(p)
        lines.append(nl)

    height = st.TOP + len(steps) * (st.H + st.GAP) + 10
    src = open("weight_chart.py").read()
    css = src.split('css = """')[1].split('"""')[0]
    js = src.split('JS = """')[1].split('"""')[0]
    css += "\n.ti{fill:var(--fg);font-size:13px;font-weight:650}"
    css += "\ncode{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;background:var(--line);padding:1px 5px;border-radius:4px}"

    out = a.out or ml.report_path("evolution.html")
    blurb = a.blurb or (
        f"The same trajectory{(' (' + html.escape(a.scene) + ')') if a.scene else ''}, "
        f"traced once per generation of the filter. Every panel shares the turn axis and the "
        f"weight scale, so a column reads as one moment in the conversation across every "
        f"version. Read down a column to see what each fix changed at that moment.")
    open(out, "w", encoding="utf-8").write(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(a.title)}</title><style>{css}</style></head><body><div class="wrap">'
        f'<h1>{html.escape(a.title)}</h1><p>{blurb}</p>'
        f'<p>Horizontal axis: {axis}. Weight axis is shared by every panel and '
        f'capped at {ymax:.2f}' + (f' &mdash; {over} of {len(ws)} points sit above it and are '
        f'drawn at the ceiling with a caret; hover gives the true weight.' if over else '.') + '</p>'
        f'{wc.LEGEND}'
        f'<svg viewBox="0 0 {st.W} {height}">{"".join(body)}</svg>'
        f'<div id="tip"></div><script>{js}</script>'
        f'</div></body></html>')
    print(f"wrote {out}   ymax {ymax:.3f} ({over} of {len(ws)} points above it, drawn at the ceiling)")
    for lab, s, nl in zip(labels, steps, lines):
        print(f"   {lab:<22} {len(s):>3} turns  {nl:>3} hypotheses")


if __name__ == "__main__":
    main()

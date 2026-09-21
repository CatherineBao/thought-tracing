"""Side-by-side: same context, same N, same turns. Only the filter differs."""
import glob, html, sys
import trace_log as t

CW, PL, PR, H, Y0 = 470, 52, 60, 250, 26


def build(st):
    series, parent_of, born, prev = {}, {}, {}, {}
    for i, s in enumerate(st):
        cur = {}
        for p in s.particles:
            lid = p.lineage_id or p.particle_id
            series.setdefault(lid, {})[i] = p.weight or 0.0
            if lid not in born:
                born[lid] = i
                parent_of[lid] = prev.get(p.parent_id)
            cur[p.particle_id] = lid
        prev = cur
    return series, parent_of, born


def panel(st, title, ymax):
    n = len(st)
    series, parent_of, born = build(st)
    lids = sorted(series, key=lambda k: (born[k], k))
    hue = {l: (i * 53) % 360 for i, l in enumerate(lids)}
    X = lambda i: PL + (CW - PL - PR) * (i / max(1, n - 1))
    Y = lambda v: Y0 + H - (v / ymax) * H
    o = [f'<text x="{PL}" y="{Y0-10}" class="pt">{html.escape(title)}</text>']
    for k in range(5):
        v = ymax * k / 4
        o.append(f'<line x1="{PL}" y1="{Y(v):.1f}" x2="{CW-PR}" y2="{Y(v):.1f}" class="g"/>')
        o.append(f'<text x="{PL-6}" y="{Y(v)+4:.1f}" class="tk" text-anchor="end">{v:.2f}</text>')
    for i in range(0, n, max(1, n // 6)):
        o.append(f'<text x="{X(i):.1f}" y="{Y0+H+15}" class="tk" text-anchor="middle">{i}</text>')
    for i, s in enumerate(st):
        if s.operators_fired:
            tag = "".join(c[0].upper() for c in s.operators_fired)
            o.append(f'<line x1="{X(i):.1f}" y1="{Y0}" x2="{X(i):.1f}" y2="{Y0+H}" class="op"/>')
            o.append(f'<text x="{X(i):.1f}" y="{Y0-1}" class="tk" text-anchor="middle">{tag}</text>')
    for l in lids:
        par, b = parent_of.get(l), born[l]
        if par and par in series and b > 0 and (b - 1) in series[par]:
            o.append(f'<line x1="{X(b-1):.1f}" y1="{Y(series[par][b-1]):.1f}" x2="{X(b):.1f}" '
                     f'y2="{Y(series[l][b]):.1f}" stroke="hsl({hue[l]},55%,55%)" '
                     f'stroke-width="1.1" stroke-dasharray="3 2" opacity=".8"/>')
    for l in lids:
        pts = [f"{X(i):.1f},{Y(v):.1f}" for i, v in sorted(series[l].items())]
        if len(pts) > 1:
            o.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                     f'stroke="hsl({hue[l]},58%,52%)" stroke-width="1.9" opacity=".9"/>')
        else:
            i, v = next(iter(series[l].items()))
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2.6" fill="hsl({hue[l]},58%,52%)"/>')
    for i, s in enumerate(st):
        ps = [p for p in s.particles if p.weight is not None]
        if ps:
            top = max(ps, key=lambda p: p.weight)
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(top.weight):.1f}" r="2.3" fill="none" '
                     f'stroke="var(--fg)" stroke-width="1.2" opacity=".75"/>')
    return f'<svg viewBox="0 0 {CW} {Y0+H+30}">{"".join(o)}</svg>', len(lids)


def stats(st):
    lids = {p.lineage_id or p.particle_id for s in st for p in s.particles}
    w = [(p.weight or 0) for s in st for p in s.particles]
    summ = t.nonlinearity_summary(st)
    ops = {}
    for s in st:
        for o in s.operators_fired:
            ops[o] = ops.get(o, 0) + 1
    return {
        "lines": len(lids),
        "spread": f"{min(w):.3f} &ndash; {max(w):.3f}",
        "commitments": f"{st[0].distinct_roots} &rarr; {st[-1].distinct_roots}",
        "born": max(0, len(lids) - st[0].distinct_roots),
        "lead changes": summ["argmax_churn"],
        "reversals": summ["reversals_median_long_lived"],
        "operators": ", ".join(f"{k} {v}" for k, v in sorted(ops.items())) or "none fired",
    }


b = sorted(glob.glob("musing_out/ab_before_1*.steps.jsonl"))
a = sorted(glob.glob("musing_out/ab_after_1*.steps.jsonl"))
if not b or not a:
    sys.exit("need both arms")
bs, as_ = t.read_steps(b[0]), t.read_steps(a[0])
ymax = max(max((p.weight or 0) for s in bs + as_ for p in s.particles) * 1.1, 0.3)
psvg, bl = panel(bs, "BEFORE — original algorithm", ymax)
qsvg, al = panel(as_, "AFTER — full stack", ymax)
sb, sa = stats(bs), stats(as_)
rows = "".join(f"<tr><td>{k}</td><td class=b>{sb[k]}</td><td class=a>{sa[k]}</td></tr>" for k in sb)

CSS = """:root{--bg:#fbfbfd;--fg:#1b1d22;--mut:#6b7280;--line:#dde0e6;--card:#fff;--bc:#c2410c;--ac:#0f766e}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#131519;--fg:#e9ebef;--mut:#98a0ad;--line:#2b303a;--card:#1b1e24;--bc:#fb923c;--ac:#2dd4bf}}
body{margin:0;padding:0 18px 50px;background:var(--bg);color:var(--fg);font:15px/1.6 ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1000px;margin:0 auto}h1{font-size:22px;margin:26px 0 4px}p{color:var(--mut);margin:6px 0}
.row{display:flex;gap:14px;flex-wrap:wrap}.col{flex:1;min-width:330px}
svg{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
.g{stroke:var(--line);stroke-width:1}.tk{fill:var(--mut);font-size:9px}
.pt{fill:var(--fg);font-size:11px;font-weight:600}.op{stroke:var(--fg);opacity:.18;stroke-dasharray:3 3}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:16px}
td,th{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
td.b{color:var(--bc);font-weight:600}td.a{color:var(--ac);font-weight:600}"""

open("musing_out/side_by_side.html", "w", encoding="utf-8").write(
    f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
    f'<meta name="viewport" content="width=device-width,initial-scale=1">'
    f'<title>Before / after</title><style>{CSS}</style></head><body><div class="wrap">'
    f'<h1>Hypothesis weight by turn &mdash; same context, same N</h1>'
    f'<p>Oppenheimer, Strauss 0045&rarr;0052, {len(bs)} turns, N=8. One line per hypothesis. '
    f'Ringed dots mark the strongest at each turn; dashed connectors are forks; vertical dashes '
    f'mark operators (R resample, P perturb, S split, M merge).</p>'
    f'<div class="row"><div class="col">{psvg}</div><div class="col">{qsvg}</div></div>'
    f'<table><tr><th></th><th>before</th><th>after</th></tr>{rows}</table>'
    f'</div></body></html>')
print(f"wrote musing_out/side_by_side.html  before={bl} lines  after={al} lines")

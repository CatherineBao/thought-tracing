"""Per-hypothesis weight over turns, with forks at split and joins at merge.

One line per lineage. A split forks the parent line into its children; a merge
joins two lines; a perturbation ends one line and starts another (the commitment
was replaced, so it is genuinely a different hypothesis from that turn on).
"""
import argparse, glob, html, sys
import trace_log as t

W, PL, PR, H, Y0 = 940, 62, 150, 330, 30


def build(st):
    """lineage -> series, plus the text/anchor it held at each turn."""
    series, parent_of, born, detail = {}, {}, {}, {}
    prev_by_pid = {}
    for i, s in enumerate(st):
        cur = {}
        for p in s.particles:
            lid = p.lineage_id or p.particle_id
            series.setdefault(lid, {})[i] = p.weight or 0.0
            detail.setdefault(lid, {})[i] = (p.anchor or "", p.text or "")
            if lid not in born:
                born[lid] = i
                parent_of[lid] = prev_by_pid.get(p.parent_id)
            cur[p.particle_id] = lid
        prev_by_pid = cur
    return series, parent_of, born, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fs = sorted(glob.glob(f"musing_out/{a.run}*.steps.jsonl"))
    if not fs:
        sys.exit(f"no steps for {a.run}")
    st = t.read_steps(fs[0])
    n = len(st)
    series, parent_of, born, detail = build(st)

    # label + colour per lineage, ordered by first appearance
    lids = sorted(series, key=lambda k: (born[k], k))
    lab = {l: (chr(ord('A') + i) if i < 26 else f"z{i}") for i, l in enumerate(lids)}
    hue = {l: (i * 53) % 360 for i, l in enumerate(lids)}
    ymax = max((v for d in series.values() for v in d.values()), default=1.0)
    ymax = max(0.35, ymax * 1.08)

    def X(i): return PL + (W - PL - PR) * (i / max(1, n - 1))
    def Y(v): return Y0 + H - (v / ymax) * H

    o = []
    # grid
    for k in range(6):
        v = ymax * k / 5
        o.append(f'<line x1="{PL}" y1="{Y(v):.1f}" x2="{W-PR}" y2="{Y(v):.1f}" class="g"/>')
        o.append(f'<text x="{PL-8}" y="{Y(v)+4:.1f}" class="tk" text-anchor="end">{v:.2f}</text>')
    for i in range(0, n, max(1, n // 12)):
        o.append(f'<text x="{X(i):.1f}" y="{Y0+H+16}" class="tk" text-anchor="middle">{i}</text>')
    # operator markers
    for i, s in enumerate(st):
        if s.operators_fired:
            tag = "".join(c[0].upper() for c in s.operators_fired)
            o.append(f'<line x1="{X(i):.1f}" y1="{Y0}" x2="{X(i):.1f}" y2="{Y0+H}" class="op"/>')
            o.append(f'<text x="{X(i):.1f}" y="{Y0-6}" class="tk" text-anchor="middle">{tag}</text>')
    # fork connectors: parent's last point -> child's first point
    for l in lids:
        par = parent_of.get(l)
        b = born[l]
        if par and par in series and b > 0 and (b - 1) in series[par]:
            o.append(f'<line x1="{X(b-1):.1f}" y1="{Y(series[par][b-1]):.1f}" '
                     f'x2="{X(b):.1f}" y2="{Y(series[l][b]):.1f}" '
                     f'stroke="hsl({hue[l]},55%,55%)" stroke-width="1.2" '
                     f'stroke-dasharray="3 2" opacity=".75"/>')
    # lines
    for l in lids:
        pts = [f"{X(i):.1f},{Y(v):.1f}" for i, v in sorted(series[l].items())]
        if len(pts) < 2:
            i, v = next(iter(series[l].items()))
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="hsl({hue[l]},58%,52%)"/>')
            continue
        o.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                 f'stroke="hsl({hue[l]},58%,52%)" stroke-width="2" opacity=".92"/>')
        # invisible wide stroke: the hover target for the whole line
        first = min(detail[l]); anc0, _ = detail[l][first]
        o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="transparent" '
                 f'stroke-width="12" class="hit" data-lab="{html.escape(lab[l])}" '
                 f'data-anchor="{html.escape(anc0)}" data-born="{first}"/>')
        # per-point targets carry that turn's belief text
        for i, v in sorted(series[l].items()):
            anc, txt = detail[l][i]
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5" fill="transparent" '
                     f'class="hit" data-lab="{html.escape(lab[l])}" data-turn="{i}" '
                     f'data-w="{v:.3f}" data-anchor="{html.escape(anc)}" '
                     f'data-text="{html.escape(txt[:600])}"/>')
        li, lv = max(series[l].items())
        o.append(f'<text x="{X(li)+6:.1f}" y="{Y(lv)+4:.1f}" class="lb" '
                 f'fill="hsl({hue[l]},58%,42%)">{lab[l]}</text>')
    # strongest-at-each-turn marker
    for i, s in enumerate(st):
        ps = [p for p in s.particles if p.weight is not None]
        if not ps: continue
        top = max(ps, key=lambda p: p.weight)
        o.append(f'<circle cx="{X(i):.1f}" cy="{Y(top.weight):.1f}" r="2.6" '
                 f'fill="none" stroke="var(--fg)" stroke-width="1.3" opacity=".8"/>')

    # commitments legend
    seen = {}
    for s in st:
        for p in s.particles:
            l = p.lineage_id or p.particle_id
            if l not in seen and p.anchor:
                seen[l] = (born[l], p.anchor)
    rows = "".join(
        f'<tr><td style="color:hsl({hue[l]},58%,45%)"><b>{lab[l]}</b></td>'
        f'<td>{seen[l][0]}</td><td>{html.escape(seen[l][1][:74])}</td></tr>'
        for l in lids if l in seen)

    css = """:root{--bg:#fbfbfd;--fg:#1b1d22;--mut:#6b7280;--line:#dde0e6;--card:#fff}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#131519;--fg:#e9ebef;--mut:#98a0ad;--line:#2b303a;--card:#1b1e24}}
body{margin:0;padding:0 18px 50px;background:var(--bg);color:var(--fg);font:15px/1.6 ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto}h1{font-size:22px;margin:26px 0 4px}p{color:var(--mut)}
svg{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;margin:10px 0}
.g{stroke:var(--line);stroke-width:1}.tk{fill:var(--mut);font-size:10px}
.lb{font-size:11px;font-weight:700}.op{stroke:var(--fg);opacity:.18;stroke-dasharray:3 3}
table{border-collapse:collapse;font-size:13px;margin-top:10px}
td{border-bottom:1px solid var(--line);padding:5px 12px 5px 0;vertical-align:top}
.k{color:var(--mut);font-size:12px}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .08s;z-index:9;
max-width:430px;background:var(--card);color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:10px 12px;font-size:12.5px;line-height:1.45;
box-shadow:0 6px 22px rgba(0,0,0,.18)}
#tip .h{font-weight:700;margin-bottom:3px}
#tip .a{color:var(--mut);font-style:italic;margin-bottom:6px}
.hit{cursor:crosshair}"""

    JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('.hit').forEach(el=>{
  el.addEventListener('mousemove',e=>{
    const d=el.dataset;
    let h='<div class="h">Hypothesis '+d.lab+(d.turn!==undefined?' &middot; turn '+d.turn:'')+
          (d.w?' &middot; weight '+d.w:'')+'</div>';
    if(d.anchor) h+='<div class="a">wants: '+d.anchor+'</div>';
    h+= d.text ? d.text : ('first appears at turn '+(d.born||0));
    tip.innerHTML=h; tip.style.opacity=1;
    const pad=14, w=tip.offsetWidth, ht=tip.offsetHeight;
    let x=e.clientX+pad, y=e.clientY+pad;
    if(x+w>innerWidth-8) x=e.clientX-w-pad;
    if(y+ht>innerHeight-8) y=e.clientY-ht-pad;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  });
  el.addEventListener('mouseleave',()=>tip.style.opacity=0);
});
    """
    out = a.out or f"musing_out/{a.run}_weights.html"
    open(out, "w", encoding="utf-8").write(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Hypothesis weights</title><style>{css}</style></head><body><div class="wrap">'
        f'<h1>Hypothesis weight by turn</h1>'
        f'<p>One line per hypothesis. Dashed connectors are forks &mdash; a split line, or a '
        f'replaced commitment. Ringed dots mark the strongest hypothesis at each turn. '
        f'Vertical dashes mark operators (R resample, P perturb, S split, M merge).</p>'
        f'<svg viewBox="0 0 {W} {Y0+H+34}">{"".join(o)}</svg>'
        f'<div id="tip"></div>'
        f'<script>{JS}</script>' 
        f'<p class="k">commitments, with the turn they first appear</p>'
        f'<table>{rows}</table></div></body></html>')
    print(f"wrote {out}  ({n} turns, {len(lids)} hypothesis lines)")


if __name__ == "__main__":
    main()

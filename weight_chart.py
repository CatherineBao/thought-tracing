"""Per-hypothesis weight over turns, with forks at split and joins at merge.

One line per lineage. A split forks the parent line into its children; a merge
joins two lines; a perturbation ends one line and starts another (the commitment
was replaced, so it is genuinely a different hypothesis from that turn on).
"""
import argparse, glob, html, sys
import musing_layout as ml
import trace_log as t

W, PL, PR, H, Y0 = 940, 62, 150, 330, 30

LEGEND = """
<div class="lg">
  <div class="li"><svg width="46" height="14">
    <polyline points="2,10 14,5 26,9 44,3" fill="none" stroke="hsl(200,58%,52%)" stroke-width="2"/>
    </svg><span><b>Solid line</b> &mdash; one hypothesis over time. It ends when that
    commitment is retired, merged away, or resampled out.</span></div>
  <div class="li"><svg width="46" height="14">
    <line x1="3" y1="11" x2="43" y2="3" stroke="hsl(320,55%,55%)" stroke-width="1.4"
      stroke-dasharray="3 2"/></svg>
    <span><b>Dotted diagonal</b> &mdash; a fork, running from the ringed fork node at the
    parent's last point to where the new hypothesis begins. Two dotted lines leaving one
    node is a split into mutually exclusive refinements; a single one is a commitment
    replaced by expiry or perturbation. Follow it to read what a hypothesis became.</span></div>
  <div class="li"><svg width="46" height="14">
    <circle cx="10" cy="7" r="3.4" fill="var(--card)" stroke="currentColor" stroke-width="1.5"/>
    <line x1="14" y1="6" x2="43" y2="2" stroke="hsl(320,60%,58%)" stroke-width="1.8" stroke-dasharray="4 2.5"/>
    <line x1="14" y1="8" x2="43" y2="12" stroke="hsl(150,60%,50%)" stroke-width="1.8" stroke-dasharray="4 2.5"/>
    </svg><span><b>Fork node</b> &mdash; where one hypothesis ends and its successors begin.
    Two lines leaving it is a split.</span></div>
  <div class="li"><svg width="46" height="14">
    <polyline points="2,3 16,5 30,12" fill="none" stroke="hsl(0,60%,55%)" stroke-width="2"/>
    <line x1="26" y1="8" x2="34" y2="16" stroke="hsl(0,60%,55%)" stroke-width="1.7"/>
    <line x1="26" y1="16" x2="34" y2="8" stroke="hsl(0,60%,55%)" stroke-width="1.7"/>
    </svg><span><b>Cross</b> &mdash; the hypothesis was removed here. The line falls to the
    weight it last held first, so you can see whether it was refuted by evidence or cleared
    while still strong.</span></div>
  <div class="li"><svg width="46" height="14">
    <circle cx="23" cy="7" r="3.2" fill="none" stroke="currentColor" stroke-width="1.4"/>
    </svg><span><b>Ringed dot</b> &mdash; the strongest hypothesis at that turn.</span></div>
  <div class="li"><svg width="46" height="14">
    <line x1="23" y1="1" x2="23" y2="13" stroke="currentColor" opacity=".45"
      stroke-dasharray="3 3"/></svg>
    <span><b>Vertical dash</b> &mdash; an operator fired, labelled above the chart.</span></div>
</div>
<table class="ops">
  <tr><td><b>R</b></td><td>resample</td><td>duplicate the strong, drop the weak</td></tr>
  <tr><td><b>P</b></td><td>perturb</td><td>replace a redundant copy with a new commitment</td></tr>
  <tr><td><b>E</b></td><td>expire</td><td>retire a hypothesis held below fair share for 6 turns, mint a replacement</td></tr>
  <tr><td><b>S</b></td><td>split</td><td>partition one hypothesis into 2&ndash;3 alternatives that cannot both hold</td></tr>
  <tr><td><b>X</b></td><td>expand</td><td>narrowed to a single more-specific commitment &mdash; not a partition, so it adds no diversity</td></tr>
  <tr><td><b>M</b></td><td>merge</td><td>absorb near-duplicate hypotheses into one</td></tr>
  <tr><td colspan="3" class="k">Letters combine when several fire on the same turn &mdash;
      <b>PE</b> is perturb + expire, <b>SM</b> is split then merge.</td></tr>
</table>
"""



def build(st):
    """One line per COMMITMENT, keyed on (lineage_id, root_id).

    NOT on lineage_id alone. Expiry and perturbation replace a commitment in
    place: the particle keeps its lineage_id while update_anchor mints a new
    root_id under the anchor==root invariant. Keying on lineage therefore drew
    every replacement as a continuation of the hypothesis it replaced -- in
    exp_4, 8 lines for 33 distinct roots, a population that looked untouched
    when in fact almost every commitment had been swapped out. One observed
    line held "Gather information" and then "Protect reputation" as a single
    unbroken series.
    """
    series, parent_of, born, detail = {}, {}, {}, {}
    last_key_of_lineage, prev_by_pid = {}, {}
    for i, s in enumerate(st):
        # SNAPSHOT taken before the step. Split gives its children the SAME
        # lineage_id -- they differ only by root_id -- so updating the lineage's
        # last-key inside the particle loop made the first child overwrite the
        # parent's entry, and the second child then forked from its SIBLING.
        # The parent's actual fork was never drawn. (parent_id cannot be used as
        # a fallback here: propagation reassigns particle_ids, so every logged
        # parent_id fails to resolve against the previous step.)
        at_step_start = dict(last_key_of_lineage)
        cur = {}
        for p in s.particles:
            lid = p.lineage_id or p.particle_id
            key = (lid, p.root_id)
            series.setdefault(key, {})[i] = p.weight or 0.0
            detail.setdefault(key, {})[i] = (p.anchor or "", p.text or "")
            if key not in born:
                born[key] = i
                # a REPLACED commitment forks from that same particle's previous
                # commitment; a genuinely new particle forks from its parent.
                parent_of[key] = (at_step_start.get(lid)
                                  or prev_by_pid.get(p.parent_id))
            last_key_of_lineage[lid] = key
            cur[p.particle_id] = key
        prev_by_pid = cur

    # EXTEND EACH DYING LINE TO ITS LAST OBSERVED WEIGHT.
    # A particle removed during step i is absent from step i's post-operator
    # population, so the line stopped at i-1 -- at whatever weight it last held.
    # T ended its line at 0.368, the leader, and simply vanished, which read as
    # a random death. In fact the scorer gave it 0.0 that turn and its weight
    # collapsed to 0.020 BEFORE merge absorbed it. That final value is recorded
    # in weights_pre, so draw it: the line falls to the floor and then ends,
    # which is what actually happened.
    removed = {}
    for i, s in enumerate(st):
        for q in (s.pre_operator_particles or []):
            key = (q.get('lineage_id') or q.get('particle_id'), q.get('root_id'))
            if key in series and i not in series[key] and max(series[key]) == i - 1:
                series[key][i] = float(q.get('weight') or 0.0)
                detail.setdefault(key, {})[i] = detail[key][i - 1]
                removed[key] = i
    return series, parent_of, born, detail, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fs = ml.find(f"{a.run}*.steps.jsonl")
    if not fs:
        sys.exit(f"no steps for {a.run}")
    st = t.read_steps(fs[0])
    n = len(st)
    series, parent_of, born, detail, removed = build(st)

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
    forks = set()
    for l in lids:
        par = parent_of.get(l)
        b = born[l]
        if par and par in series and b > 0 and (b - 1) in series[par]:
            x0, y0 = X(b - 1), Y(series[par][b - 1])
            o.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" '
                     f'x2="{X(b):.1f}" y2="{Y(series[l][b]):.1f}" '
                     f'stroke="hsl({hue[l]},60%,58%)" stroke-width="1.8" '
                     f'stroke-dasharray="4 2.5" opacity=".9"/>')
            # fork node at the origin: with two children this reads as one point
            # splitting in two, rather than two unrelated dotted strokes.
            forks.add((x0, y0))
    for fx, fy in forks:
        o.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="3.4" fill="var(--bg)" '
                 f'stroke="var(--fg)" stroke-width="1.5" opacity=".9"/>')
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
        if l in removed:
            # cross = removed here, after the drop the line just made
            cx, cy, r = X(li), Y(lv), 3.6
            o.append(f'<line x1="{cx-r:.1f}" y1="{cy-r:.1f}" x2="{cx+r:.1f}" y2="{cy+r:.1f}" '
                     f'stroke="hsl({hue[l]},58%,52%)" stroke-width="1.7" opacity=".95"/>')
            o.append(f'<line x1="{cx-r:.1f}" y1="{cy+r:.1f}" x2="{cx+r:.1f}" y2="{cy-r:.1f}" '
                     f'stroke="hsl({hue[l]},58%,52%)" stroke-width="1.7" opacity=".95"/>')
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
            l = ((p.lineage_id or p.particle_id), p.root_id)
            if l not in seen and p.anchor and l in born:
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
.hit{cursor:crosshair}
.lg{display:grid;grid-template-columns:1fr;gap:7px;margin:14px 0 6px;font-size:13px}
@media(min-width:680px){.lg{grid-template-columns:1fr 1fr;gap:9px 22px}}
.li{display:flex;gap:9px;align-items:flex-start;line-height:1.45}
.li svg{flex:0 0 46px;margin-top:2px;color:var(--fg)}
.li b{font-weight:650}
table.ops{border-collapse:collapse;font-size:13px;margin:12px 0 4px}
table.ops td{border-bottom:1px solid var(--line);padding:4px 14px 4px 0;vertical-align:top}
table.ops td:first-child{font-size:14px;width:26px}
table.ops td:nth-child(2){color:var(--fg);white-space:nowrap}
table.ops td:nth-child(3){color:var(--mut)}
table.ops td.k{color:var(--mut);border-bottom:none;padding-top:8px}"""

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
    out = a.out or ml.run_path(a.run, f"{a.run}_weights.html", create=True)
    open(out, "w", encoding="utf-8").write(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Hypothesis weights</title><style>{css}</style></head><body><div class="wrap">'
        f'<h1>Hypothesis weight by turn</h1>'
        f'<p>Each line is one hypothesis &mdash; one commitment about what the agent wants '
        f'&mdash; and its share of belief mass at each turn.</p>'
        f'{LEGEND}'
        f'<svg viewBox="0 0 {W} {Y0+H+34}">{"".join(o)}</svg>'
        f'<div id="tip"></div>'
        f'<script>{JS}</script>' 
        f'<p class="k">commitments, with the turn they first appear</p>'
        f'<table>{rows}</table></div></body></html>')
    print(f"wrote {out}  ({n} turns, {len(lids)} hypothesis lines)")


if __name__ == "__main__":
    main()

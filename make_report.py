"""Self-contained before/after HTML report. Offline: no CDN, no external font."""
import glob, html, json, statistics
import trace_log as t

W, PL, PR = 900, 56, 20

def load(pat, min_steps=20):
    out = []
    for f in sorted(glob.glob(pat)):
        st = t.read_steps(f)
        if len(st) >= min_steps:
            out.append((f.split("/")[-1].split("-")[0], st))
    return out

def x(i, n): return PL + (W - PL - PR) * (i / max(1, n - 1))

def axis(n, h, y0, title, ymax, ylab=""):
    o = [f'<text x="{PL}" y="{y0-8}" class="pt">{html.escape(title)}</text>']
    o.append(f'<line x1="{PL}" y1="{y0+h}" x2="{W-PR}" y2="{y0+h}" class="ax"/>')
    o.append(f'<line x1="{PL}" y1="{y0}" x2="{PL}" y2="{y0+h}" class="ax"/>')
    for k in range(0, 5):
        v = ymax * k / 4
        yy = y0 + h - (v / ymax) * h
        o.append(f'<line x1="{PL-4}" y1="{yy:.1f}" x2="{PL}" y2="{yy:.1f}" class="ax"/>')
        o.append(f'<text x="{PL-8}" y="{yy+4:.1f}" class="tk" text-anchor="end">{v:.0f}</text>')
    for i in range(0, n, max(1, n // 8)):
        o.append(f'<text x="{x(i,n):.1f}" y="{y0+h+15}" class="tk" text-anchor="middle">{i}</text>')
    if ylab:
        o.append(f'<text x="14" y="{y0+h/2}" class="tk" transform="rotate(-90 14 {y0+h/2})" text-anchor="middle">{ylab}</text>')
    return "".join(o)

def line(vals, n, h, y0, ymax, cls, w=2):
    pts = [f"{x(i,n):.1f},{y0+h-(min(v,ymax)/ymax)*h:.1f}" for i, v in enumerate(vals) if v is not None]
    return f'<polyline points="{" ".join(pts)}" fill="none" class="{cls}" stroke-width="{w}"/>' if pts else ""

# ---------- chart 1: commitments over time, before vs after ----------
def chart_commitments(before, after):
    n = max(max(len(s) for _, s in before), max(len(s) for _, s in after))
    h, y0 = 190, 26
    o = [axis(n, h, y0, "Distinct commitments alive, per step", 9, "commitments")]
    for _, st in before:
        o.append(line([s.distinct_roots for s in st], n, h, y0, 9, "before"))
    for _, st in after:
        o.append(line([s.distinct_roots for s in st], n, h, y0, 9, "after"))
    return f'<svg viewBox="0 0 {W} {y0+h+30}" class="c">' + "".join(o) + "</svg>"

# ---------- chart 2: the metric blindness ----------
def chart_blind(st):
    n = len(st); h, y0 = 170, 26
    part = [(s.posterior_ess / len(s.particles)) if s.posterior_ess else None for s in st]
    root = [s.root_mass_ess for s in st]
    o = [axis(n, h, y0, "Particle ESS reads healthy while commitment diversity collapses", 1.0, "normalized")]
    o.append(line(part, n, h, y0, 1.0, "part"))
    o.append(line(root, n, h, y0, 1.0, "rootm"))
    for i, s in enumerate(st):
        if "resample" in s.operators_fired:
            o.append(f'<line x1="{x(i,n):.1f}" y1="{y0}" x2="{x(i,n):.1f}" y2="{y0+h}" class="rs"/>')
    return f'<svg viewBox="0 0 {W} {y0+h+30}" class="c">' + "".join(o) + "</svg>"

# ---------- chart 3: population streamgraph with births ----------
def chart_stream(st):
    n = len(st); h, y0 = 230, 26
    lab, i2 = {}, 0
    for s in st:
        for p in s.particles:
            r = p.root_id or p.lineage_id
            if r not in lab:
                lab[r] = i2; i2 += 1
    order = list(lab)
    cum = [0.0] * n
    o = [f'<text x="{PL}" y="{y0-8}" class="pt">Belief mass per commitment &mdash; bands appear when new commitments are born</text>']
    for r in order:
        top, bot = [], []
        for i, s in enumerate(st):
            v = sum(p.weight or 0 for p in s.particles if (p.root_id or p.lineage_id) == r)
            xx = x(i, n); lo = y0 + h - cum[i] * h; hi = lo - v * h
            bot.append(f"{xx:.1f},{lo:.1f}"); top.append(f"{xx:.1f},{hi:.1f}")
            cum[i] += v
        hue = (lab[r] * 47) % 360
        o.append(f'<polygon points="{" ".join(top+list(reversed(bot)))}" fill="hsl({hue},58%,55%)" opacity=".88"/>')
    for i, s in enumerate(st):
        if s.operators_fired:
            tag = "".join(c[0].upper() for c in s.operators_fired)
            o.append(f'<line x1="{x(i,n):.1f}" y1="{y0}" x2="{x(i,n):.1f}" y2="{y0+h}" class="op"/>')
            o.append(f'<text x="{x(i,n):.1f}" y="{y0-1}" class="tk" text-anchor="middle">{tag}</text>')
    for i in range(0, n, max(1, n // 10)):
        o.append(f'<text x="{x(i,n):.1f}" y="{y0+h+15}" class="tk" text-anchor="middle">{i}</text>')
    return f'<svg viewBox="0 0 {W} {y0+h+28}" class="c">' + "".join(o) + "</svg>"

# ---------- chart 4: non-linearity bars ----------
def chart_bars(bsum, asum):
    keys = [("distinct_roots","commitments seen"), ("reversals_median_long_lived","weight reversals"),
            ("argmax_churn","lead changes"), ("split_count","splits"),
            ("merge_count","merges"), ("perturb_count","new commitments")]
    h, y0, bh = len(keys)*40+10, 26, 14
    o = [f'<text x="{PL}" y="{y0-8}" class="pt">Non-linear progression: strengthen, weaken, merge, split, revise</text>']
    mx = max(max(bsum.get(k,0), asum.get(k,0)) for k,_ in keys) or 1
    for r,(k,lbl) in enumerate(keys):
        yy = y0 + r*40
        b, aa = bsum.get(k,0), asum.get(k,0)
        o.append(f'<text x="{PL-8}" y="{yy+12}" class="tk" text-anchor="end">{lbl}</text>')
        o.append(f'<rect x="{PL}" y="{yy}" width="{max(1,(W-PL-PR)*b/mx):.1f}" height="{bh}" class="bbar"/>')
        o.append(f'<text x="{PL+max(1,(W-PL-PR)*b/mx)+6:.1f}" y="{yy+11}" class="tk">{b} before</text>')
        o.append(f'<rect x="{PL}" y="{yy+17}" width="{max(1,(W-PL-PR)*aa/mx):.1f}" height="{bh}" class="abar"/>')
        o.append(f'<text x="{PL+max(1,(W-PL-PR)*aa/mx)+6:.1f}" y="{yy+28}" class="tk">{aa} after</text>')
    return f'<svg viewBox="0 0 {W} {y0+h}" class="c">' + "".join(o) + "</svg>"

before = load("musing_out/p2clean*.steps.jsonl")
after  = load("musing_out/p3e_n2_*.steps.jsonl")
demo   = load("musing_out/demo*.steps.jsonl")
demo_st = demo[0][1] if demo else after[0][1]

def agg(runs):
    out = {}
    for _, st in runs:
        s = t.nonlinearity_summary(st)
        for k, v in s.items():
            if isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + v
    n = max(1, len(runs))
    return {k: round(v / n) for k, v in out.items()}

bsum, asum = agg(before), agg(after)
brt = [s[1][-1].distinct_roots for s in before]
art = [s[1][-1].distinct_roots for s in after]

CSS = """
:root{--bg:#fbfbfd;--fg:#1b1d22;--mut:#666e7a;--line:#dde0e6;--card:#fff;
--before:#c2410c;--after:#0f766e}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#131519;--fg:#e9ebef;
--mut:#98a0ad;--line:#2b303a;--card:#1b1e24;--before:#fb923c;--after:#2dd4bf}}
*{box-sizing:border-box}body{margin:0;padding:0 18px 60px;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:940px;margin:0 auto}h1{font-size:26px;margin:30px 0 6px}
h2{font-size:18px;margin:34px 0 8px;padding-top:14px;border-top:1px solid var(--line)}
p{color:var(--mut);margin:8px 0}p.lead{color:var(--fg)}
.c{width:100%;height:auto;background:var(--card);border:1px solid var(--line);
border-radius:10px;margin:12px 0;display:block}
.ax{stroke:var(--line);stroke-width:1}.tk{fill:var(--mut);font-size:10px}
.pt{fill:var(--fg);font-size:12px;font-weight:600}
.before{stroke:var(--before);opacity:.85}.after{stroke:var(--after);opacity:.85}
.part{stroke:#6366f1;stroke-width:2.2}.rootm{stroke:#e11d48;stroke-width:2.2}
.rs{stroke:var(--line)}.op{stroke:var(--fg);opacity:.25;stroke-dasharray:3 3}
.bbar{fill:var(--before);opacity:.75}.abar{fill:var(--after);opacity:.85}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.k{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px 14px;flex:1;min-width:150px}
.k .n{font-size:22px;font-weight:650}.k .l{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.k .d{font-size:12px;color:var(--mut)}
.key{font-size:12px;color:var(--mut);margin:4px 0 0}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px;margin-right:4px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
.caveat{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--before);
border-radius:8px;padding:12px 16px;margin:16px 0}
code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:12px}
"""

body = f"""
<div class="wrap">
<h1>Particle filter: before &amp; after</h1>
<p class="lead">What changed in the hypothesis population, measured on narrative dialogue.
Orange is the released code; teal is after the changes.</p>

<div class="cards">
 <div class="k"><div class="l">Commitments at run end</div>
   <div class="n" style="color:var(--before)">{min(brt)}&ndash;{max(brt)}</div>
   <div class="d">&rarr; <b style="color:var(--after)">{min(art)}&ndash;{max(art)}</b> after</div></div>
 <div class="k"><div class="l">Commitments born mid-run</div>
   <div class="n" style="color:var(--before)">0</div>
   <div class="d">&rarr; <b style="color:var(--after)">8</b> in the demo run</div></div>
 <div class="k"><div class="l">Parse failures / run</div>
   <div class="n" style="color:var(--before)">7</div>
   <div class="d">&rarr; <b style="color:var(--after)">0</b></div></div>
 <div class="k"><div class="l">Rank stability (&tau;)</div>
   <div class="n" style="color:var(--before)">0.687</div>
   <div class="d">&rarr; <b style="color:var(--after)">0.926</b></div></div>
</div>

<h2>1. Commitments survive instead of decaying to one</h2>
<p>Each line is one run. Before, ancestral commitments could only die &mdash; monotone decay to a
median of 3, worst case <b>a single ancestor in 28 steps</b>. After, they hold near 8 because
repair can found new ones.</p>
{chart_commitments(before, after)}
<p class="key"><span class="sw" style="background:var(--before)"></span>before ({len(before)} runs)
<span class="sw" style="background:var(--after);margin-left:14px"></span>after ({len(after)} runs)</p>

<h2>2. The standard metric could not see the collapse</h2>
<p>Particle ESS counts five copies of one winner as five independent particles. At a resample it
<i>jumps</i> &mdash; reading as recovery &mdash; while diversity measured over ancestral
commitments <i>falls</i>. Roughly 71&ndash;75% of the apparent recovery is an artifact.</p>
{chart_blind(before[0][1])}
<p class="key"><span class="sw" style="background:#6366f1"></span>particle ESS (what the method tracked)
<span class="sw" style="background:#e11d48;margin-left:14px"></span>commitment diversity (actual)</p>

<h2>3. What non-linear progression looks like</h2>
<p>One 34-step run with the full stack. Each band is an ancestral commitment; band height is its
share of belief mass. Dashed lines mark operators (R resample, P perturb, S split, M merge).
New bands appearing mid-run are commitments that <b>did not exist at step 0</b>.</p>
{chart_stream(demo_st)}
<p>At step 0 every commitment concerns an appeal &mdash; <i>&ldquo;to enable the appeal&rdquo;</i>,
<i>&ldquo;to appear generous&rdquo;</i>. By step 31 the dominant ones are <i>&ldquo;to maintain a
public image of unwavering integrity&rdquo;</i> and <i>&ldquo;to secure lasting personal
recognition&rdquo;</i>, born at step 31 &mdash; tracking the subject&rsquo;s shift from managing an
appeal to securing a confirmation.</p>

<h2>4. Non-linearity, counted</h2>
<p>Strengthen, weaken, merge, split, revise. Before, every one of these was zero by construction.</p>
{chart_bars(bsum, asum)}

<h2>5. What did not change</h2>
<div class="caveat">
<p class="lead" style="margin-top:0"><b>Belief-tracking quality is unchanged.</b></p>
<table>
<tr><th>metric</th><th>ground truth</th><th>before</th><th>after</th></tr>
<tr><td>false attribution</td><td>content the target was absent for</td><td>0</td><td>0</td></tr>
<tr><td>false belief</td><td>disguised speaker identity</td><td>10/10</td><td>10/10</td></tr>
<tr><td>goal change</td><td>subject&rsquo;s goal shifts, both sides quotable</td><td>10/10</td><td>10/10</td></tr>
</table>
<p>All three were already at ceiling for the baseline. The third was built as a falsification test:
one route minted <b>0</b> new commitments, the other <b>12&ndash;14</b>, and both scored identically.
The anchor pins a particle&rsquo;s commitment, but the belief <i>text</i> propagates freely with
context, and the summary is built from texts.</p>
<p style="margin-bottom:0">On these corpora, the work of belief tracking happens <b>upstream of the
particle population</b> &mdash; in perception tracking and propagation. The filter&rsquo;s diversity
operates where these checks do not reach.</p>
</div>

<h2>Bottom line</h2>
<p class="lead">The system is correct where it was broken &mdash; several components were inoperative
rather than badly tuned. Its internal dynamics are substantially better: weights accumulate,
commitments survive and are born, thresholds are derived rather than assumed. Whether those dynamics
improve the output is <b>unproven</b> on the tasks tested.</p>
</div>
"""

open("musing_out/before_after.html", "w", encoding="utf-8").write(
    f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
    f'<meta name="viewport" content="width=device-width,initial-scale=1">'
    f'<title>Particle filter: before &amp; after</title><style>{CSS}</style></head>'
    f'<body>{body}</body></html>')
print(f"before: {len(before)} runs, final commitments {brt}")
print(f"after : {len(after)} runs, final commitments {art}")
print(f"demo  : {len(demo_st)} steps")

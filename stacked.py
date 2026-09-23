"""Two runs, stacked and aligned, for comparing how one scene is modelled for
two different target agents.

Shares the x axis and the weight axis across both panels so the two are read
against each other rather than side by side from memory. Reuses weight_chart's
build(), so forks, drops and removals mean the same thing in both.
"""
import argparse, html, json, re, sys
import musing_layout as ml
import trace_log as t
import weight_chart as wc


def scene_turns(corpus, set_ids):
    """Turns for one set, or several stitched in order (a whole episode)."""
    d = json.load(open(f"data/musing/{corpus}_dialogue.json"))
    sets = d if isinstance(d, list) else d.get("sets", d)
    by = {st.get("set_id"): st for st in sets}
    out = []
    for sid in [x.strip() for x in str(set_ids).split(",") if x.strip()]:
        if sid not in by:
            sys.exit(f"set {sid} not found")
        out.extend(by[sid]["turns"])
    return out


def _turn_key(text):
    """Normalise an utterance into a match key.

    Slack turns carry <@U...> mentions that the logged prompt drops, and the
    logged <response> sometimes has no "Speaker: " prefix at all -- in which
    case partition(":") put the whole message in the speaker slot and left the
    text empty, so every one of those steps failed to match and fell through to
    the monotonic filler.
    """
    t = re.sub(r"<@[A-Z0-9]+>", " ", text or "")
    t = re.sub(r"&[a-z]+;", " ", t)
    t = re.sub(r"[^0-9a-z]+", " ", t.lower()).strip()
    return t[:40]


def scene_positions(steps, turns):
    """Map each filter step to its position in the underlying scene.

    Each run steps on ITS OWN target's utterances, so step 10 of one run and
    step 10 of another are different moments in the conversation -- plotting
    both against a shared "turn" axis implies an alignment that does not exist.
    Anchoring on the scene turn each step actually scored makes the two panels
    comparable in real conversation time, which is the only way a side-by-side
    read of two agents means anything.
    """
    idx, by_text, by_norm = {}, {}, {}
    for k, tn in enumerate(turns):
        txt = re.sub(r"\s+", " ", (tn.get("text") or "")).strip()[:40]
        idx.setdefault((tn["speaker"], txt), k)
        by_norm.setdefault(_turn_key(tn.get("text")), k)
        # Second index on the utterance alone. --merge-speakers rewrites the
        # speaker to the coalition name ("Customer"), so the (speaker, text) key
        # can never match a transcript turn attributed to a real person, and
        # EVERY step falls through to the monotonic filler -- measured 8/17 on
        # sz_Product, which crushed 17 steps into the left 13% of a 66-turn axis.
        # The utterance is the thing that actually identifies the turn.
        by_text.setdefault(txt, k)
    pos = []
    for s in steps:
        p = getattr(s, "likelihood_prompt", None)
        m = re.findall(r"<response>\s*(.+?)\s*</response>", p, re.S) if p else []
        k = None
        if m:
            resp = m[-1].strip()
            sp, sep, txt = resp.partition(":")
            # A speaker prefix is a short bare name. Anything longer is a message
            # that simply had no prefix, and the whole response is the utterance.
            if not sep or len(sp) > 30 or sp.count(" ") > 2:
                sp, txt = "", resp
            key = re.sub(r"\s+", " ", txt).strip()[:40]
            k = idx.get((sp.strip(), key))
            if k is None:
                k = by_text.get(key)
            if k is None:
                k = by_norm.get(_turn_key(txt))
        pos.append(k)
    # fill gaps monotonically so an unmatched step still lands between its
    # neighbours rather than collapsing to 0
    last = 0
    for i, v in enumerate(pos):
        if v is None:
            nxt = next((x for x in pos[i+1:] if x is not None), last + 1)
            pos[i] = min(max(last, 0) + 0.5, nxt)
        last = pos[i]
    return pos

# ---------------------------------------------------------------- key events
#
# The weight chart shows every line at once, which is the right picture for
# "did anything move" and the wrong one for "what moved, and on what". This
# section answers the second question by picking the turns where the top of
# the population actually changed hands and printing the top 3 commitments on
# either side of that turn.
#
# Commitments are aggregated by ROOT, not by particle. Resampling duplicates a
# strong particle, so a per-particle top 3 can show one commitment three times
# and call it a diverse population.

EV_CSS = """
.evs{display:grid;grid-template-columns:1fr;gap:18px;margin:14px 0 6px}
@media(min-width:900px){.evs{grid-template-columns:1fr 1fr;gap:18px 26px}}
.evcol h3{font-size:14px;margin:0 0 2px;font-weight:650}
.evcol .cap{color:var(--mut);font-size:12.5px;margin:0 0 10px;line-height:1.5}
.ev{background:var(--card);border:1px solid var(--line);border-radius:10px;
    padding:11px 13px 12px;margin-bottom:10px}
.ev .hd{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;
    font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}
.ev .hd .ops{color:var(--fg);opacity:.62}
.ev .q{margin:6px 0 10px;color:var(--fg);font-size:13.5px;line-height:1.45}
.ev .q i{color:var(--mut);font-style:normal}
.ba{display:grid;grid-template-columns:1fr;gap:9px}
@media(min-width:520px){.ba{grid-template-columns:1fr 1fr;gap:14px}}
.ba .k{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
    color:var(--mut);margin-bottom:5px}
.hy{margin-bottom:6px;font-size:12.5px;line-height:1.3}
.hy .t{display:flex;gap:6px;align-items:baseline}
.hy .n{color:var(--mut);font-variant-numeric:tabular-nums;font-size:11.5px;
    margin-left:auto;padding-left:6px}
.hy .bars{position:relative;height:3px;margin-top:4px}
.hy .bar{position:absolute;top:0;height:3px;border-radius:2px;
    background:currentColor;opacity:.5;min-width:1px}
.hy .bar.du{background:hsl(150,55%,42%);opacity:.9}
.hy .bar.dd{background:hsl(8,62%,54%);opacity:.9}
.dl{font-variant-numeric:tabular-nums;font-size:11px;font-weight:650;
    margin-left:5px;white-space:nowrap}
.dl.up{color:hsl(150,48%,36%)}
.dl.dn{color:hsl(8,58%,48%)}
.dl.flat{color:var(--mut);font-weight:400}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]) .dl.up{color:hsl(150,48%,58%)}
  :root:not([data-theme=light]) .dl.dn{color:hsl(8,66%,66%)}}
.hy.gone{opacity:.42}
.hy.gone .t span:first-child{text-decoration:line-through}
.mk{font-size:10px;font-weight:700;letter-spacing:.04em;border-radius:3px;
    padding:0 4px;border:1px solid currentColor;opacity:.85;white-space:nowrap}
"""


def root_mass(s):
    """Belief mass per commitment, with the anchor of its heaviest particle."""
    agg = {}
    for p in s.particles:
        r = p.root_id or p.lineage_id
        e = agg.setdefault(r, {"w": 0.0, "anchor": p.anchor or "", "best": -1.0})
        e["w"] += p.weight or 0.0
        if (p.weight or 0.0) > e["best"]:
            e["best"], e["anchor"] = p.weight or 0.0, p.anchor or ""
    return agg


def _top(agg, k=3):
    return sorted(agg.items(), key=lambda kv: -kv[1]["w"])[:k]


def key_events(st, xs=None, limit=6):
    """Turns ranked by how much the top 3 changed across them.

    Score is top-3 set churn plus total weight moved, with a bonus when the
    lead changes hands -- a new leader is the event this page exists to show.

    Uniform steps are EXCLUDED. Resampling flattens every weight to 1/n, so
    the top 3 straight after one is an arbitrary slice of a tie and scores as
    maximum churn while meaning nothing. Requiring the post-state leader to
    hold 1.6x its fair share drops those and keeps the real hand-overs.
    """
    aggs = [root_mass(s) for s in st]
    seen_top = set()
    was_top = []                       # roots that had reached the top 3 by i-1
    for a in aggs:
        was_top.append(set(seen_top))
        seen_top |= {r for r, _ in _top(a)}
    cand = []
    for i in range(1, len(st)):
        a, b = aggs[i - 1], aggs[i]
        ta, tb = _top(a), _top(b)
        if not ta or not tb:
            continue
        if tb[0][1]["w"] < 1.6 / max(1, len(b)):
            continue
        sa, sb = {r for r, _ in ta}, {r for r, _ in tb}
        churn = 1 - len(sa & sb) / len(sa | sb)
        drift = sum(abs(b.get(r, {}).get("w", 0.0) - a.get(r, {}).get("w", 0.0))
                    for r in sa | sb)
        lead = ta[0][0] != tb[0][0]
        cand.append((churn * 1.6 + drift + (0.5 if lead else 0), i))
    cand.sort(reverse=True)
    keep = sorted(i for _, i in cand[:limit])

    out = []
    for i in keep:
        a, b = aggs[i - 1], aggs[i]
        ta, tb = _top(a), _top(b)
        sa, sb = {r for r, _ in ta}, {r for r, _ in tb}
        ra = {r: k for k, (r, _) in enumerate(ta)}
        rows_b, rows_a = [], []
        for r, d in ta:
            # where it ended up, so a commitment that leaves the top 3 still
            # says whether it was refuted or merely overtaken
            rows_b.append({"anchor": d["anchor"], "w": d["w"],
                           "other": b.get(r, {}).get("w", 0.0),
                           "d": b.get(r, {}).get("w", 0.0) - d["w"],
                           "gone": r not in sb, "mark": "" if r in sb else "out"})
        for k, (r, d) in enumerate(tb):
            if r in ra:
                mark = "same" if ra[r] == k else ("up" if k < ra[r] else "down")
            elif r in was_top[i - 1]:
                mark = "back"           # held the top 3 earlier, lost it, returned
            elif r in a:
                mark = "rise"           # alive but below the top 3 until now
            else:
                mark = "new"            # not in the population at all last turn
            # delta against the SAME ROOT last turn. A root absent from the
            # previous population had 0 mass, so its whole weight is the
            # delta -- which is the point of showing a mint as +0.33.
            rows_a.append({"anchor": d["anchor"], "w": d["w"],
                           "other": a.get(r, {}).get("w", 0.0),
                           "d": d["w"] - a.get(r, {}).get("w", 0.0),
                           "gone": False, "mark": mark})
        tags = list(st[i].operators_fired or [])
        if st[i].surprise:
            tags.insert(0, "surprise")
        if st[i].minted_roots:
            tags.append("mint")
        out.append({
            "step": i,
            "turn": (int(round(xs[i])) if xs and xs[i] is not None else None),
            "action": st[i].scored_action or "",
            "tags": tags,
            "before": rows_b,
            "after": rows_a,
            "promoted": bool(sb - sa),
            "revived": any(r["mark"] == "back" for r in rows_a),
        })
    return out


_MARKS = {"up": ("&uarr;", "rose"), "down": ("&darr;", "fell"),
          "same": ("&ndash;", "held"), "new": ("+", "new"),
          "rise": ("&uarr;", "entered"), "back": ("&#8634;", "back"),
          "out": ("&times;", "out")}


def _delta(d):
    """Signed change in belief mass across the turn.

    Printed for every row, including the ones that did not move: a commitment
    that held the lead THROUGH a reversal is a different fact from one that
    was never tested, and only the delta separates them.
    """
    if abs(d) < 0.005:
        return '<span class="dl flat">&plusmn;0.00</span>'
    cls = "up" if d > 0 else "dn"
    return f'<span class="dl {cls}">{"+" if d > 0 else "&minus;"}{abs(d):.2f}</span>'


def _rows_html(rows, scale, hue):
    o = []
    for r in rows:
        sym, word = _MARKS.get(r["mark"], ("", ""))
        tag = (f'<span class="mk" style="color:hsl({hue},52%,46%)">{sym}&nbsp;{word}</span>'
               if word else "")
        # the delta bar is drawn from the smaller of the two weights, so the
        # tinted segment IS the change and its direction is readable without
        # reading the number
        lo, hi = min(r["w"], r["other"]), max(r["w"], r["other"])
        o.append(
            f'<div class="hy{" gone" if r["gone"] else ""}" style="color:hsl({hue},58%,50%)">'
            f'<div class="t"><span style="color:var(--fg)">'
            f'{html.escape(r["anchor"]) if r["anchor"] else "&mdash;"}</span>'
            f'{tag}<span class="n">{r["w"]:.2f} {_delta(r["d"])}</span></div>'
            f'<div class="bars">'
            f'<div class="bar" style="width:{100 * lo / scale:.1f}%"></div>'
            f'<div class="bar d{"u" if r["d"] > 0 else "d"}" '
            f'style="left:{100 * lo / scale:.1f}%;width:{100 * (hi - lo) / scale:.1f}%"></div>'
            f'</div></div>')
    return "".join(o)


def events_html(st, label, xs=None, limit=6, hue=210):
    evs = key_events(st, xs, limit)
    if not evs:
        return ""
    o = [f'<div class="evcol"><h3>{html.escape(label)}</h3>']
    promoted = sum(1 for e in evs if e["promoted"])
    revived = sum(1 for e in evs if e["revived"])
    o.append(f'<p class="cap">{len(evs)} turns where the top of the population changed hands. '
             f'{promoted} of them put a commitment in the top 3 that was not there the turn '
             f'before'
             + (f', and {revived} brought back one that had already been displaced earlier in '
                f'the run &mdash; the population does not converge, it doubles back.'
                if revived else '. ')
             + '</p>')
    for e in evs:
        scale = max([r["w"] for r in e["before"] + e["after"]] + [0.01])
        # without --align-set the turn axis IS the step index, so printing both
        # just repeats the same number
        where = f'turn {e["turn"]}' if e["turn"] is not None else f'step {e["step"]}'
        step = f'<span>step {e["step"]}</span>' if e["turn"] is not None else ""
        tags = (f'<span class="ops">'
                f'{" &middot; ".join(html.escape(x) for x in e["tags"])}</span>'
                if e["tags"] else "")
        act = html.escape(e["action"])
        sp, _, said = act.partition(":")
        o.append(
            f'<div class="ev"><div class="hd"><span>{where}</span>{step}{tags}</div>'
            f'<p class="q"><i>{sp}:</i> {said.strip()}</p>'
            f'<div class="ba">'
            f'<div><div class="k">top 3 before this turn</div>'
            f'{_rows_html(e["before"], scale, hue)}</div>'
            f'<div><div class="k">top 3 after</div>'
            f'{_rows_html(e["after"], scale, hue)}</div>'
            f'</div></div>')
    o.append("</div>")
    return "".join(o)


W, PL, PR, H, GAP, TOP = 940, 62, 150, 250, 58, 34


def panel(st, y0, ymax, n, title, note, xs=None):
    series, parent_of, born, detail, removed = wc.build(st)
    lids = sorted(series, key=lambda k: (born[k], k))
    lab = {l: (chr(ord('A') + i) if i < 26 else f"z{i}") for i, l in enumerate(lids)}
    hue = {l: (i * 53) % 360 for i, l in enumerate(lids)}
    xs = xs or list(range(len(st)))
    X = lambda i: PL + (W - PL - PR) * (xs[i] / max(1, n - 1))
    # Clamp to the panel. With ymax at the global maximum this never binds, but
    # evolution.py sets a percentile ceiling so one run that reaches 0.89 does not
    # flatten four that live under 0.45 -- and an unclamped point above the ceiling
    # would be drawn ABOVE y0, on top of the panel stacked above it. Hover still
    # reports the true weight, and clipped points get a caret below.
    Y = lambda v: y0 + H - (min(v, ymax) / ymax) * H
    o = [f'<text x="{PL}" y="{y0-12}" class="ti">{html.escape(title)}</text>',
         f'<text x="{W-PR}" y="{y0-12}" class="tk" text-anchor="end">{html.escape(note)}</text>']
    for k in range(5):
        v = ymax * k / 4
        o.append(f'<line x1="{PL}" y1="{Y(v):.1f}" x2="{W-PR}" y2="{Y(v):.1f}" class="g"/>')
        o.append(f'<text x="{PL-8}" y="{Y(v)+4:.1f}" class="tk" text-anchor="end">{v:.2f}</text>')
    for g in range(0, n, max(1, n // 12)):
        gx = PL + (W - PL - PR) * (g / max(1, n - 1))
        o.append(f'<text x="{gx:.1f}" y="{y0+H+15}" class="tk" text-anchor="middle">{g}</text>')
    for i, s in enumerate(st):
        if s.operators_fired:
            tag = "".join(c[0].upper() for c in s.operators_fired)
            o.append(f'<line x1="{X(i):.1f}" y1="{y0}" x2="{X(i):.1f}" y2="{y0+H}" class="op"/>')
            o.append(f'<text x="{X(i):.1f}" y="{y0-2}" class="tk" text-anchor="middle">{tag}</text>')
    forks = set()
    for l in lids:
        par, b = parent_of.get(l), born[l]
        if par and par in series and b > 0 and (b - 1) in series[par]:
            x0, yy = X(b - 1), Y(series[par][b - 1])
            o.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{X(b):.1f}" y2="{Y(series[l][b]):.1f}" '
                     f'stroke="hsl({hue[l]},60%,58%)" stroke-width="1.6" stroke-dasharray="4 2.5" opacity=".9"/>')
            forks.add((x0, yy))
    for fx, fy in forks:
        o.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="3.2" fill="var(--card)" '
                 f'stroke="var(--fg)" stroke-width="1.4" opacity=".9"/>')
    for l in lids:
        pts = [f"{X(i):.1f},{Y(v):.1f}" for i, v in sorted(series[l].items())]
        if len(pts) < 2:
            i, v = next(iter(series[l].items()))
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="hsl({hue[l]},58%,52%)"/>')
        else:
            o.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                     f'stroke="hsl({hue[l]},58%,52%)" stroke-width="2" opacity=".92"/>')
        for i, v in sorted(series[l].items()):
            if v > ymax:
                cx, cy = X(i), y0 + 1
                o.append(f'<path d="M{cx-4:.1f},{cy+5} L{cx:.1f},{cy} L{cx+4:.1f},{cy+5}" '
                         f'fill="none" stroke="hsl({hue[l]},58%,52%)" stroke-width="1.6"/>')
            anc, txt = detail[l][i]
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5" fill="transparent" '
                     f'class="hit" data-lab="{html.escape(lab[l])}" data-turn="{i}" '
                     f'data-w="{v:.3f}" data-anchor="{html.escape(anc or "")}" '
                     f'data-text="{html.escape((txt or "")[:600])}"/>')
        li, lv = max(series[l].items())
        if l in removed:
            cx, cy, r = X(li), Y(lv), 3.4
            for a, b2 in (((cx-r, cy-r), (cx+r, cy+r)), ((cx-r, cy+r), (cx+r, cy-r))):
                o.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b2[0]:.1f}" y2="{b2[1]:.1f}" '
                         f'stroke="hsl({hue[l]},58%,52%)" stroke-width="1.6" opacity=".95"/>')
        o.append(f'<text x="{X(li)+6:.1f}" y="{Y(lv)+4:.1f}" class="lb" '
                 f'fill="hsl({hue[l]},58%,42%)">{lab[l]}</text>')
    for i, s in enumerate(st):
        ps = [p for p in s.particles if p.weight is not None]
        if ps:
            top = max(ps, key=lambda p: p.weight)
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(top.weight):.1f}" r="2.6" fill="none" '
                     f'stroke="var(--fg)" stroke-width="1.3" opacity=".8"/>')
    return "".join(o), len(lids)


def load(run):
    fs = ml.find(f"{run}*.steps.jsonl")
    if not fs:
        sys.exit(f"no steps for {run}")
    return t.read_steps(fs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("top"); ap.add_argument("bottom")
    ap.add_argument("--top-label", default=None); ap.add_argument("--bottom-label", default=None)
    ap.add_argument("--scene", default="")
    ap.add_argument("--align-set", default=None,
                    help="set_id, or comma-separated set_ids (a whole episode), to align "
                         "both panels on real scene turns")
    ap.add_argument("--corpus", default="oppenheimer")
    ap.add_argument("--events", type=int, default=6,
                    help="how many key turns to expand in the events section (0 = omit)")
    ap.add_argument("--intro", default=None,
                    help="HTML fragment inserted under the lede. Narrative about a "
                         "particular pair of runs does not belong in this generic "
                         "template, but regenerating must not silently drop it either.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A, B = load(a.top), load(a.bottom)
    xa = xb = None
    if a.align_set:
        turns = scene_turns(a.corpus, a.align_set)
        xa, xb = scene_positions(A, turns), scene_positions(B, turns)
        # Trim the axis to where the data actually is. The stitched span is as
        # long as every set listed, but the two targets may fall silent well
        # before the end of it -- Deskins and Rodriguez both stop around turn 28
        # of a 118-turn span, so three quarters of the chart was blank and the
        # part that carried the argument was squeezed into the left edge.
        # Only ever trims: a target speaking to the end leaves the axis alone.
        last = max([p for xs in (xa, xb) for p in xs] or [0])
        n = min(len(turns), int(last) + 2)
        if n < len(turns):
            print(f"   axis trimmed to turn {n - 1} of {len(turns)} "
                  f"(neither target speaks after that)")
        axis = "turn in the scene"
    else:
        n = max(len(A), len(B))
        axis = "step (each agent's own turns &mdash; NOT aligned between panels)"
    ymax = max(0.35, 1.08 * max([p.weight or 0 for s in A + B for p in s.particles] or [1]))
    y1, y2 = TOP, TOP + H + GAP
    pa, na = panel(A, y1, ymax, n, a.top_label or a.top, f"{len(A)} turns, {0} lines", xa)
    pb, nb = panel(B, y2, ymax, n, a.bottom_label or a.bottom, f"{len(B)} turns", xb)
    pa = pa.replace(f"{len(A)} turns, 0 lines", f"{len(A)} turns, {na} hypotheses")
    pb = pb.replace(f"{len(B)} turns", f"{len(B)} turns, {nb} hypotheses")
    src = open("weight_chart.py").read()
    css = src.split('css = """')[1].split('"""')[0]
    # JS is a local inside weight_chart.main(), so it cannot be imported --
    # lift it from source the same way as the CSS, or the tooltips go missing
    # silently and the hover text the chart advertises does nothing.
    js = src.split('JS = """')[1].split('"""')[0]
    css += "\n.ti{fill:var(--fg);font-size:13px;font-weight:650}"
    css += EV_CSS
    css += ("\n.evh{font-size:19px;margin:34px 0 4px;font-weight:650}"
            "\n.evlede{margin:0 0 4px;max-width:78ch}"
            "\n.evkey{margin:8px 0 2px;max-width:78ch;font-size:12.5px}"
            "\n.evkey b{color:var(--fg);font-weight:650}")
    intro = open(a.intro, encoding="utf-8").read() if a.intro else ""
    evs = ""
    if a.events:
        ea = events_html(A, a.top_label or a.top, xa, a.events, hue=210)
        eb = events_html(B, a.bottom_label or a.bottom, xb, a.events, hue=8)
        if ea or eb:
            evs = (
                f'<h2 class="evh">Where the belief actually turned</h2>'
                f'<p class="evlede">The chart above shows every hypothesis at once, which answers '
                f'&ldquo;did anything move&rdquo; but not &ldquo;on what&rdquo;. These are the turns '
                f'where the top of the population changed hands &mdash; ranked by how much the top 3 '
                f'changed across them, not picked by hand. Each card gives the line that was scored '
                f'and the top 3 commitments immediately before and after it, with the change in '
                f'belief mass for each. Read down the pairs and the evolution is plainly not '
                f'monotone: commitments are overtaken, fall out of the top 3, and come back.</p>'
                f'<p class="evkey">'
                f'<b>&uarr; rose</b> / <b>&darr; fell</b> moved within the top 3 &middot; '
                f'<b>&uarr; entered</b> was already in the population, below the top 3 &middot; '
                f'<b>+ new</b> did not exist last turn &mdash; a split child or a mint &middot; '
                f'<b>&#8634; back</b> held the top 3 earlier in the run, lost it, and returned '
                f'&middot; <b>&times; out</b> left the top 3 on this turn. The signed number is '
                f'the change in that commitment&rsquo;s belief mass across the turn, and the '
                f'tinted segment of each bar is that change.</p>'
                f'<div class="evs">{ea}{eb}</div>')
    a.out = a.out or ml.report_path("stacked.html")
    open(a.out, "w", encoding="utf-8").write(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Two minds, one scene</title><style>{css}</style></head><body><div class="wrap">'
        f'<h1>Two minds, one scene</h1>'
        f'<p>The same conversation{(" (" + html.escape(a.scene) + ")") if a.scene else ""}, '
        f'traced twice &mdash; once for each speaker. Both panels share the same turn axis and '
        f'the same weight scale, so they can be read against each other. '
        f'Horizontal axis: {axis}.</p>'
        f'{intro}'
        f'{wc.LEGEND}'
        f'<svg viewBox="0 0 {W} {y2+H+34}">{pa}{pb}</svg>'
        f'<div id="tip"></div><script>{js}</script>'
        f'{evs}'
        f'</div></body></html>')
    print(f"wrote {a.out}  (top {len(A)} turns/{na} lines, bottom {len(B)} turns/{nb} lines)")


if __name__ == "__main__":
    main()

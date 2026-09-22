"""Find where two parties judge the same question by different standards.

The existing culture axis on this corpus is a TOPIC axis -- what a turn is
about. It scores the sharpest turn in the Wolf trace at exactly 0.0, because
topic is not what the dispute runs on. This detector scores the other thing:
the STANDARD OF EVIDENCE a party appeals to when it says a number is right.

The claim it tests is falsifiable and narrow. Two parties are misaligned in
this sense when, on the same turns of the same thread, their appeals separate
persistently -- not when they disagree about an answer, and not when they
happen to be talking about different topics. So the detector reports three
things and refuses to call it a detection unless all three hold:

  separation   the running per-party lean, and the gap between parties
  persistence  how many consecutive scored turns the gap holds its sign
  referent     both parties are on the same thread -- a gap between two
               people who are not talking to each other is not a misalignment

Run:
  detect_standard.py --corpus bloomfield --lexicon bloomfield_standard \
      --a Wolf --b Lyubovsky,Rovani
"""
import argparse
import collections
import json

from scene_player import LEXICONS, culture


def lean(text, lex):
    return culture(text, lex)


def from_traces(runs, lex):
    """Read the axis off the filter's own STANDARD field.

    The vocabulary route has to learn the axis from who says what, which makes
    it circular when the pair it is tested on is the pair it was fitted to --
    measured: a Wolf/McLafferty firing at the WRONG SIGN that vanished under a
    held-out split. A run that records STANDARD per particle needs no fitting:
    the filter states the settling condition, and the axis is scored on that
    sentence alone.
    """
    import stacked as st
    out = []
    for rid in runs:
        steps = st.load(rid)
        vals = []
        for rec in steps:
            tot = sum(p.weight or 0 for p in rec.particles) or 1.0
            v = [(p.weight or 0, getattr(p, "standard", None)) for p in rec.particles]
            v = [(w, sd) for w, sd in v if sd]
            if not v:
                continue
            # mass-weighted: the run is judged on what it is actually holding
            vals.append(sum(w * culture(sd, lex) for w, sd in v) / tot)
        out.append((rid, vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="bloomfield")
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--a", required=True, help="one party, comma separated")
    ap.add_argument("--b", required=True, help="the other party")
    ap.add_argument("--sets", default=None, help="restrict to these set ids")
    ap.add_argument("--min-turns", type=int, default=3,
                    help="a party must speak this often in a thread to count")
    ap.add_argument("--runs", default=None,
                    help="comma-separated run ids; score the filter's own STANDARD "
                         "field instead of the transcript vocabulary")
    ap.add_argument("--min-run", type=int, default=3,
                    help="consecutive same-sign gaps needed to report a stretch")
    a = ap.parse_args()

    lex = LEXICONS.get(a.lexicon or a.corpus)
    if not lex:
        raise SystemExit(f"no lexicon {a.lexicon or a.corpus!r}")
    A = set(a.a.split(","))
    B = set(a.b.split(","))
    d = json.load(open(f"data/musing/{a.corpus}_dialogue.json", encoding="utf-8"))
    if a.sets:
        keep = set(a.sets.split(","))
        d = [s for s in d if s["set_id"] in keep]

    print(f"axis: {lex[0][0]}  (-1)   <-->   (+1)  {lex[1][0]}")
    if a.runs:
        rows = from_traces(a.runs.split(","), lex)
        print()
        for rid, vals in rows:
            if not vals:
                print(f"  {rid:<16} no STANDARD recorded -- run predates the field")
                continue
            m = sum(vals) / len(vals)
            side = lex[1][0] if m > 0 else lex[0][0]
            print(f"  {rid:<16} {len(vals):>3} steps   mean {m:+.2f}   -> {side}")
        scored = [(r, v) for r, v in rows if v]
        if len(scored) >= 2:
            ms = [sum(v) / len(v) for _, v in scored]
            print(f"\n  spread across runs: {max(ms) - min(ms):+.2f}")
        return
    print(f"  {'/'.join(sorted(A))}  vs  {'/'.join(sorted(B))}\n")

    # Only threads where BOTH parties actually speak -- the shared-referent
    # condition. Without it the detector fires on two people who never met.
    rows, shared = [], 0
    for s in d:
        ts = s.get("turns", [])
        sa = [t for t in ts if t["speaker"] in A]
        sb = [t for t in ts if t["speaker"] in B]
        if len(sa) < a.min_turns or len(sb) < a.min_turns:
            continue
        shared += 1
        va = [lean(t.get("text") or "", lex) for t in sa]
        vb = [lean(t.get("text") or "", lex) for t in sb]
        na = [v for v in va if v]
        nb = [v for v in vb if v]
        if not na or not nb:
            continue
        ma, mb = sum(na) / len(na), sum(nb) / len(nb)
        rows.append((mb - ma, ma, mb, len(na), len(nb), s["set_id"]))

    print(f"{shared} threads where both parties speak >= {a.min_turns} times; "
          f"{len(rows)} of them score on this axis\n")
    if not rows:
        return
    rows.sort()
    print(f"{'thread':<18}{'A lean':>8}{'B lean':>8}{'gap':>8}   scored turns")
    for gap, ma, mb, ca, cb, sid in rows:
        print(f"{sid:<18}{ma:>8.2f}{mb:>8.2f}{gap:>8.2f}   {ca} / {cb}")

    gaps = [r[0] for r in rows]
    mean = sum(gaps) / len(gaps)
    same = sum(1 for g in gaps if (g > 0) == (mean > 0))
    print(f"\nmean gap {mean:+.2f}; {same}/{len(gaps)} threads agree on its sign")
    side = lex[1][0] if mean > 0 else lex[0][0]
    other = lex[0][0] if mean > 0 else lex[1][0]
    # A gap that flips sign thread to thread is two people varying, not two
    # standards. Consistency across threads is what separates the two.
    if same >= a.min_run and same / len(gaps) >= 0.7 and abs(mean) >= 0.15:
        print(f"DETECTED: {'/'.join(sorted(B))} appeals to {side}, "
              f"{'/'.join(sorted(A))} to {other}, consistently across threads.")
    else:
        print("not a detection: the gap does not hold its sign across threads.")


if __name__ == "__main__":
    main()

"""Are the commitments MOTIVES, or restatements of the thread?

Built after a measured miss. Prompt variant v4 drove the settlement-mode
classifier to 0% unclassified, which every metric in eval_motive_sep.py read as
success -- while the commitments themselves degraded into nine variants of
'Define "2-Purple" by <assay method>'. The classification rate could not see
that, because it scores the STANDARD field and the damage was in the ANCHOR.

Three measures, none of which needs an LLM:

  echo       share of the commitment's content words that appear in the turn it
             was scored on. High = lifting the line rather than explaining it.
  bound      share of commitments naming a specific object, number, document or
             metric from this thread. A motive should survive being moved to a
             different week; "Correct row assignment in data" does not. This is
             the property the proposer's own anti-restatement rule forbids --
             and that rule is gated behind --infer-motive, so it has been off.
  redundancy 1 - (distinct motive stems / commitments). Nine ways to say
             'define purple' is one motive and eight duplicates; the population
             looks diverse and is not.

Usage:
  restatement.py cov1s0_Wolf cov4s0_Wolf covms0_Wolf
"""
import argparse
import collections
import glob
import json
import re

WORD = re.compile(r"[a-z]{4,}")
STOP = set("that this with from they what would have been than then them their there "
           "here about which will your very also more most some such only just make made "
           "take need want know look into over when have".split())

# Objects, artefacts and quantities that belong to ONE THREAD.
#
# Calibrated wrong the first time, in a way worth recording: the original
# pattern matched the corpus's ordinary domain nouns (data, berry, colour,
# label, row, plant), which every motive at a crop-data company contains. It
# scored "Protect company's data reputation" as thread-bound -- a clause that
# transports to any week -- and so reported the run WITH the anti-restatement
# rule enabled as the worst of the three. The test is not whether a clause
# names the subject matter. It is whether the clause could only be uttered in
# this particular conversation: a quoted category, a specific assay, a named
# artefact, a version, a number.
BOUND = re.compile(
    r'"[^"]+"|\b\d|2-Purple|1-Green|\bv\d\b|'
    r'anthocyanin|HPLC|spectral reflectance|pigment chemistry|verasion|'
    r'\bdashboard|spreadsheet|\bcolumn\b|hand ?count|row assignment|plant spacing|'
    r"Lyubovsky|Rovani|Wolf|McLafferty", re.I)

# Leading verb + its object head, as a crude motive stem: "Define 2-Purple by
# pigment" and "Define 2-Purple by HPLC results" share one.
def stem(a):
    w = [x for x in re.findall(r"[A-Za-z][\w'-]*", a.lower()) if x not in STOP]
    return " ".join(w[:2]) if w else a.lower()


def bag(s):
    return {w for w in WORD.findall((s or "").lower()) if w not in STOP}


def measure(run):
    fs = glob.glob(f"musing_out/runs/*/{run}-*.steps.jsonl")
    if not fs:
        return None
    anchors, echo = [], []
    for line in open(fs[0], encoding="utf-8"):
        st = json.loads(line)
        act = bag(st.get("scored_action") or "")
        top = max(st.get("particles") or [{}], key=lambda p: p.get("weight") or 0)
        h = bag(top.get("anchor"))
        if h and act:
            echo.append(len(h & act) / len(h))
        for p in st.get("particles", []):
            a = p.get("anchor")
            if a and a not in anchors:
                anchors.append(a)
    if not anchors:
        return None
    stems = collections.Counter(stem(a) for a in anchors)
    return {
        "n": len(anchors),
        "echo": sum(echo) / len(echo) if echo else 0.0,
        "bound": sum(1 for a in anchors if BOUND.search(a)) / len(anchors),
        "redundancy": 1 - len(stems) / len(anchors),
        "worst": stems.most_common(1)[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--show", action="store_true", help="print the duplicated family")
    a = ap.parse_args()
    print(f"{'run':<20}{'n':>4}{'echo':>8}{'bound':>8}{'redund':>8}   most duplicated stem")
    for r in a.runs:
        m = measure(r)
        if not m:
            print(f"{r:<20}  (no steps)")
            continue
        print(f"{r:<20}{m['n']:>4}{m['echo']:>8.3f}{m['bound']:>8.2f}"
              f"{m['redundancy']:>8.2f}   {m['worst'][0]!r} x{m['worst'][1]}")
    print("\nlower is better on all three. bound > 0.5 means most 'motives' could only "
          "be said in this thread.")


if __name__ == "__main__":
    main()

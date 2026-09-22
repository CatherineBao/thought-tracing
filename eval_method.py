"""Per-method hygiene: does each method fail the way methods.py predicted?

Companion to audit_methods.py, which scores YIELD. This scores HYGIENE -- the
rate at which each method produces commitments the system has to throw away,
and the shape of what survives. Every number here is read off run output;
nothing calls an LLM.

The columns, and where each comes from:

  echo/bound/redund   restatement.measure(by_method=True) -- is the commitment
                      a reason to send the message, or the message read back?
  form-reject         StepRecord.rejected_commitments, stamped with the method.
                      The direct test of Method.form_risk. Collected since the
                      validator was added and only now persisted.
  gate-reject         ParticleRecord.perturb_accepted is False, keyed by the
                      step's perturb_method. The coherence gate's CONTRADICTS
                      rate -- devil is predicted to top this.
  mint-survive        of the roots a method minted, how many were still alive
                      five steps later.
  first-person        anchors beginning "I " -- role's predicted leak. NOT a
                      new rejection reason in valid_commitment: every rejection
                      costs a particle, and the anchor site is the source term.

The point of printing Method.expected_failure beside the measured row is that
the registry is a PRE-REGISTRATION. A method whose measured failure matches its
declared one is understood; one that fails a different way is the finding.

  python eval_method.py --runs me_katara
  python eval_method.py --runs me_k_s1,me_k_s2 --launch --methods anomaly,silence
"""

import argparse
import collections
import glob
import json
import subprocess

from methods import METHODS
import restatement
from audit_methods import steps_for

SURVIVE_WINDOW = 5


def launch(run_id, corpus, target, set_ids, methods, seed, model, extra=None):
    """One run, with the flag block eval_motive_sep uses, plus --methods.

    Deliberately the same block: a method sweep that also changed the operator
    set would confound the two, and the operators here are the ones every
    earlier measurement was made under.
    """
    cmd = [".venv/bin/python", "run_musing.py", "--corpus", corpus,
           "--target", target, "--set-ids", set_ids,
           "--n-hypotheses", "8", "--tracing-model", model,
           "--goal-seeding", "--extract-anchors", "--use-anchor",
           "--anchored-perturbation", "--enable-split", "--enable-expiry",
           "--seed", str(seed), "--methods", methods,
           "--max-chars", "99999", "--run-id", run_id]
    if extra:
        cmd += extra
    print(f"  {run_id} ...", flush=True)
    with open(f"musing_out/{run_id}.log", "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)


def hygiene(run):
    """Form rejections, gate rejections and mint survival, keyed by method."""
    loaded = steps_for(run)
    if not loaded:
        return None
    steps = loaded[0][1]
    form = collections.Counter()         # method -> candidates dropped on form
    proposed = collections.Counter()     # method -> candidates that parsed
    gate_rej = collections.Counter()
    gate_all = collections.Counter()
    minted = collections.defaultdict(list)   # method -> (root_id, step_idx)
    first_person = collections.Counter()
    anchors_seen = collections.defaultdict(set)

    for i, st in enumerate(steps):
        for r in (st.get("rejected_commitments") or []):
            form[r.get("method") or "(none)"] += 1
        pm = st.get("perturb_method") or "(none)"
        for p in st.get("particles", []):
            if p.get("perturbed"):
                acc = p.get("perturb_accepted")
                if acc is not None:          # None = unparsed, not a verdict
                    gate_all[pm] += 1
                    if acc is False:
                        gate_rej[pm] += 1
            a = (p.get("anchor") or "").strip()
            m = p.get("method") or "(none)"
            if a and a not in anchors_seen[m]:
                anchors_seen[m].add(a)
                proposed[m] += 1
                # role reasons from inside the seat and is meant to step back
                # out; a clause that did not is the leak, counted not rejected
                if a.startswith("I ") or a.lower().startswith("i'm "):
                    first_person[m] += 1
        for rid in (st.get("minted_roots") or []):
            minted[pm].append((rid, i))

    alive_at = []
    for st in steps:
        alive_at.append({p.get("root_id") for p in st.get("particles", [])})

    survive = {}
    for m, births in minted.items():
        checked = [(rid, j) for rid, j in births if j + SURVIVE_WINDOW < len(steps)]
        if not checked:
            continue
        lived = sum(1 for rid, j in checked if rid in alive_at[j + SURVIVE_WINDOW])
        survive[m] = (lived, len(checked))

    return {
        "form": form, "proposed": proposed,
        "gate_rej": gate_rej, "gate_all": gate_all,
        "survive": survive, "first_person": first_person,
        "steps": len(steps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run-id prefixes")
    ap.add_argument("--launch", action="store_true", help="run them first")
    ap.add_argument("--methods", default=None, help="--methods value, with --launch")
    ap.add_argument("--corpus", default="atla")
    ap.add_argument("--target", default="Katara")
    ap.add_argument("--set-ids", default="atla-0604,atla-0605,atla-0606,atla-0607")
    ap.add_argument("--model", default="gemini-2.5-flash")
    a = ap.parse_args()

    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    if a.launch:
        if not a.methods:
            raise SystemExit("--launch needs --methods")
        for i, r in enumerate(runs):
            launch(r, a.corpus, a.target, a.set_ids, a.methods, i + 1, a.model)

    # pooled across runs, so a single short trace does not set the numbers
    pool = collections.defaultdict(lambda: collections.Counter())
    rest = collections.defaultdict(list)
    for r in runs:
        h = hygiene(r)
        if not h:
            print(f"!! {r}: no steps found")
            continue
        for m, v in h["form"].items():
            pool[m]["form"] += v
        for m, v in h["proposed"].items():
            pool[m]["proposed"] += v
        for m, v in h["gate_rej"].items():
            pool[m]["gate_rej"] += v
        for m, v in h["gate_all"].items():
            pool[m]["gate_all"] += v
        for m, v in h["first_person"].items():
            pool[m]["fp"] += v
        for m, (lived, n) in h["survive"].items():
            pool[m]["surv"] += lived
            pool[m]["surv_n"] += n
        m_rest = restatement.measure(r, by_method=True)
        for m, d in ((m_rest or {}).get("methods") or {}).items():
            rest[m].append(d)

    labelled = [m for m in pool if m != "(none)"]
    if not labelled:
        raise SystemExit(
            "no method labels in any of those runs -- they were run without "
            "--methods, so there is nothing to attribute.")

    def mean(ds, k):
        vs = [d[k] for d in ds if d.get(k) is not None]
        return sum(vs) / len(vs) if vs else None

    print(f"\n{'method':<14}{'echo':>7}{'bound':>7}{'redund':>8}"
          f"{'form-rej':>10}{'gate-rej':>10}{'survive':>9}{'1st-pp':>7}   risk")
    for m in sorted(labelled):
        c = pool[m]
        ds = rest.get(m, [])
        e = mean(ds, "echo")
        b = mean(ds, "bound")
        rd = mean(ds, "redundancy")
        form_rate = c["form"] / (c["form"] + c["proposed"]) if (c["form"] + c["proposed"]) else None
        gate_rate = c["gate_rej"] / c["gate_all"] if c["gate_all"] else None
        surv = c["surv"] / c["surv_n"] if c["surv_n"] else None
        f = lambda x, w=7, p=2: (f"{x:>{w}.{p}f}" if x is not None else f"{'--':>{w}}")
        print(f"{m:<14}{f(e,7,3)}{f(b)}{f(rd,8)}{f(form_rate,10)}{f(gate_rate,10)}"
              f"{f(surv,9)}{c['fp']:>7}   {METHODS[m].form_risk if m in METHODS else ''}")

    print("\npredicted failure modes (methods.py) -- the row above is the test:")
    for m in sorted(labelled):
        if m in METHODS:
            print(f"  {m:<14}{METHODS[m].expected_failure}")
    print("\nlower is better on echo, bound, redund, form-rej, gate-rej and 1st-pp; "
          "higher on survive.")
    print("A method whose measured failure matches its declared one is understood. "
          "One that fails a DIFFERENT way is the finding.")


if __name__ == "__main__":
    main()

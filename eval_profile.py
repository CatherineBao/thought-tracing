"""Launch the three-arm profile sweep, then audit it.

One variable moves between arms. Everything else -- corpus, targets, set ids,
population size, model, RNG seed -- is pinned, because eval_motive_sep already
measured what happens when it is not: the same person on two seeds separated
about as far as the two real parties did, so an unpaired comparison on this
corpus measures the sampler.

  none     the default prompts
  infer    --infer-motive
  profile  --character-profile

  eval_profile.py --seeds 3                    # launch all three arms
  eval_profile.py --seeds 3 --arms profile     # re-run one arm
  eval_profile.py --score-only --judge         # audit what is on disk

Scoring is audit_profile.py; this file only launches and hands over.
"""
import argparse
import os
import subprocess
import sys

from audit_profile import ARMS, SCRAMBLE, run_id

# The color thread: Wolf (management) against Lyubovsky/Rovani (engineering).
# The same sets eval_motive_sep.CASES["color"] uses, so the two evaluations are
# talking about the same runs.
SETS = ("bloomfield-0002,bloomfield-0006,bloomfield-0009,"
        "bloomfield-0015,bloomfield-0031,bloomfield-0039")
TARGETS = ("Wolf", "Lyubovsky", "Rovani")

ARM_FLAGS = {
    "none": [],
    "infer": ["--infer-motive"],
    "profile": ["--character-profile"],
    # scram takes --profile-as per target, appended in launch()
    "scram": ["--character-profile"],
}

# Held fixed across arms. Matches eval_motive_sep.launch so the runs are
# comparable with what is already on disk.
COMMON = ["--corpus", "bloomfield", "--set-ids", SETS, "--chronological",
          "--n-hypotheses", "8", "--goal-seeding", "--extract-anchors",
          "--use-anchor", "--anchored-perturbation", "--enable-split",
          "--enable-expiry", "--max-chars", "99999"]


def launch(arm, seed, target, model, dry=False, skip_existing=False):
    rid = run_id(arm, seed, target)
    if skip_existing:
        from audit_profile import steps_for
        if steps_for(rid):
            print(f"  {rid} -- already on disk, skipped")
            return rid
    cmd = ([sys.executable, "run_musing.py", "--target", target]
           + COMMON + ["--tracing-model", model, "--seed", str(seed),
                       "--run-id", rid] + ARM_FLAGS[arm])
    if arm == "scram":
        wrong = SCRAMBLE.get(target)
        if not wrong:
            print(f"  [!] no scramble partner for {target}; skipped")
            return None
        cmd += ["--profile-as", wrong]
    if dry:
        print("  " + " ".join(cmd))
        return rid
    os.makedirs("musing_out", exist_ok=True)
    print(f"  {rid} ...", flush=True)
    with open(f"musing_out/{rid}.log", "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        # run_musing exits non-zero on a short/failed trace. Say so here rather
        # than letting the audit quietly score a truncated run.
        print(f"  [!] {rid} exited {r.returncode} -- see musing_out/{rid}.log")
    return rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds, 0..n-1")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not re-trace a run whose steps are already on disk")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--head-to-head", action="store_true")
    a = ap.parse_args()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    targets = [t.strip() for t in a.targets.split(",") if t.strip()]
    seeds = list(range(a.seeds))

    if not a.score_only:
        n = len(arms) * len(targets) * len(seeds)
        print(f"launching {n} runs ({len(arms)} arms x {len(targets)} targets "
              f"x {len(seeds)} seeds)")
        for arm in arms:
            for seed in seeds:
                for target in targets:
                    launch(arm, seed, target, a.model, dry=a.dry_run,
                           skip_existing=a.skip_existing)
        if a.dry_run:
            return

    cmd = [sys.executable, "audit_profile.py",
           "--targets", ",".join(targets),
           "--seeds", ",".join(str(s) for s in seeds),
           "--arms", ",".join(arms), "--model", a.model,
           "--json", "musing_out/meta/profile_audit.json"]
    if a.judge:
        cmd.append("--judge")
    if a.head_to_head:
        cmd.append("--head-to-head")
    print("\n" + " ".join(cmd))
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

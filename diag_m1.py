"""M1 diagnostics D2-D4. DEV ONLY -- the test partition has had its one look."""
import collections, json, math, statistics as st
import forecast_likelihood as F

ARMS = ["context_neutral", "obvious", "hypothesis", "placebo", "swap"]
d = json.load(open("m1_casino.json", encoding="utf-8"))
dev = [r for r in d["records"] if r["split"] == "dev"]
ALTS = dev[0]["alts"]
print(f"dev n={len(dev)}  k={len(ALTS)}  labels={ALTS}\n")


def consensus(mean_rank, alts):
    return [alts.index(a) for a in sorted(alts, key=lambda x: (mean_rank.get(x, 99), x))]


def dists(rows, arm, theta, bias=None):
    """Per-row distribution over actions from the averaged rank, optional per-class bias."""
    out = []
    for r in rows:
        mr = r.get(f"dist_{arm}")
        if not mr:
            out.append(None); continue
        rk = consensus(mr, r["alts"])
        dd = F.to_actions(F.rank_to_distribution(rk, len(r["alts"]), theta), r["alts"], None)
        if bias:
            dd = {a: dd.get(a, 0.0) * math.exp(bias.get(a, 0.0)) for a in r["alts"]}
            z = sum(dd.values()) or 1.0
            dd = {a: v / z for a, v in dd.items()}
        out.append(dd)
    return out


def score(rows, ds):
    return [math.log(max(dd.get(r["actual"], 0.0), 1e-12)) for r, dd in zip(rows, ds) if dd]


def fit_theta(rows, arm):
    best, bll = 1.0, -1e18
    for th in [round(0.05 * i, 2) for i in range(1, 20)]:
        s = sum(score(rows, dists(rows, arm, th)))
        if s > bll:
            best, bll = th, s
    return best


counts = collections.Counter(r["actual"] for r in dev)
MAJ = {a: (counts.get(a, 0) + 1) / (sum(counts.values()) + len(ALTS)) for a in ALTS}
maj_ll = [math.log(MAJ[r["actual"]]) for r in dev]
print(f"majority (Laplace, dev-fitted) mean log-score {st.mean(maj_ll):+.4f}\n")

thetas = {a: fit_theta(dev, a) for a in ARMS}

# ---------------- D2: loss split by label type ----------------
print("=== D2  log-loss by label type (tie labels are 33.5% of the corpus)")
print(f"  {'arm':<18}{'single':>10}{'tie':>10}{'n_single':>10}{'n_tie':>8}")
is_tie = ["+" in r["actual"] for r in dev]
for arm in ARMS + ["majority"]:
    ll = maj_ll if arm == "majority" else score(dev, dists(dev, arm, thetas[arm]))
    sing = [v for v, t in zip(ll, is_tie) if not t]
    ties = [v for v, t in zip(ll, is_tie) if t]
    print(f"  {arm:<18}{st.mean(sing):>10.4f}{st.mean(ties):>10.4f}{len(sing):>10}{len(ties):>8}")

# ---------------- D3: vector scaling ----------------
print("\n=== D3  vector scaling: per-class bias fitted on dev alongside THETA")
print(f"  {'arm':<18}{'theta-only':>12}{'+bias':>10}{'gain':>9}{'vs majority':>13}")
for arm in ARMS:
    base = st.mean(score(dev, dists(dev, arm, thetas[arm])))
    bias = {a: 0.0 for a in ALTS}
    for _ in range(60):                       # coordinate ascent, dev only
        for a in ALTS:
            best_b, best_s = bias[a], st.mean(score(dev, dists(dev, arm, thetas[arm], bias)))
            for step in (-0.4, -0.2, -0.1, 0.1, 0.2, 0.4):
                trial = dict(bias); trial[a] = bias[a] + step
                s = st.mean(score(dev, dists(dev, arm, thetas[arm], trial)))
                if s > best_s:
                    best_b, best_s = trial[a], s
            bias[a] = best_b
    after = st.mean(score(dev, dists(dev, arm, thetas[arm], bias)))
    print(f"  {arm:<18}{base:>12.4f}{after:>10.4f}{after-base:>+9.4f}{after-st.mean(maj_ll):>+13.4f}")

# ---------------- D4: signal test ----------------
print("\n=== D4  signal test: p = (1-lam)*majority + lam*arm, lam fitted on dev")
print(f"  {'arm':<18}{'best lam':>10}{'logscore':>11}{'vs majority':>13}  verdict")
for arm in ARMS:
    ds = dists(dev, arm, thetas[arm])
    best_l, best_s = 0.0, st.mean(maj_ll)
    for i in range(0, 21):
        lam = i / 20
        mix = [{a: (1 - lam) * MAJ[a] + lam * dd.get(a, 0.0) for a in ALTS} for dd in ds]
        s = st.mean([math.log(max(m[r["actual"]], 1e-12)) for r, m in zip(dev, mix)])
        if s > best_s + 1e-9:
            best_l, best_s = lam, s
    verdict = ("carries usable signal" if best_l > 0 else
               "NO usable signal at any lambda > 0")
    print(f"  {arm:<18}{best_l:>10.2f}{best_s:>11.4f}{best_s-st.mean(maj_ll):>+13.4f}  {verdict}")

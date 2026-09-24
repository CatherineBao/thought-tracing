"""Milestone 1: the choice likelihood, one-shot, no filter. CaSiNo.

Registered in PREREG. What M1 asks: can a motive inferred from the prefix
forecast the choice better than the frozen bar, and does the PLACEBO stay down?

Arms (PREREG "Arms and scoring"):
  majority        corpus-level dev prior, Laplace-smoothed. 0 calls.
  context_neutral full context + an equal-length NEUTRAL block. Length control.
  obvious         a colleague-style reading OF THE PERSON, no methods. Bar candidate.
  hypothesis      a portfolio generated from the prefix. The thing being tested.
  placebo         same shape and length, generic non-person-specific content.
                  THE BINDING CONTROL -- Phase CP's failure was a think-harder
                  effect, and only a generic same-shape portfolio isolates it.
  swap            another person's portfolio, by deterministic rotation.

THE BAR IS FROZEN ON DEV before test is read: whichever of obvious/majority has
the better dev log-score. Taking a pointwise max at the realised outcome would
not be a proper score.

THETA is fitted on dev PER ARM (standing rule 9): calibrating one arm and not the
others converts a calibration gain into apparent lift.

The silent subgroup -- people who stated no priority in their own prefix, 0.630
of points -- is the one genuine test here, because CaSiNo's priorities are
otherwise spoken aloud. It is reported separately and was defined before any
forecast existed.
"""
import collections, json, math, os, random, statistics as st, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')):
    if '=' in line:
        k, v = line.strip().split('=', 1); os.environ[k] = v
from google import genai
from google.genai import types
import forecast_likelihood as F, choice_points as cp

MODEL = "gemini-2.5-flash-lite"
ORDERS = 2                  # k=8 sits between the confirmed sizes; invariance is measured below
N_DEV, N_TEST = 40, 80
OUT = "m1_casino.json"
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
CALLS = [0]


def gen(system, user, max_tokens=256):
    CALLS[0] += 1
    for attempt in range(4):
        try:
            r = client.models.generate_content(
                model=MODEL, contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system, temperature=0, max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)))
            return (r.text or "").strip()
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def prefix_of(p, n=40):
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in p["prefix_turns"][-n:]) \
        or "(nothing said yet)"


def make_portfolio(p):
    """One to three motives in priority order, from the PREFIX ONLY.

    Never sees value2issue or value2reason: the answer key is scoring-only, and
    value2reason is keyed High/Medium/Low so prompting it would hand over the
    ground-truth order outright.
    """
    s = ("You infer what somebody wants from how they talk, before they have said it.\n"
         "Give one to three motives in priority order, most important first. Each is a "
         "short aim in the person's own terms, at most 8 words. Do not restate what they "
         "said; name what they are trying to get.\n\nAnswer exactly:\n"
         "1. <aim>\n2. <aim>\n3. <aim>   (fewer lines is fine)")
    u = (f"<conversation so far>\n{prefix_of(p)}\n</conversation so far>\n\n"
         f"What does {p['person']} want out of this negotiation?")
    raw = gen(s, u, 160)
    aims = [ln.split(".", 1)[1].strip() for ln in raw.splitlines()
            if ln.strip()[:2] in ("1.", "2.", "3.") and "." in ln][:3]
    return "; then ".join(aims) if aims else None


def make_obvious(p):
    """A colleague-style reading OF THE PERSON, generated with no methods."""
    s = ("You are a thoughtful colleague reading a conversation. In two sentences, say "
         "what you make of this person -- what they seem to care about and how they are "
         "playing it. Plain, careful, no jargon.")
    u = f"<conversation so far>\n{prefix_of(p)}\n</conversation so far>\n\nYour read on {p['person']}?"
    return gen(s, u, 160)


NEUTRAL = ("General note: negotiations of this kind involve three kinds of supplies, and "
           "participants trade them across a short conversation before proposing a split.")
PLACEBO = "wants a fair deal; then wants the conversation to go smoothly"


def forecast(p, portfolio):
    """Mean distribution over ORDERS renderings. Returns (dist, per_order)."""
    alts = [a["action"] for a in p["alternatives"]]
    n = len(alts)
    rng = random.Random(hash(p["cp_id"]) % 99991)
    orders = [list(range(n))]
    for _ in range(ORDERS - 1):
        o = list(range(n)); rng.shuffle(o); orders.append(o)
    # AVERAGE THE RANK POSITIONS, NOT A DISTRIBUTION.
    #
    # The obvious implementation -- turn each order's ranking into a
    # distribution and average those -- silently destroys the ranking unless a
    # theta is chosen first, and choosing one here would bake in before the dev
    # fit. At theta=1.0 rank_to_distribution is UNIFORM, so every point would
    # average to 1/n and the recovered consensus order would be alphabetical
    # noise. (Caught by reading the code, not by the output, which looked
    # perfectly well formed.)
    #
    # Averaging the POSITION each action was given is a Borda aggregation: it
    # preserves order across renderings, needs no theta, and leaves theta to be
    # fitted on dev afterwards where it belongs.
    ranks = []
    for o in orders:
        s, u = F.forecast_prompts(prefix_of(p), p["person"], p["situation"], alts, portfolio, o)
        rk = F.parse_ranking(gen(s, u, 96), n)
        if rk is None:
            continue
        pos = {}
        for place, slot in enumerate(rk):
            pos[alts[o[slot]]] = place
        ranks.append(pos)
    if not ranks:
        return None, []
    mean = {a: sum(r.get(a, n - 1) for r in ranks) / len(ranks) for a in alts}
    return mean, ranks


def consensus(mean_rank, alts):
    """Lowest mean position first. Ties broken by label so the series is
    deterministic across runs."""
    return [alts.index(a) for a in sorted(alts, key=lambda x: (mean_rank.get(x, 99), x))]


def score(recs, theta):
    return st.mean([math.log(max(F.rank_to_distribution(r, n, theta).get(a, 0.0), 1e-12))
                    for r, a, n in recs])


def laplace_prior(points, alts):
    c = collections.Counter(p["actual"] for p in points)
    tot = sum(c.values()) + len(alts)
    return {a: (c.get(a, 0) + 1) / tot for a in alts}


def main():
    pts = cp.load(corpus="casino")
    rng = random.Random(0)
    rng.shuffle(pts)
    dev = [p for p in pts if int(p["set_id"].split("-")[1]) % 2 == 0][:N_DEV]
    test = [p for p in pts if int(p["set_id"].split("-")[1]) % 2 == 1][:N_TEST]
    print(f"dev {len(dev)}  test {len(test)}  silent in test: "
          f"{sum(1 for p in test if p['silent'])}", flush=True)

    store = {}
    for split, pool in (("dev", dev), ("test", test)):
        for i, p in enumerate(pool):
            alts = [a["action"] for a in p["alternatives"]]
            rec = {"cp_id": p["cp_id"], "person_uid": p["person_uid"], "split": split,
                   "actual": p["actual"], "silent": p["silent"], "alts": alts}
            rec["portfolio"] = make_portfolio(p)
            rec["obvious_text"] = make_obvious(p)
            store[p["cp_id"]] = rec
            if i % 20 == 0:
                print(f"  {split} gen {i}/{len(pool)}  calls={CALLS[0]}", flush=True)

    # swap: deterministic rotation within split, so every person is used once
    for split in ("dev", "test"):
        ids = sorted([k for k, v in store.items() if v["split"] == split])
        for i, k in enumerate(ids):
            store[k]["swap_portfolio"] = store[ids[(i + 1) % len(ids)]]["portfolio"]

    ARMS = {"context_neutral": lambda r: NEUTRAL,
            "obvious": lambda r: r["obvious_text"],
            "hypothesis": lambda r: r["portfolio"],
            "placebo": lambda r: PLACEBO,
            "swap": lambda r: r["swap_portfolio"]}
    byid = {p["cp_id"]: p for p in dev + test}
    for arm, pick in ARMS.items():
        done = 0
        for k, r in store.items():
            d, per = forecast(byid[k], pick(r))
            r[f"dist_{arm}"] = d
            if arm == "context_neutral" and len(per) == 2:
                # order sensitivity as mean absolute rank displacement between
                # two renderings, in places -- the k>=9 diagnosis says the
                # instability is variance in the ordering itself, not a
                # positional prior, so measure it in ranks
                r["perm_rank_move"] = st.mean(
                    [abs(per[0].get(a, 0) - per[1].get(a, 0)) for a in r["alts"]])
            done += 1
            if done % 40 == 0:
                print(f"  {arm} {done}/{len(store)}  calls={CALLS[0]}", flush=True)
        print(f"arm {arm} done, calls={CALLS[0]}", flush=True)

    json.dump({"model": MODEL, "orders": ORDERS, "calls": CALLS[0],
               "records": list(store.values())},
              open(OUT, "w", encoding="utf-8"))
    print(f"wrote {OUT}  total calls {CALLS[0]}", flush=True)


if __name__ == "__main__":
    main()

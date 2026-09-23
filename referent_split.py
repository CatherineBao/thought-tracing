"""Find terms two people use to mean different things. No board, no LLM.

A SIDE-CHECK, not part of the critical path. Stage 3 is closed: three signal
designs failed, and FINDINGS §4 explains why they share a cause -- all three
read the particle population, and on these corpora "the work of belief
tracking is done upstream of the particle population ... population diversity
operates where those checks do not reach."

So this samples somewhere else entirely. A referent split is at root a
DISTRIBUTIONAL fact about a word: on the Bloomfield thread "size" means
commercial caliber grading for export on one side and green-berry sizing for
harvest timing on the other, and that shows up as the two parties putting
systematically different words around the same term. The transcript carries
that directly. Nothing here touches the filter.

WHAT IT DOES. For every content term both parties use often enough, build each
party's bag of co-occurring words and measure how far apart the two bags are.
Rank terms by that distance.

WHY A PERMUTATION NULL IS NOT OPTIONAL. Two bags built from different turns
differ even when the speakers mean the same thing, and they differ MORE when
one party has fewer turns. Without a null this reports vocabulary size. So
every term is scored against the distribution of distances obtained by
reshuffling which party said which turn, holding the turn counts fixed. The
statistic is the percentile of the real split in its own null -- a term only
counts if the real speakers separate further than random speakers would.

This is the discipline the three culture detectors in this repo died for lack
of, and that `eval_motive_sep.py` states outright: a variant is only an
improvement if it raises the real score WITHOUT raising the control.

LIMITS, up front. It cannot see a split carried entirely by tone or by
reference to shared context rather than by word choice, and with Slack-length
turns the bags are small. It is a cheap first look, not an instrument.
"""
import argparse
import collections
import json
import math
import random
import re

WORD = re.compile(r"[a-z][a-z'-]{2,}")
STOP = set("""the and for you that this with have has had was were are not but
can will would should could our your their they them there then than from into
out off any all some more most just like get got make made need want know look
see thing things one two use used using about over under when what which who
whom how why where does did done doing able very much many lot lots okay yeah
yes yep nope sure thanks thank please hey hi hello cc via etc ok
""".split())


# Slack markup must go BEFORE tokenising. <@U0DJRRSTA> tokenises to "djrrsta"
# and <#C123|chan> to the channel slug, and both then look like rare shared
# vocabulary -- the first run of this tool ranked "djrrsta" and "rdabe" top
# two. They are addressing, not word choice.
MARKUP = re.compile(r'<[@#!][^>]*>|<https?://[^>]*>|```.*?```|`[^`]*`|:[a-z_]+:',
                    re.S)


def content(text):
    return [w for w in WORD.findall(MARKUP.sub(' ', text or '').lower())
            if w not in STOP]


def turns_for(corpus, set_ids=None, data_dir='data/musing'):
    with open(f'{data_dir}/{corpus}_dialogue.json', encoding='utf-8') as fh:
        data = json.load(fh)
    keep = set(set_ids) if set_ids else None
    out = []
    for s in data:
        if keep and s['set_id'] not in keep:
            continue
        for t in s['turns']:
            out.append((t['speaker'], t.get('text') or '', s['set_id']))
    return out


def bags(turns, term):
    """Per-speaker bag of words co-occurring with `term`, one bag per turn."""
    per = collections.defaultdict(list)
    for spk, text, _sid in turns:
        ws = content(text)
        if term in ws:
            per[spk].append([w for w in ws if w != term])
    return per


def _dist(a_bags, b_bags):
    """Jensen-Shannon distance between two pooled co-occurrence bags.

    JS rather than cosine because it is bounded, symmetric, and defined when
    the two vocabularies barely overlap -- which is the case this exists to
    detect, and exactly where cosine degenerates to 0 and stops discriminating.
    """
    A = collections.Counter(w for b in a_bags for w in b)
    B = collections.Counter(w for b in b_bags for w in b)
    if not A or not B:
        return None
    za, zb = sum(A.values()), sum(B.values())
    keys = set(A) | set(B)
    js = 0.0
    for w in keys:
        p, q = A.get(w, 0) / za, B.get(w, 0) / zb
        m = (p + q) / 2
        if p:
            js += 0.5 * p * math.log(p / m, 2)
        if q:
            js += 0.5 * q * math.log(q / m, 2)
    return math.sqrt(max(js, 0.0))


def split_score(turns, term, A, B, n_perm=200, rng=None):
    """Real split distance, and its percentile against a label-shuffled null."""
    rng = rng or random.Random(0)
    per = bags(turns, term)
    a = [b for spk in A for b in per.get(spk, [])]
    b = [b for spk in B for b in per.get(spk, [])]
    if len(a) < 3 or len(b) < 3:
        return None
    real = _dist(a, b)
    if real is None:
        return None
    pool = a + b
    na = len(a)
    null = []
    for _ in range(n_perm):
        rng.shuffle(pool)
        d = _dist(pool[:na], pool[na:])
        if d is not None:
            null.append(d)
    if not null:
        return None
    pct = sum(1 for x in null if x < real) / len(null)
    mean = sum(null) / len(null)
    return {'term': term, 'real': real, 'null_mean': mean,
            'percentile': pct, 'n_a': len(a), 'n_b': len(b),
            'excess': real - mean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='bloomfield')
    ap.add_argument('--a', required=True, help='party A speakers, comma separated')
    ap.add_argument('--b', required=True, help='party B speakers')
    ap.add_argument('--sets', default=None, help='restrict to these set ids')
    ap.add_argument('--min-uses', type=int, default=4,
                    help='minimum turns per party containing the term')
    ap.add_argument('--perms', type=int, default=200)
    ap.add_argument('--top', type=int, default=12)
    a = ap.parse_args()

    A = set(a.a.split(','))
    B = set(a.b.split(','))
    turns = turns_for(a.corpus, a.sets.split(',') if a.sets else None)
    ta = [t for t in turns if t[0] in A]
    tb = [t for t in turns if t[0] in B]
    print(f'{"/".join(sorted(A))}  vs  {"/".join(sorted(B))}')
    print(f'turns: {len(ta)} vs {len(tb)}  (of {len(turns)} in span)\n')

    ca = collections.Counter(w for _, tx, _ in ta for w in set(content(tx)))
    cb = collections.Counter(w for _, tx, _ in tb for w in set(content(tx)))
    shared = [w for w in set(ca) & set(cb)
              if ca[w] >= a.min_uses and cb[w] >= a.min_uses]
    print(f'terms used >= {a.min_uses} times by BOTH parties: {len(shared)}')
    if not shared:
        return

    rng = random.Random(0)
    rows = [r for r in (split_score(turns, w, A, B, a.perms, rng)
                        for w in sorted(shared)) if r]
    rows.sort(key=lambda r: (-r['percentile'], -r['excess']))
    print(f'scored: {len(rows)}\n')
    print(f'{"term":<18}{"A":>4}{"B":>4}{"JS":>8}{"null":>8}{"excess":>9}{"pct":>7}')
    for r in rows[:a.top]:
        print(f'{r["term"]:<18}{r["n_a"]:>4}{r["n_b"]:>4}{r["real"]:>8.3f}'
              f'{r["null_mean"]:>8.3f}{r["excess"]:>+9.3f}{r["percentile"]:>7.0%}')

    hi = [r for r in rows if r['percentile'] >= 0.95]
    print(f'\nterms clearing the 95th percentile of their own null: '
          f'{len(hi)} of {len(rows)}')
    if hi:
        print('  ' + ', '.join(r['term'] for r in hi[:15]))
    print('\nA term only counts if the REAL speakers separate further than')
    print('shuffled ones. Expect a handful by chance at 5%: compare against a')
    print('same-side pair before believing any of them.')


if __name__ == '__main__':
    main()

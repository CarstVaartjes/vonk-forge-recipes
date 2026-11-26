"""Directed matcher fixtures (shared by the CPU tests and the GPU check): unique-token histories with planted matches
at chosen distances and lengths. Each fixture: (name, tokens, L, expected e (or -1), expected m)."""


def _unique(n, base=10000):
    return [base + i for i in range(n)]


def build(G=7, key=12, maxlen=64):
    out = []
    n = 1400
    # tile boundaries: the only full match ends exactly d tokens back
    for d in (13, 100, 255, 256, 257, 511, 512, 513, 1000):
        arr = _unique(n)
        L = n
        e = L - d
        for i in range(key):
            arr[L - key + i] = arr[e - key + i]
        out.append((f"boundary-{d}", arr, L, e, key))
    # a newer 13-token match blocks an older 30-token one (the most recent full key wins)
    arr = _unique(n)
    L = n
    old_e, new_e = 300, 900
    for i in range(30):
        arr[old_e - 30 + i] = arr[L - 30 + i]
    for i in range(13):
        arr[new_e - 13 + i] = arr[L - 13 + i]
    out.append(("newer-partial", arr, L, new_e, 13))
    # newer key-only (last-token) hits do not block an older full match
    arr = _unique(n)
    L = n
    full_e = 400
    for i in range(20):
        arr[full_e - 20 + i] = arr[L - 20 + i]
    for c in range(800, 1300, 7):
        arr[c - 1] = arr[L - 1]          # last token matches, the rest of the key does not
    out.append(("key-only-hits", arr, L, full_e, 20))
    # an 80-token match: the extension stops at the cap
    arr = _unique(n)
    L = n
    e = 600
    for i in range(80):
        arr[e - 80 + i] = arr[L - 80 + i]
    out.append(("cap-64", arr, L, e, maxlen))
    # no match at all
    out.append(("none", _unique(n), n, -1, 0))
    return out

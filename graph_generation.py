import random

def ER(n, m, frac_hidden=0.1, directed=False, has_self_loops=False, \
       coin_toss_point=0.05):

    # From empirical tests, a coin_toss_point of 0.05 seems to be about ideal.

    directed = int(directed)
    has_self_loops = int(has_self_loops)
    assert n > 0
    assert m >= 0
    max_m = ((n * (n - 1)) / (2 - directed)) + n * has_self_loops
    assert m <= max_m

    if m <= max_m * coin_toss_point:
        neighbors_collections = [set() for _ in range(0, n)]
        hidden_edges = set()
        if has_self_loops:
            self_loops = [False for _ in range(0, n)]
        else:
            self_loops = None

        add_k_random_edges(neighbors_collections, hidden_edges, m, \
                           self_loops=self_loops, frac_hidden=frac_hidden, \
                           directed=directed)
    else:
        p = float(m) / max_m
        (gen_m, neighbors_collections, hidden_edges, self_loops) = \
            coin_toss_for_edges(n, p, frac_hidden=frac_hidden, \
                                directed=directed, \
                                has_self_loops=has_self_loops)

        if gen_m < m:
            add_k_random_edges(neighbors_collections, hidden_edges, m - gen_m, \
                               self_loops=self_loops, frac_hidden=frac_hidden, \
                               directed=directed)
        elif gen_m > m:
            remove_k_random_edges(neighbors_collections, hidden_edges, \
                                  gen_m - m, \
                                  self_loops=self_loops, directed=directed)

    return (neighbors_collections, hidden_edges, self_loops)

# Use with sparse graphs.
def add_k_random_edges(neighbors_collections, hidden_edges, k, \
                       self_loops=None, \
                       frac_hidden=0.1, directed=False):
    n = len(neighbors_collections)

    for _ in range(0, k):
        valid = False
        while not valid:
            a = random.randint(0, n - 1)
            if self_loops is not None:
                b = random.randint(0, n - 1)
            else:
                b = random.randint(0, n - 2)
                if b >= a:
                    b += 1

            if directed:
                edge = (a, b)
            else:
                edge = (min(a, b), max(a, b))

            if edge in hidden_edges:
                continue

            if a == b:
                if not self_loops[a]:
                    valid = True
                    if random.random() < frac_hidden:
                        hidden_edges.add(edge)
                    else:
                        self_loops[a] = True
            else:
                if b not in neighbors_collections[a]:
                    valid = True
                    if random.random() < frac_hidden:
                        hidden_edges.add(edge)
                    else:
                        neighbors_collections[a].add(b)
                        if not directed:
                            neighbors_collections[b].add(a)

# Use with dense graphs.
def remove_k_random_edges(neighbors_collections, hidden_edges, k, \
                          self_loops=None, directed=False):

    n = len(neighbors_collections)

    for _ in range(0, k):
        valid = False
        while not valid:
            a = random.randint(0, n - 1)
            if self_loops is not None:
                b = random.randint(0, n - 1)
            else:
                b = random.randint(0, n - 2)
                if b >= a:
                    b += 1

            if directed:
                edge = (a, b)
            else:
                edge = (min(a, b), max(a, b))

            if edge in hidden_edges:
                valid = True
                hidden_edges.remove(edge)
            elif a == b:
                if self_loops[a]:
                    valid = True
                    self_loops[a] = False
            else:
                if b in neighbors_collections[a]:
                    valid = True
                    neighbors_collections[a].remove(b)
                    if not directed:
                        neighbors_collections[b].remove(a)

# Useful for generating dense graphs.
def coin_toss_for_edges(n, p, frac_hidden=0.1, directed=False, has_self_loops=False):
    directed = int(directed)
    has_self_loops = int(has_self_loops)

    neighbors_collections = [set() for _ in range(0, n)]
    hidden_edges = set()
    if has_self_loops:
        self_loops = [False for _ in range(0, n)]
    else:
        self_loops = None

    m = 0
    for a in range(0, n):
        for b in range((1 - directed) * a, n - (1 - has_self_loops)):
            if not has_self_loops:
                if b >= a:
                    b += 1

            if random.random() >= p:
                continue

            m += 1

            edge = (a, b)
            if random.random() < frac_hidden:
                hidden_edges.add(edge)
            else:
                if a == b:
                    self_loops[a] = True
                else:
                    neighbors_collections[a].add(b)
                    if not directed:
                        neighbors_collections[b].add(a)

    return (m, neighbors_collections, hidden_edges, self_loops)

def Watts_Strogatz(N, K, beta, frac_hidden=0.1):
    neighbors_collections = [set() for _ in range(0, N)]

    # Initial Lattice
    for i in range(0, N):
        for j in range(i + 1, i + 1 + K):
            l = j % N
            neighbors_collections[i].add(l)
            neighbors_collections[l].add(i)

    # Rewiring
    for i in range(0, N):
        for j in range(i + 1, i + 1 + K):
            if random.random() >= beta:
                continue
            l = j % N
            neighbors_collections[i].remove(l)
            neighbors_collections[l].remove(i)
            while True:
                l = random.randint(0, (N - 1) - 1)
                if l >= i:
                    l += 1
                if l not in neighbors_collections[i]:
                    neighbors_collections[i].add(l)
                    neighbors_collections[l].add(i)
                    break

    # Edge Hiding
    hidden_edges = set()
    for i in range(0, N):
        neighbors = list(neighbors_collections[i])
        for n in neighbors:
            if n < i:
                continue
            if random.random() < frac_hidden:
                neighbors_collections[i].remove(n)
                neighbors_collections[n].remove(i)
                hidden_edges.add((i, n))

    return (neighbors_collections, hidden_edges)

if __name__ == "__main__":
    import time

    N = 5000
    hidden_frac = 0.1
    target_M_ratio = 0.05
    cutoff_M_ratio = 0.049

    for DIRECTED in [False, True]:
        for HAS_SELF_LOOPS in [False, True]:
            max_M = ((N*(N-1)) / (2 - int(DIRECTED))) + N * int(HAS_SELF_LOOPS)

            target_M = int(max_M * target_M_ratio)
            print("Generating a%s graph with %d nodes and %d edges..." % \
                    ({(False, False): "", (False, True): "n SL", \
                      (True, False): " directed", (True, True): " directed and SL"}[(DIRECTED, HAS_SELF_LOOPS)], \
                        N, target_M))

            start_t = time.time()
            (NC, HE, SL) = ER(N, target_M, frac_hidden=hidden_frac, \
                   directed=DIRECTED, has_self_loops=HAS_SELF_LOOPS, \
                   coin_toss_point=cutoff_M_ratio)
            end_t = time.time()

            gen_m = sum([len(c) for c in NC])
            if not DIRECTED:
                assert gen_m % 2 == 0
                gen_m = int(gen_m / 2)
            if HAS_SELF_LOOPS:
                gen_m += sum([int(v) for v in SL])
            gen_m += len(HE)
            if gen_m != target_M:
                print("%s vs %s" % (gen_m, target_M))
            assert gen_m == target_M
            print("Fraction of hidden edges vs. target: %f vs. %f" % \
                    (len(HE) / float(gen_m), hidden_frac))

            print("Total time: %s" % (end_t - start_t))

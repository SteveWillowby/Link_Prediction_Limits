#!/bin/bash

# Imitations with ER of graphs which give more interesting link pred results.

# y2h_ppi with 5% removed
time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=true repercent=5 nepercent=auto num_runs=15 hash_endpoints=true g=rand_y2h_ppi

# Cora ML with 5% removed
time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=true repercent=5 nepercent=auto num_runs=15 hash_endpoints=true g=rand_cora

# Citeseer with 5% removed
time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=true repercent=5 nepercent=auto num_runs=15 hash_endpoints=true g=rand_citeseer

# Power Grid with 5% removed
time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=true repercent=5 nepercent=auto num_runs=15 hash_endpoints=true g=rand_powergrid

# Linux Calls with 5% removed
time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=true repercent=5 nepercent=auto num_runs=15 hash_endpoints=true g=rand_linux_calls

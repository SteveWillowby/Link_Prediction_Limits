#! /bin/bash
time python3 test_local_graphs.py np=59 ntpp=1 k=all py_iso=false repercent=5  nepercent=auto num_runs=15 hash_endpoints=false g=citeseer
time python3 test_local_graphs.py np=59 ntpp=1 k=all py_iso=false repercent=10 nepercent=auto num_runs=15 hash_endpoints=false g=citeseer
time python3 test_local_graphs.py np=59 ntpp=1 k=all py_iso=false repercent=20 nepercent=auto num_runs=15 hash_endpoints=false g=citeseer

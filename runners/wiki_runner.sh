#! /bin/bash
time python3 test_local_graphs.py np=43 ntpp=1 k=all py_iso=true repercent=5  nepercent=auto num_runs=15 hash_endpoints=true g=wiki
time python3 test_local_graphs.py np=43 ntpp=1 k=all py_iso=true repercent=10 nepercent=auto num_runs=15 hash_endpoints=true g=wiki
time python3 test_local_graphs.py np=43 ntpp=1 k=all py_iso=true repercent=20 nepercent=auto num_runs=15 hash_endpoints=true g=wiki

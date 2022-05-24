#! /bin/bash

time python3 test_local_graphs.py np=40 ntpp=1 k=all py_iso=true repercent=10 nepercent=auto num_runs=15 hash_endpoints=true g=gcn_citeseer

time python3 test_local_graphs.py np=40 ntpp=1 k=all py_iso=true repercent=10 nepercent=auto num_runs=15 hash_endpoints=true g=gcn_cora

time python3 test_local_graphs.py np=40 ntpp=1 k=all py_iso=true repercent=10 nepercent=auto num_runs=15 hash_endpoints=true g=gcn_pubmed

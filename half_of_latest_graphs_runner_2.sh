#! /bin/bash

time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=false repercent=10 nepercent=100 num_runs=2 hash_endpoints=true g=jazz_collab

time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=false repercent=10 nepercent=100 num_runs=2 hash_endpoints=true g=linux_calls
time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=false repercent=10 nepercent=100 num_runs=2 hash_endpoints=true g=mysql_calls

time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=false repercent=10 nepercent=100 num_runs=2 hash_endpoints=true g=roget_thesaurus

time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=false repercent=10 nepercent=100 num_runs=2 hash_endpoints=true g=species_1_brain

time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=false repercent=10 nepercent=100 num_runs=2 hash_endpoints=true g=US_500_airports_u

time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=false repercent=10 nepercent=100 num_runs=2 hash_endpoints=true g=US_airports_2010_l

time python3 test_local_graphs.py np=58 ntpp=1 k=all py_iso=false repercent=10 nepercent=100 num_runs=2 hash_endpoints=true g=GR_coauth

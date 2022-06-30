#! /bin/bash

time python3 test_local_graphs.py np=45 ntpp=1 k=all py_iso=true repercent=10 nepercent=100 num_runs=4 hash_endpoints=true g=species_1_brain

time python3 test_local_graphs.py np=45 ntpp=1 k=all py_iso=true repercent=10 nepercent=100 num_runs=4 hash_endpoints=true g=US_500_airports_u

time python3 test_local_graphs.py np=45 ntpp=1 k=all py_iso=true repercent=10 nepercent=100 num_runs=4 hash_endpoints=true g=eucore

time python3 test_local_graphs.py np=45 ntpp=1 k=all py_iso=true repercent=10 nepercent=100 num_runs=4 hash_endpoints=true g=roget_thesaurus

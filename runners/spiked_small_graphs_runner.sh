#! /bin/bash
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=5  nepercent=auto num_runs=20 hash_endpoints=false g=karate
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=10 nepercent=auto num_runs=20 hash_endpoints=false g=karate
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=20 nepercent=auto num_runs=20 hash_endpoints=false g=karate

time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=5  nepercent=auto num_runs=20 hash_endpoints=false g=innovation
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=10 nepercent=auto num_runs=20 hash_endpoints=false g=innovation
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=20 nepercent=auto num_runs=20 hash_endpoints=false g=innovation

time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=5  nepercent=auto num_runs=20 hash_endpoints=false g=convote
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=10 nepercent=auto num_runs=20 hash_endpoints=false g=convote
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=20 nepercent=auto num_runs=20 hash_endpoints=false g=convote

time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=5  nepercent=auto num_runs=20 hash_endpoints=false g=highschool
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=10 nepercent=auto num_runs=20 hash_endpoints=false g=highschool
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=20 nepercent=auto num_runs=20 hash_endpoints=false g=highschool

time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=5  nepercent=auto num_runs=20 hash_endpoints=false g=celegans_m
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=10 nepercent=auto num_runs=20 hash_endpoints=false g=celegans_m
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=20 nepercent=auto num_runs=20 hash_endpoints=false g=celegans_m

time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=5  nepercent=auto num_runs=20 hash_endpoints=false g=foodweb
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=10 nepercent=auto num_runs=20 hash_endpoints=false g=foodweb
time python3 link_pred_comparison_runner.py np=7 ntpp=1 k=all py_iso=true repercent=20 nepercent=auto num_runs=20 hash_endpoints=false g=foodweb

# ROC and AUPR Limits for Link Prediction

This repository contains the code associated with the paper
"Inherent Limits on Topology-Based Link Prediction," under review at Transactions on Machine Learning Research (TMLR).

## How to Run

Many examples of running the code can be found in the `runners` folder.

If you're simply interested in the functions to calculate maximum possible scores, check out `max_score_functions.py`.

### Note

The code uses a python wrapper around the compiled executable of graph isomorphism solvers Nauty and Traces. The binary is compiled for linux. If you wish to run on another OS, you may need to compile Nauty and Traces separately. The code that references the binary is `py_NT_session.py` and `ram_friendly_NT_session.py`.

To download Nauty and Traces, use [this website](https://pallini.di.uniroma1.it/).

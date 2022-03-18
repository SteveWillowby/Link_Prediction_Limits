#!/bin/bash

python3 plot_results.py test_results/ER_100_400_k-all_ref-0.1_nef-auto_he-true_raw.txt 410 420 430 440 450

python3 plot_results.py test_results/ER_200_800_k-all_ref-0.1_nef-auto_he-true_raw.txt 820 840 860 880 900

python3 plot_results.py test_results/ER_300_1200_k-all_ref-0.1_nef-auto_he-true_raw.txt 1230 1260 1290 1320 1350

python3 plot_results.py test_results/ER_400_1600_k-all_ref-0.1_nef-auto_he-true_raw.txt 1640 1680 1720 1760 1800

python3 plot_results.py test_results/ER_500_2000_k-all_ref-0.1_nef-auto_he-true_raw.txt 2050 2100 2150 2200 2250

python3 plot_results.py test_results/ER_600_2400_k-all_ref-0.1_nef-auto_he-true_raw.txt 2460 2520 2580 2640 2700

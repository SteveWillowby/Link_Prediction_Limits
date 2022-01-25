import matplotlib.pyplot as plt
import random
import statistics
import sys

if __name__ == "__main__":

    #################### Read the Results ####################

    filename = sys.argv[1]
    graph_name = filename.split("/")[1].split("_")[0]

    ROC_endpoints = [[], []]
    ROC_between_points = {}
    AUPR_endpoints = [[], []]
    AUPR_between_points = {}
    with open(filename, "r") as f:
        lines = f.readlines()

    KINF = 0
    K1 = 1
    KBET = 2
    phase = KBET
    searching = False
    for l in lines:
        if "k = " in l:
            k = l.strip()[4:]
            if k != "inf":
                k = int(k)

            if phase == KINF:
                phase = K1

            elif phase == K1:
                phase = KBET

            elif type(k) is str and k == "inf":
                phase = KINF

            searching = True

        if searching and " ROC: " in l:
            value = float(l.strip().split(" ")[2])

            if phase == KINF:
                ROC_endpoints[1].append(value)
            elif phase == K1:
                ROC_endpoints[0].append(value)
            else:
                if k not in ROC_between_points:
                    ROC_between_points[k] = []
                ROC_between_points[k].append(value)

        if searching and " AUPR: " in l:
            value = float(l.strip().split(" ")[2])

            if phase == KINF:
                AUPR_endpoints[1].append(value)
            elif phase == K1:
                AUPR_endpoints[0].append(value)
            else:
                if k not in AUPR_between_points:
                    AUPR_between_points[k] = []
                AUPR_between_points[k].append(value)

            searching = False

    #################### Get Means & Stdevs ####################

    ROC_avg_between_points = {k: (sum(l) / float(len(l)), statistics.pstdev(l)) \
                               for k, l in ROC_between_points.items()}
    AUPR_avg_between_points = {k: (sum(l) / float(len(l)), statistics.pstdev(l)) \
                               for k, l in AUPR_between_points.items()}

    ROC_max_k = max([k for k, l in ROC_avg_between_points.items()])
    AUPR_max_k = max([k for k, l in AUPR_avg_between_points.items()])

    ROC_avg_endpoints = [(sum(l) / float(len(l)), statistics.pstdev(l)) \
                          for l in ROC_endpoints]
    AUPR_avg_endpoints = [(sum(l) / float(len(l)), statistics.pstdev(l)) \
                           for l in AUPR_endpoints]

    #################### Create Plots ####################

    SPS_BASE = 0.0025
    STARTPOINT_SHIFTS = SPS_BASE * ROC_max_k
    ENDPOINT_SHIFT = 0.1
    JITTER_WIDTH = 0.05
    LW = 4.0  # linewidth
    A = 0.3  # alpha (i.e. transparency) \in (0, 1]

    x = [i for i in range(1, ROC_max_k + 1)]
    x_plotted = list(x)
    x_plotted[0] = x_plotted[0] + STARTPOINT_SHIFTS
    y = [ROC_avg_between_points[i][0] for i in x]
    yerr = [ROC_avg_between_points[i][1] for i in x]
    plt.errorbar(x_plotted, y, yerr=yerr, color="teal", linewidth=LW, label="nearly exact")
    x_start = [x[0] - STARTPOINT_SHIFTS]
    y_start = [ROC_avg_endpoints[0][0]]
    yerr_start = [ROC_avg_endpoints[0][1]]
    plt.errorbar(x_start, y_start, yerr=yerr_start, color="brown", linewidth=LW, label="k = 1")
    x_end = [x[-1] + ENDPOINT_SHIFT]
    y_end = [ROC_avg_endpoints[1][0]]
    yerr_end = [ROC_avg_endpoints[1][1]]
    plt.errorbar(x_end, y_end, yerr=yerr_end, color="orange", linewidth=LW, label="k = inf")

    # Add the raw points.
    for x_val in x:
        sub_y = ROC_between_points[x_val]
        if x_val == 1:
            x_val += STARTPOINT_SHIFTS
        sub_x = [x_val + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]
        plt.scatter(sub_x, sub_y, color="teal", alpha=A)

    sub_y = ROC_endpoints[0]
    sub_x = [1 + STARTPOINT_SHIFTS + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]
    plt.scatter(sub_x, sub_y, color="brown", alpha=A)
    sub_y = ROC_endpoints[1]
    sub_x = [x_end[0] + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]
    plt.scatter(sub_x, sub_y, color="orange", alpha=A)

    plt.title("Maximum Possible Link Prediction ROC Scores\nfor %s Graph with 10%% Missing Edges" % graph_name)
    plt.xlabel("number of hops (\"k\") of information")
    plt.ylabel("ROC")
    plt.xticks(x)
    margin = 0.05
    plt.ylim([-margin, 1 + margin])
    plt.legend()
    plt.show()

    plt.close()

    # ... now for AUPR

    STARTPOINT_SHIFTS = SPS_BASE * AUPR_max_k

    x = [i for i in range(1, AUPR_max_k + 1)]
    x_plotted = list(x)
    x_plotted[0] = x_plotted[0] + STARTPOINT_SHIFTS
    y = [AUPR_avg_between_points[i][0] for i in x]
    yerr = [AUPR_avg_between_points[i][1] for i in x]
    plt.errorbar(x_plotted, y, yerr=yerr, color="teal", linewidth=LW, label="nearly exact")
    x_start = [x[0] - STARTPOINT_SHIFTS]
    y_start = [AUPR_avg_endpoints[0][0]]
    yerr_start = [AUPR_avg_endpoints[0][1]]
    plt.errorbar(x_start, y_start, yerr=yerr_start, color="brown", linewidth=LW, label="k = 1")
    x_end = [x[-1] + ENDPOINT_SHIFT]
    y_end = [AUPR_avg_endpoints[1][0]]
    yerr_end = [AUPR_avg_endpoints[1][1]]
    plt.errorbar(x_end, y_end, yerr=yerr_end, color="orange", linewidth=LW, label="k = inf")

    # Add the raw points.
    for x_val in x:
        sub_y = AUPR_between_points[x_val]
        if x_val == 1:
            x_val += STARTPOINT_SHIFTS
        sub_x = [x_val + 0.05 * (random.random() - 0.5) for _ in sub_y]
        plt.scatter(sub_x, sub_y, color="teal", alpha=A)

    sub_y = AUPR_endpoints[0]
    sub_x = [1 + STARTPOINT_SHIFTS + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]
    plt.scatter(sub_x, sub_y, color="brown", alpha=A)
    sub_y = AUPR_endpoints[1]
    sub_x = [x_end[0] + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]
    plt.scatter(sub_x, sub_y, color="orange", alpha=A)

    plt.title("Maximum Possible Link Prediction AUPR Scores\nfor %s Graph with 10%% Missing Edges" % graph_name)
    plt.xlabel("number of hops (\"k\") of information")
    plt.ylabel("AUPR")
    plt.xticks(x)
    margin = 0.05
    plt.ylim([-margin, 1 + margin])
    plt.legend()
    plt.show()

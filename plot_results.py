import matplotlib.pyplot as plt
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
    counter = None
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

            counter = 0

        if counter is not None and counter == 6:
            value = float(l.strip().split(" ")[2])

            if phase == KINF:
                ROC_endpoints[1].append(value)
            elif phase == K1:
                ROC_endpoints[0].append(value)
            else:
                if k not in ROC_between_points:
                    ROC_between_points[k] = []
                ROC_between_points[k].append(value)

        if counter is not None and counter == 7:
            value = float(l.strip().split(" ")[2])

            if phase == KINF:
                AUPR_endpoints[1].append(value)
            elif phase == K1:
                AUPR_endpoints[0].append(value)
            else:
                if k not in AUPR_between_points:
                    AUPR_between_points[k] = []
                AUPR_between_points[k].append(value)

            counter = None

        if counter is not None:
            counter += 1

    #################### Get Means & Stdevs ####################

    ROC_between_points = {k: (sum(l) / float(len(l)), statistics.pstdev(l)) \
                            for k, l in ROC_between_points.items()}
    AUPR_between_points = {k: (sum(l) / float(len(l)), statistics.pstdev(l)) \
                            for k, l in AUPR_between_points.items()}

    ROC_max_k = max([k for k, l in ROC_between_points.items()])
    AUPR_max_k = max([k for k, l in AUPR_between_points.items()])

    ROC_endpoints = [(sum(l) / float(len(l)), statistics.pstdev(l)) \
                        for l in ROC_endpoints]
    AUPR_endpoints = [(sum(l) / float(len(l)), statistics.pstdev(l)) \
                        for l in AUPR_endpoints]

    #################### Create Plots ####################

    x = [i for i in range(1, ROC_max_k + 1)]
    y = [ROC_between_points[i][0] for i in x]
    yerr = [ROC_between_points[i][1] for i in x]
    plt.errorbar(x, y, yerr=yerr, color="blue", label="nearly exact")
    x_start = [x[0]]
    y_start = [ROC_endpoints[0][0]]
    yerr_start = [ROC_endpoints[0][1]]
    plt.errorbar(x_start, y_start, yerr=yerr_start, color="red", label="k = 1")
    x_end = [x[1]]
    y_end = [ROC_endpoints[1][0]]
    yerr_end = [ROC_endpoints[1][1]]
    plt.errorbar(x_end, y_end, yerr=yerr_end, color="orange", label="k = inf")

    plt.title("Maximum Possible Link Prediction ROC Scores\nfor %s Graph with 10%% Missing Edges" % graph_name)
    plt.xlabel("number of hops (\"k\") of information")
    plt.ylabel("ROC")
    plt.xticks(x)
    margin = 0.05
    plt.ylim([-margin, 1 + margin])
    plt.legend()
    plt.show()

    plt.close()

    x = [i for i in range(1, AUPR_max_k + 1)]
    y = [AUPR_between_points[i][0] for i in x]
    yerr = [AUPR_between_points[i][1] for i in x]
    plt.errorbar(x, y, yerr=yerr, color="blue", label="nearly exact")
    x_start = [x[0]]
    y_start = [AUPR_endpoints[0][0]]
    yerr_start = [AUPR_endpoints[0][1]]
    plt.errorbar(x_start, y_start, yerr=yerr_start, color="red", label="k = 1")
    x_end = [x[1]]
    y_end = [AUPR_endpoints[1][0]]
    yerr_end = [AUPR_endpoints[1][1]]
    plt.errorbar(x_end, y_end, yerr=yerr_end, color="orange", label="k = inf")

    plt.title("Maximum Possible Link Prediction AUPR Scores\nfor %s Graph with 10%% Missing Edges" % graph_name)
    plt.xlabel("number of hops (\"k\") of information")
    plt.ylabel("AUPR")
    plt.xticks(x)
    margin = 0.05
    plt.ylim([-margin, 1 + margin])
    plt.legend()
    plt.show()

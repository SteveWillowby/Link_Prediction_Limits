import matplotlib.pyplot as plt
import random
import statistics
import sys

if __name__ == "__main__":

    #################### Read the Results ####################

    filename = sys.argv[1]
    plot_name = filename.split("/")[1].split(".")[0]
    graph_name = plot_name.split("_")[0]

    table_dir = "plots/data/"

    ROC_endpoints = [[], []]
    ROC_between_points = []
    AUPR_endpoints = [[], []]
    AUPR_between_points = []
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
                ROC_between_points.append([])
            else:
                ROC_between_points[-1].append(value)

        if searching and " AUPR: " in l:
            value = float(l.strip().split(" ")[2])

            if phase == KINF:
                AUPR_endpoints[1].append(value)
            elif phase == K1:
                AUPR_endpoints[0].append(value)
                AUPR_between_points.append([])
            else:
                AUPR_between_points[-1].append(value)

            searching = False

    #################### Extend Out Values ####################

    ROC_max_k = max([len(l) for l in ROC_between_points])
    AUPR_max_k = max([len(l) for l in AUPR_between_points])
    assert ROC_max_k == AUPR_max_k

    for l in ROC_between_points:
        while len(l) < ROC_max_k:
            l.append(l[-1])
    for l in AUPR_between_points:
        while len(l) < AUPR_max_k:
            l.append(l[-1])

    # for l in ROC_between_points:
    #     for i in range(0, len(l) - 1):
    #         assert l[0] <= l[i + 1]
    # for l in AUPR_between_points:  # only true if 100% non-edges used
    #     for i in range(0, len(l) - 1):
    #         assert l[0] <= l[i + 1]

    ################## Convert to Dictionaries #################

    ROC_between_points = {i + 1: [l[i] for l in ROC_between_points] \
                            for i in range(0, ROC_max_k)}
    AUPR_between_points = {i + 1: [l[i] for l in AUPR_between_points] \
                            for i in range(0, AUPR_max_k)}

    #################### Get Means & Stdevs ####################

    ROC_avg_between_points = {k: (sum(l) / float(len(l)), statistics.pstdev(l)) \
                               for k, l in ROC_between_points.items()}
    AUPR_avg_between_points = {k: (sum(l) / float(len(l)), statistics.pstdev(l)) \
                               for k, l in AUPR_between_points.items()}

    ROC_avg_endpoints = [(sum(l) / float(len(l)), statistics.pstdev(l)) \
                          for l in ROC_endpoints]
    AUPR_avg_endpoints = [(sum(l) / float(len(l)), statistics.pstdev(l)) \
                           for l in AUPR_endpoints]

    #################### Create Plots ####################

    ############## Also Export Values to pgfplots Tables ############

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

    avg_points_table_file = table_dir + plot_name + "_AUROC_average_points.tex"
    f1 = open(avg_points_table_file, "w")
    f1.write("\\pgfplotstableread{\n")
    f1.write("k\tAUROC\tstdev\n")
    # For now, skip the exact points since the hashed ones are good enough.
    # f1.write("%f\t%f\t%f\n" % (x_start[0], y_start[0], yerr_start[0]))
    for i in range(0, len(x_plotted)):
        f1.write("%f\t%f\t%f\n" % (x_plotted[i], y[i], yerr[i]))
    f1.write("%f\t%f\t%f\n" % (x_end[0], y_end[0], yerr_end[0]))
    f1.write("}{\\" + plot_name + "_AUROC_average_points}")
    f1.close()

    scatterpoint_table_file = table_dir + plot_name + "_AUROC_between_points.tex"
    f1 = open(scatterpoint_table_file, "w")
    f1.write("\\pgfplotstableread{\n")
    f1.write("k\tAUROC\n")

    # Add the raw points.
    for x_val in x:
        sub_y = ROC_between_points[x_val]
        if x_val == 1:
            x_val += STARTPOINT_SHIFTS
        sub_x = [x_val + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]
        plt.scatter(sub_x, sub_y, color="teal", alpha=A)

        for i in range(0, len(sub_y)):
            f1.write("%f\t%f\n" % (sub_x[i], sub_y[i]))

    f1.write("}{\\" + plot_name + "_AUROC_between_points}")
    f1.close()

    sub_y = ROC_endpoints[0]
    sub_x = [1 + STARTPOINT_SHIFTS + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]

    scatterpoint_table_file = table_dir + plot_name + "_AUROC_start_points.tex"
    f1 = open(scatterpoint_table_file, "w")
    f1.write("\\pgfplotstableread{\n")
    f1.write("k\tAUROC\n")
    for i in range(0, len(sub_y)):
        f1.write("%f\t%f\n" % (sub_x[i], sub_y[i]))
    f1.write("}{\\" + plot_name + "_AUROC_start_points}")
    f1.close()

    plt.scatter(sub_x, sub_y, color="brown", alpha=A)

    sub_y = ROC_endpoints[1]
    sub_x = [x_end[0] + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]

    scatterpoint_table_file = table_dir + plot_name + "_AUROC_end_points.tex"
    f1 = open(scatterpoint_table_file, "w")
    f1.write("\\pgfplotstableread{\n")
    f1.write("k\tAUROC\n")
    for i in range(0, len(sub_y)):
        f1.write("%f\t%f\n" % (sub_x[i], sub_y[i]))
    f1.write("}{\\" + plot_name + "_AUROC_end_points}")
    f1.close()

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

    avg_points_table_file = table_dir + plot_name + "_AUPR_average_points.tex"
    f1 = open(avg_points_table_file, "w")
    f1.write("\\pgfplotstableread{\n")
    f1.write("k\tAUPR\tstdev\n")
    # For now, skip the exact points since the hashed ones are good enough.
    # f1.write("%f\t%f\t%f\n" % (x_start[0], y_start[0], yerr_start[0]))
    for i in range(0, len(x_plotted)):
        f1.write("%f\t%f\t%f\n" % (x_plotted[i], y[i], yerr[i]))
    f1.write("%f\t%f\t%f\n" % (x_end[0], y_end[0], yerr_end[0]))
    f1.write("}{\\" + plot_name + "_AUPR_average_points}")
    f1.close()

    scatterpoint_table_file = table_dir + plot_name + "_AUPR_between_points.tex"
    f1 = open(scatterpoint_table_file, "w")
    f1.write("\\pgfplotstableread{\n")
    f1.write("k\tAUPR\n")

    # Add the raw points.
    for x_val in x:
        sub_y = AUPR_between_points[x_val]
        if x_val == 1:
            x_val += STARTPOINT_SHIFTS
        sub_x = [x_val + 0.05 * (random.random() - 0.5) for _ in sub_y]
        plt.scatter(sub_x, sub_y, color="teal", alpha=A)

        for i in range(0, len(sub_y)):
            f1.write("%f\t%f\n" % (sub_x[i], sub_y[i]))

    f1.write("}{\\" + plot_name + "_AUPR_between_points}")
    f1.close()

    sub_y = AUPR_endpoints[0]
    sub_x = [1 + STARTPOINT_SHIFTS + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]

    scatterpoint_table_file = table_dir + plot_name + "_AUPR_start_points.tex"
    f1 = open(scatterpoint_table_file, "w")
    f1.write("\\pgfplotstableread{\n")
    f1.write("k\tAUPR\n")
    for i in range(0, len(sub_y)):
        f1.write("%f\t%f\n" % (sub_x[i], sub_y[i]))
    f1.write("}{\\" + plot_name + "_AUPR_start_points}")
    f1.close()

    plt.scatter(sub_x, sub_y, color="brown", alpha=A)
    sub_y = AUPR_endpoints[1]
    sub_x = [x_end[0] + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]

    scatterpoint_table_file = table_dir + plot_name + "_AUPR_end_points.tex"
    f1 = open(scatterpoint_table_file, "w")
    f1.write("\\pgfplotstableread{\n")
    f1.write("k\tAUPR\n")
    for i in range(0, len(sub_y)):
        f1.write("%f\t%f\n" % (sub_x[i], sub_y[i]))
    f1.write("}{\\" + plot_name + "_AUPR_end_points}")
    f1.close()

    plt.scatter(sub_x, sub_y, color="orange", alpha=A)

    plt.title("Maximum Possible Link Prediction AUPR Scores\nfor %s Graph with 10%% Missing Edges" % graph_name)
    plt.xlabel("number of hops (\"k\") of information")
    plt.ylabel("AUPR")
    plt.xticks(x)
    margin = 0.05
    plt.ylim([-margin, 1 + margin])
    plt.legend()
    plt.show()

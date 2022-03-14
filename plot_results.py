import math
import matplotlib.pyplot as plt
from max_score_functions import get_max_AUPR, get_max_ROC, __manual_AUPR_checker__
import random
import statistics
import sys

# Example calls:
#
# python3 plot_results.py test_results/powergrid_k-all_....txt
#
# python3 plot_results.py test_results/ER_513_512_k-all_....txt 4096

def read_results_file(filename):

    AUPR_MAX_FUNC = get_max_AUPR

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
    observed_T = None
    for l in lines:
        if len(l) > 11 and l[:11] == "observed_T=":
            observed_T = int(l.strip()[11:])

        if len(l) > 2 and "k=" == l[:2]:
            k = l.strip()[2:]
            if k != "inf":
                k = int(k)

            if phase == KINF:
                phase = K1

            elif phase == K1:
                phase = KBET

            elif type(k) is str and k == "inf":
                phase = KINF

        if len(l) > 12 and l[:12] == "raw_classes=":
            l = l.strip()[14:-2].split("), (")
            l = [x.split(", ") for x in l]
            l = [(int(x[0][1:]), int(x[1][:-1]), int(x[2]), int(x[3])) for x in l]
            class_info = []
            for (t, p, occ, _) in l:
                if p > 0:
                    for __ in range(0, occ):
                        class_info.append((t, p))

            ROC_value = get_max_ROC(class_info, observed_edges=observed_T)
            AUPR_value = AUPR_MAX_FUNC(class_info)

            if phase == KINF:
                ROC_endpoints[1].append(ROC_value)
                AUPR_endpoints[1].append(AUPR_value)
            elif phase == K1:
                ROC_endpoints[0].append(ROC_value)
                ROC_between_points.append([])
                AUPR_endpoints[0].append(AUPR_value)
                AUPR_between_points.append([])
            else:
                ROC_between_points[-1].append(ROC_value)
                AUPR_between_points[-1].append(AUPR_value)

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

    ################ Get Means & Conf. Interval ################

    CI_constant = 1.96  # For 95% confidence interval

    ROC_avg_between_points = {k: (sum(l) / float(len(l)), \
                              (2.0 * CI_constant * statistics.pstdev(l)) / \
                                math.sqrt(float(len(l)))) \
                               for k, l in ROC_between_points.items()}
    AUPR_avg_between_points = {k: (sum(l) / float(len(l)), \
                               (2.0 * CI_constant * statistics.pstdev(l)) / \
                                math.sqrt(float(len(l)))) \
                               for k, l in AUPR_between_points.items()}

    ROC_avg_endpoints = [(sum(l) / float(len(l)), \
                            (2.0 * CI_constant * statistics.pstdev(l)) / \
                             math.sqrt(float(len(l)))) \
                          for l in ROC_endpoints]
    AUPR_avg_endpoints = [(sum(l) / float(len(l)), \
                            (2.0 * CI_constant * statistics.pstdev(l)) / \
                             math.sqrt(float(len(l)))) \
                           for l in AUPR_endpoints]

    return (ROC_max_k, AUPR_max_k, \
            ROC_endpoints, ROC_between_points, \
            AUPR_endpoints, AUPR_between_points, \
            ROC_avg_endpoints, ROC_avg_between_points, \
            AUPR_avg_endpoints, AUPR_avg_between_points)

def basic_plots(filename):

    plot_name = filename.split("/")[1]
    pn = plot_name.split(".")
    plot_name = plot_name[:-len(pn[-1])]

    graph_name = plot_name.split("_k-")[0]
    frac_missing_edges = plot_name.split("_ref-")[1].split("_")[0]
    frac_missing_edges = int(float(frac_missing_edges) * 100.0)

    table_dir = "plots/data/"

    (ROC_max_k, AUPR_max_k, \
       ROC_endpoints, ROC_between_points, \
       AUPR_endpoints, AUPR_between_points, \
       ROC_avg_endpoints, ROC_avg_between_points, \
       AUPR_avg_endpoints,AUPR_avg_between_points) = read_results_file(filename)

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
    f1.write("k\tAUROC\tconfintwidth\tlabel\n")
    f1.write("%f\t%f\t%f\t%d\n" % (x_start[0], y_start[0], yerr_start[0], 0))
    for i in range(0, len(x_plotted)):
        f1.write("%f\t%f\t%f\t%d\n" % (x_plotted[i], y[i], yerr[i], 1))
    f1.write("%f\t%f\t%f\t%d\n" % (x_end[0], y_end[0], yerr_end[0], 2))
    f1.write("}{\\" + plot_name.replace("_", "").replace("-", "") + \
                        "AUROCaveragepoints}")
    f1.close()

    scatterpoint_table_file = table_dir + plot_name + "_AUROC_raw_points.tex"
    f1 = open(scatterpoint_table_file, "w")
    f1.write("\\pgfplotstableread{\n")
    f1.write("k\tAUROC\tlabel\n")

    sub_y = ROC_endpoints[0]
    sub_x = [1 + STARTPOINT_SHIFTS + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]

    for i in range(0, len(sub_y)):
        f1.write("%f\t%f\t%d\n" % (sub_x[i], sub_y[i], 0))

    plt.scatter(sub_x, sub_y, color="brown", alpha=A)

    # Add the raw points.
    for x_val in x:
        sub_y = ROC_between_points[x_val]
        if x_val == 1:
            x_val += STARTPOINT_SHIFTS
        sub_x = [x_val + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]
        plt.scatter(sub_x, sub_y, color="teal", alpha=A)

        for i in range(0, len(sub_y)):
            f1.write("%f\t%f\t%d\n" % (sub_x[i], sub_y[i], 1))

    sub_y = ROC_endpoints[1]
    sub_x = [x_end[0] + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]

    for i in range(0, len(sub_y)):
        f1.write("%f\t%f\t%d\n" % (sub_x[i], sub_y[i], 2))

    plt.scatter(sub_x, sub_y, color="orange", alpha=A)

    f1.write("}{\\" + plot_name.replace("_", "").replace("-", "") + \
                        "AUROCrawpoints}")
    f1.close()

    plt.title("Maximum Possible Link Prediction ROC Scores\n" + \
              "for %s Graph with %d%% Missing Edges" % \
                (graph_name, frac_missing_edges))

    plt.xlabel("number of hops (\"k\") of information")
    plt.ylabel("AUROC")
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
    f1.write("k\tAUPR\tconfintwidth\tlabel\n")
    f1.write("%f\t%f\t%f\t%d\n" % (x_start[0], y_start[0], yerr_start[0], 0))
    for i in range(0, len(x_plotted)):
        f1.write("%f\t%f\t%f\t%d\n" % (x_plotted[i], y[i], yerr[i], 1))
    f1.write("%f\t%f\t%f\t%d\n" % (x_end[0], y_end[0], yerr_end[0], 2))
    f1.write("}{\\" + plot_name.replace("_", "").replace("-", "") + \
                        "AUPRaveragepoints}")
    f1.close()

    scatterpoint_table_file = table_dir + plot_name + "_AUPR_raw_points.tex"
    f1 = open(scatterpoint_table_file, "w")
    f1.write("\\pgfplotstableread{\n")
    f1.write("k\tAUPR\tlabel\n")

    sub_y = AUPR_endpoints[0]
    sub_x = [1 + STARTPOINT_SHIFTS + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]

    for i in range(0, len(sub_y)):
        f1.write("%f\t%f\t%d\n" % (sub_x[i], sub_y[i], 0))

    plt.scatter(sub_x, sub_y, color="brown", alpha=A)

    # Add the raw points.
    for x_val in x:
        sub_y = AUPR_between_points[x_val]
        if x_val == 1:
            x_val += STARTPOINT_SHIFTS
        sub_x = [x_val + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]
        plt.scatter(sub_x, sub_y, color="teal", alpha=A)

        for i in range(0, len(sub_y)):
            f1.write("%f\t%f\t%d\n" % (sub_x[i], sub_y[i], 1))

    sub_y = AUPR_endpoints[1]
    sub_x = [x_end[0] + JITTER_WIDTH * (random.random() - 0.5) for _ in sub_y]

    for i in range(0, len(sub_y)):
        f1.write("%f\t%f\t%d\n" % (sub_x[i], sub_y[i], 2))

    plt.scatter(sub_x, sub_y, color="orange", alpha=A)

    f1.write("}{\\" + plot_name.replace("_", "").replace("-", "") + \
                        "AUPRrawpoints}")
    f1.close()

    plt.title("Maximum Possible Link Prediction AUPR Scores\n" + \
              "for %s Graph with %d%% Missing Edges" % \
                (graph_name, frac_missing_edges))
    plt.xlabel("number of hops (\"k\") of information")
    plt.ylabel("AUPR")
    plt.xticks(x)
    margin = 0.05
    plt.ylim([-margin, 1 + margin])
    plt.legend()
    plt.show()

def str_patch(str_arr, connector):
    s = str_arr[0]
    for i in range(1, len(str_arr)):
        s += connector
        s += str_arr[i]
    return s

def ER_progression_plot(first_filename, subsequent_Ms):

    base_dir = str_patch(first_filename.split("/")[:-1], "/")
    plot_name = first_filename.split("/")[-1]
    file_extension = "." + plot_name.split(".")[-1]
    plot_name = str_patch(plot_name.split(".")[:-1], ".")

    _split = plot_name.split("_")
    start_M = int(_split[2])
    N = int(_split[1])
    plot_name_beginning = str_patch(_split[:2], "_")
    plot_name_ending = str_patch(_split[3:], "_")

    # graph_name = plot_name.split("_k-")[0]
    frac_missing_edges = plot_name.split("_ref-")[1].split("_")[0]
    frac_missing_edges = int(float(frac_missing_edges) * 100.0)

    table_dir = "plots/data/"

    max_ks = []
    avg_endpoints = []
    avg_between_points = []

    m_values = [start_M] + subsequent_Ms

    log_scale = False
    if len(m_values) > 2:
        l = float(m_values[-1])
        ll = float(m_values[-2])
        lll = float(m_values[-3])

        # If dividing makes the gaps closer to relatively equal (i.e. closer to
        #   1.0) than subtracting does, then assume it's log scale.
        if abs(1.0 - (ll / lll) / (l / ll)) < \
                abs(1.0 - (ll - lll) / (l - ll)):
            log_scale = True
    print("Log Scale? %s" % log_scale)

    for M in m_values:
        filename = base_dir + "/" + \
            str_patch([plot_name_beginning, str(M), plot_name_ending], "_") + \
            file_extension

        print("Loading %s" % filename)

        (ROC_max_k, AUPR_max_k, \
           ROC_endpoints, ROC_between_points, \
           AUPR_endpoints, AUPR_between_points, \
           ROC_avg_endpoints, ROC_avg_between_points, \
           AUPR_avg_endpoints,AUPR_avg_between_points) = \
                            read_results_file(filename)

        max_ks.append(AUPR_max_k)
        avg_endpoints.append(AUPR_avg_endpoints)
        avg_between_points.append(AUPR_avg_between_points)

    max_k = max(max_ks)
    points_by_k = [[] for _ in range(0, max_k + 2)]

    for i in range(0, len(m_values)):
        sub_max_k = max_ks[i]
        sub_avg_ends = avg_endpoints[i]
        sub_avg_middles = avg_between_points[i]

        for k in range(1, sub_max_k + 1):
            points_by_k[k].append((m_values[i], sub_avg_middles[k][0], \
                                                sub_avg_middles[k][1]))

        points_by_k[max_k + 1].append((m_values[i], sub_avg_ends[1][0], \
                                                    sub_avg_ends[1][1]))

    color_wheel = ["", "red", "blue", "teal", "brown"] + \
                    ["black" for _ in range(4, max_k)] + ["orange"]
    legend = [""] + ["k = %d" % k for k in range(1, max_k + 1)] + ["k = inf"]

    LW = 2
    NUM_K_TO_SHOW = 7

    for k in [v for v in range(1, min(NUM_K_TO_SHOW, max_k + 1))] + [max_k + 1]:
        if log_scale:
            x_axis = [int(math.log2(m)) for (m, _, __) in points_by_k[k]]
        else:
            x_axis = [m for (m, _, __) in points_by_k[k]]

        y_axis = [y for (_, y, __) in points_by_k[k]]
        y_err = [ye for (_, __, ye) in points_by_k[k]]
        color = color_wheel[k]
        label = legend[k]
        plt.errorbar(x_axis, y_axis, yerr=y_err, color=color, linewidth=LW, label=label)

    plt.title("Maximum Possible Link Prediction AUPR Scores for\n" + \
              "Erdos-Renyi Graphs on %d Nodes with %d%% Missing Edges" % \
                (N, frac_missing_edges))

    plt.ylabel("AUPR Max")
    if log_scale:
        plt.xticks([int(math.log2(m)) for m in m_values])
        plt.xlabel("Log2 of the Number of Edges")
    else:
        plt.xticks(m_values)
        plt.xlabel("Number of Edges")
    margin = 0.05
    plt.ylim([-margin, 1 + margin])
    plt.legend()
    plt.show()

if __name__ == "__main__":

    assert len(sys.argv) > 1

    argv = sys.argv[1:]
    if len(argv) == 1:
        filename = argv[0]
        basic_plots(filename)
        exit(0)

    first_filename = argv[0]
    further_edge_numbers = [int(v) for v in argv[1:]]
    assert "ER_" in first_filename

    ER_progression_plot(first_filename, further_edge_numbers)

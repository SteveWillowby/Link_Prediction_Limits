import math
import random
import statistics

# Functions:
#   get_max_AUPR(class_info)
#   get_max_ROC(class_info)
#
# where `class_info` is a collection of triples:
#   (class_label, class_size, num_positives_in_class)

# Removes classes without positives, sorts classes by density, and lumps classes
#   of equal density together.
def refine_class_info(class_info):

    refined_class_info = []
    for (t, p) in class_info:
        if p > 0:
            refined_class_info.append((t, p))
    class_info = refined_class_info

    class_info = [(float(x[0]) / x[1], x[1], x[0]) for x in class_info]
    class_info.sort()
    refined_class_info = [[class_info[0][2], class_info[0][1]]]
    for i in range(1, len(class_info)):
        prev_t = refined_class_info[-1][0]
        prev_p = refined_class_info[-1][1]
        (_, p, t) = class_info[i]
        if p * prev_t == t * prev_p:
            # Ratios are the same.
            refined_class_info[-1][0] += t
            refined_class_info[-1][1] += p
        else:
            refined_class_info.append([t, p])
    return refined_class_info

def get_max_AUPR(class_info, mention_errors=True):

    if len(class_info) == 0:
        return 1.0  # TODO: should this be zero instead of one?

    class_info = refine_class_info(class_info)

    return get_AUPR(class_info, mention_errors=mention_errors)

def get_AUPR(class_info, mention_errors=True):
    P = sum([x[1] for x in class_info])
    T = sum([x[0] for x in class_info])
    N = T - P

    b = 0.0
    d = 0.0
    AUPR = 0.0
    for (c, a) in class_info:
        a = float(a)
        c = float(c)
        if a == 0.0:
            addition = 0.0

        # Precision of this specific class is ((a + b) / (c + d)).
        # The width of this slice (amount of recall) is a / P.
        #   Since we divide by P at the end, the area of the "box" part is
        #   a * ((a + b) / (c + d)). In the very first round this is just
        #   (a * a) / c.

        elif d == 0.0:
            # Only occurs once. No previous point, so the only area is the box.
            addition = (a * a) / c
        else:
            addition = ((a * a) / c) * (1.0 + ((b / a) - (d / c)) * math.log((d + c) / d))
            # TODO: Look into the meaning of this assertion margin.
            assert_margin = 0.0001
            if addition + assert_margin < (a * a) / c and mention_errors:
                print("Err1 -- [for context: math.log(%f) = %f]" % ((d + c) / d, math.log((d + c) / d)))
            if addition - assert_margin > a * (((a + b) / (c + d)) + (b / c)) / 2.0 \
                    and mention_errors:
                print("Err2 -- [for context: math.log(%f) = %f]" % ((d + c) / d, math.log((d + c) / d)))

        assert addition >= 0.0
        # print("a: %f, b: %f, c: %f, d: %f -----> %f" % (a, b, c, d, addition))
        AUPR += addition

        b += a
        d += c

    assert P == int(b + 0.1)  # The +0.1 is just to avoid rounding errors.

    AUPR /= float(P)
    return AUPR

def get_max_ROC(class_info, observed_edges):

    if len(class_info) == 0:
        return 1.0  # TODO: should this be zero instead of one?

    class_info = refine_class_info(class_info)

    return get_ROC(class_info, observed_edges)

def get_ROC(class_info, observed_edges):

    P = sum([x[1] for x in class_info])  # P is the same as "observed P"
    observed_N = observed_edges - P
    assert P > 0
    n_acc = 0
    p_acc = 0
    # I chose to add the corners.
    TPR = [0.0]  # Goes up from 0 to 1
    FPR = [0.0]  # Goes up from 0 to 1
    for (t, p) in class_info:
        p_acc += p
        n_acc += t - p

        TPR.append(float(p_acc) / P)

        if observed_N == 0:
            FPR.append(0.0)
        else:
            FPR.append(float(n_acc) / observed_N)

    # I chose to add the corners.
    TPR.append(1.0)
    FPR.append(1.0)
    ROC = 0.0
    for i in range(1, len(TPR)):
        tpr_a = TPR[i - 1]
        tpr_b = TPR[i]
        fpr_a = FPR[i - 1]
        fpr_b = FPR[i]
        addition = (fpr_b - fpr_a) * ((tpr_a + tpr_b) / 2.0)
        assert addition >= 0.0
        ROC += addition
    return ROC

def estimate_min_frac_for_AUPR(class_info, desired_stdev):

    refined_class_info = []
    for (t, p) in class_info:
        if p > 0:
            refined_class_info.append((t, p))
    class_info = refined_class_info

    FULL_AUPR = get_max_AUPR(class_info)

    MARGIN_EXP = 10
    ITERATIONS = 100
    PROB_A_TRUE_IS_INCLUDED = 0.99999
    PROB_NO_TRUE_IS_INCLUDED = 1.0 - PROB_A_TRUE_IS_INCLUDED

    T = sum([x[0] for x in class_info])

    # Set min_frac so that the probability ALL T elements are excluded is
    #   <= PROB_NO_TRUE_IS_INCLUDED
    #
    # I.e.
    #   (1.0 - min_frac)^T <= PROB_NO_TRUE_IS_INCLUDED
    #   -->
    #       log_2(1.0 - min_frac) <= log_2(PROB_NO_TRUE_IS_INCLUDED) / T
    #   -->
    #       1.0 - min_frac <= 2.0 ^ (log_2(PROB_NO_TRUE_IS_INCLUDED) / T)
    #   -->
    #       min_frac >= 1.0 - 2.0 ^ (log_2(PROB_NO_TRUE_IS_INCLUDED) / T)
    min_frac = 1.0 - 2.0 ** (math.log2(PROB_NO_TRUE_IS_INCLUDED) / float(T))
    # print("Initial min_frac: %f" % min_frac)

    if min_frac >= 1.0:
        print("Unsuccessful min_frac calculation -- " + \
                "got 1 or more rather than a small number.")
        return 1.0

    if min_frac <= 0.0:
        print("Unsuccessful min_frac calculation -- got %f" % min_frac)
        min_frac = 0.5 ** MARGIN_EXP

    max_frac = 1.0

    FRAC_MARGIN = 0.5 ** MARGIN_EXP

    while (max_frac - min_frac) > FRAC_MARGIN:
        frac = (min_frac + max_frac) / 2.0

        AUPR_values = []
        for _ in range(0, ITERATIONS):
            fake_class_info = []
            for (t, p) in class_info:
                fake_n = 0
                fake_p = 0
                n = t - p
                for __ in range(0, p):
                    if random.random() < frac:
                        fake_p += 1
                if fake_p == 0:
                    continue
                for __ in range(0, n):
                    if random.random() < frac:
                        fake_n += 1
                fake_class_info.append((fake_p + fake_n, fake_p))

            if len(fake_class_info) == 0:
                print(("Error! A 1-in-%f event " % ((1.0 / (1.0 - frac))**T)) +\
                        "has occurred, or we have FP errors -- or both.")

            AUPR_values.append(get_max_AUPR(fake_class_info, \
                                            mention_errors=False))

        average = sum(AUPR_values) / float(len(AUPR_values))
        stdev = statistics.stdev(AUPR_values)

        if stdev <= desired_stdev and \
                abs(FULL_AUPR - average) <= desired_stdev:
            max_frac = frac
        else:
            min_frac = frac

    return (max_frac + min_frac) / 2.0

def __manual_AUPR_checker__(class_info):
    STEPS = 10000 + 1

    class_info = [(float(x[0]) / x[1], x[1], x[0]) for x in class_info]
    class_info.sort()
    class_info = [(x[1], x[2]) for x in class_info]  # Positives, Total Size
    P = sum([x[0] for x in class_info])
    # T = sum([x[1] for x in class_info])
    AUPR = 0.0
    p_sum = 0
    t_sum = 0
    for (p, t) in class_info:
        precision_sum = 0.0
        for i in range(0, STEPS):
            # Change in alpha is linear with change in recall.
            alpha = float(i) / (STEPS - 1)
            if i == 0 and t_sum == 0:
                precision = float(p) / t
            else:
                precision = (p_sum + alpha * p) / (t_sum + alpha * t)
            precision_sum += precision
        p_sum += p
        t_sum += t
        avg_precision = precision_sum / STEPS
        change_in_recall = float(p) / P
        AUPR += avg_precision * change_in_recall
    return AUPR

if __name__ == "__main__":
    test_class_info = [(5, 4), (7, 2), (10, 10)]
    print("Error: %f" % (__manual_AUPR_checker__(test_class_info) - \
                get_max_AUPR(test_class_info)))
    print(estimate_min_frac_for_AUPR(test_class_info, desired_stdev=0.01))

    test_class_info = [(500, 400), (700, 200), \
                       (10, 10), (1000, 10)]
    print("Error: %f" % (__manual_AUPR_checker__(test_class_info) - \
                get_max_AUPR(test_class_info)))
    print(estimate_min_frac_for_AUPR(test_class_info, desired_stdev=0.01))

    test_class_info = [(500, 40), (700, 20), \
                       (10, 2), (1000, 10)]
    print("Error: %f" % (__manual_AUPR_checker__(test_class_info) - \
                get_max_AUPR(test_class_info)))
    print(estimate_min_frac_for_AUPR(test_class_info, desired_stdev=0.01))

import math

# Functions:
#   get_max_AUPR(class_info)
#   get_max_ROC(class_info)
#
# where `class_info` is a collection of triples:
#   (class_label, class_size, num_positives_in_class)

def get_max_AUPR(class_info):
    class_info = [(float(x[1]) / x[2], x[2], x[1]) for x in class_info]
    class_info.sort()
    class_info = [(x[1], x[2]) for x in class_info]  # Positives, Total Size
    P = sum([x[0] for x in class_info])
    T = sum([x[1] for x in class_info])
    N = T - P

    b = 0.0
    d = 0.0
    AUPR = 0.0
    for (a, c) in class_info:
        a = float(a)
        c = float(c)
        if a == 0.0:
            addition = 0.0
        elif d == 0.0:
            # Only occurs once.
            addition = (a * a) / c
        else:
            addition = ((a * a) / c) * (1.0 + ((b / a) - (d / c)) * math.log((d + c) / d))

        assert addition >= 0.0
        # print("a: %f, b: %f, c: %f, d: %f -----> %f" % (a, b, c, d, addition))
        AUPR += addition

        b += a
        d += c
    AUPR /= float(P)
    return AUPR

def get_max_ROC(class_info):
    class_info = [(float(x[1]) / x[2], x[2], x[1]) for x in class_info]
    class_info.sort()
    class_info = [(x[1], x[2]) for x in class_info]  # Positives, Total Size
    P = sum([x[0] for x in class_info])
    T = sum([x[1] for x in class_info])
    N = T - P
    print("T: %d, P: %d, N: %d" % (T, P, N))
    n_acc = 0
    p_acc = 0
    TPR = []  # Goes up from 0 to 1
    FPR = []  # Goes up from 0 to 1
    for (p, t) in class_info:
        p_acc += p
        n_acc += t - p

        if P == 0:
            print("WAT? P == 0 in get_max_ROC???")
            TPR.append(1.0)
        else:
            TPR.append(float(p_acc) / P)

        if N == 0:
            FPR.append(0.0)
        else:
            FPR.append(float(n_acc) / N)
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

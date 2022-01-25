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
            if addition + assert_margin < (a * a) / c:
                print("Err1 -- math.log(%f) = %f" % ((d + c) / d, math.log((d + c) / d)))
            if addition - assert_margin > a * (((a + b) / (c + d)) + (b / c)) / 2.0:
                print("Err2 -- math.log(%f) = %f" % ((d + c) / d, math.log((d + c) / d)))

        assert addition >= 0.0
        # print("a: %f, b: %f, c: %f, d: %f -----> %f" % (a, b, c, d, addition))
        AUPR += addition

        b += a
        d += c

    assert P == int(b + 0.1)  # The +0.1 is just to avoid rounding errors.

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
    assert P > 0
    n_acc = 0
    p_acc = 0
    # I chose to add the corners.
    TPR = [0.0]  # Goes up from 0 to 1
    FPR = [0.0]  # Goes up from 0 to 1
    for (p, t) in class_info:
        p_acc += p
        n_acc += t - p

        TPR.append(float(p_acc) / P)

        if N == 0:
            FPR.append(0.0)
        else:
            FPR.append(float(n_acc) / N)

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

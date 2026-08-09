import math
import numpy as np


def intra_quality(start, end, matrix):
    intra = 0
    sum = 0
    count = 0
    for i in range(start, end):
        for j in range(i + 1, end):
            count += 1
            sum += matrix[i, j]

    if sum > 0 and count > 0:
        intra = sum / count
    else:
        intra = 0

    return intra


def inter_quality(start1, end1, start2, end2, matrix):
    output = [0.0, 0]
    sum_val = 0
    count = 0
    incr = 0
    for i in range(start1, start2):
        incr += 1
        c = 0
        for j in range(end1, end2):
            c += 1
            count += 1
            sum_val += matrix[i, j]
            if c == incr:
                break
    output[0] = sum_val
    output[1] = count
    return output


def default_quality(tads, matrix):
    tad_count = len(tads)
    outinter1 = [0.0, 0.0]
    outinter2 = [0.0, 0.0]
    avg = 0.0
    sum = 0.0

    for j in range(tad_count):
        domain1_start = tads[j, 0]
        domain1_end = tads[j, 1]
        intra = intra_quality(domain1_start, domain1_end, matrix)
        inter = 0.0

        if j == 0 and tad_count > 1:
            domain2_start = tads[j + 1, 0]
            domain2_end = tads[j + 1, 1]
            outinter1 = inter_quality(
                domain1_start, domain1_end, domain2_start, domain2_end, matrix)
            if outinter1[1] == 0:
                outinter1[1] = 1e10
            inter = outinter1[0] / outinter1[1]
        elif 0 < j < tad_count - 1:
            domain2_start = tads[j - 1, 0]
            domain2_end = tads[j - 1, 1]
            outinter1 = inter_quality(
                domain2_start, domain2_end, domain1_start, domain1_end, matrix)
            domain2_start = tads[j + 1, 0]
            domain2_end = tads[j + 1, 1]
            outinter2 = inter_quality(
                domain1_start, domain1_end, domain2_start, domain2_end, matrix)
            if outinter1[1] == 0:
                outinter1[1] = 1e10
            if outinter2[1] == 0:
                outinter2[1] = 1e10
            inter = (outinter1[0] + outinter2[0]) / \
                (outinter1[1] + outinter2[1])
        elif j == tad_count - 1 and tad_count > 1:
            domain2_start = tads[j - 1, 0]
            domain2_end = tads[j - 1, 1]
            outinter1 = inter_quality(
                domain2_start, domain2_end, domain1_start, domain1_end, matrix)
            if outinter1[1] == 0:
                outinter1[1] = 1e10
            inter = outinter1[0] / outinter1[1]

        if np.isnan(intra):
            intra = 0.0
        if np.isnan(inter):
            inter = 0.0
        sum += (intra - inter)

    if tad_count == 0:
        avg = sum
    else:
        avg = round(sum / tad_count, 2)
    print(f"TAD Quality Score = {avg}")

    return avg


def caspian_quality(tads, matrix):
    n = len(matrix)
    intra = 0
    intra_num = 0
    for n in range(len(tads)):
        for i in range(int(tads[n, 0]), int(tads[n, 1])+1):
            for j in range(int(tads[n, 0]), int(tads[n, 1])+1):
                intra = intra + matrix[i, j]
                intra_num = intra_num + 1

    if intra_num != 0:
        intra = intra / intra_num
    else:
        intra = 0

    inter = 0
    inter_num = 0
    for n in range(len(tads) - 1):
        for i in range(int(tads[n, 0]), int(tads[n, 1])+1):
            for j in range(int(tads[n+1, 0]), int(tads[n+1, 1])+1):
                inter = inter + matrix[i, j]
                inter_num = inter_num + 1
    if inter_num != 0:
        inter = inter / inter_num
    else:
        inter = 0

    quality = round(intra - inter, 2)

    print(f"TAD Quality (Caspian) = {quality}")

    return quality


def get_tad_quality(tads, matrix, metric=None):
    if metric == "caspian":
        return caspian_quality(tads=tads, matrix=matrix)
    return default_quality(tads=tads, matrix=matrix)


def get_moc(tads, true_tads):
    moc = 0
    avg_moc = []
    for _, check_row in true_tads.iterrows():
        true_start = int(check_row[0])
        true_end = int(check_row[1])
        for _, row in tads.iterrows():
            ref_start = int(row[0])
            ref_end = int(row[1])
            if true_start < ref_end and true_end > ref_start:
                if true_end <= ref_end and true_start >= ref_start:
                    avg_moc.append(math.pow(true_end - true_start, 2) / (
                        (true_end - true_start) * (ref_end - ref_start)))
                elif true_end <= ref_end and true_start <= ref_start:
                    avg_moc.append(math.pow(true_end - ref_start, 2) / (
                        (true_end - true_start) * (ref_end - ref_start)))
                elif true_end >= ref_end and true_start >= ref_start:
                    avg_moc.append(math.pow(ref_end - true_start, 2) / (
                        (true_end - true_start) * (ref_end - ref_start)))
                else:
                    nu = (true_end - true_start) * (ref_end - ref_start)
                    avg_moc.append(math.pow(ref_end - ref_start,
                                   2) / nu if nu > 0 else .000001)
            else:
                avg_moc.append(0)
    if len(avg_moc) == 1 and avg_moc[0] > 0:
        moc = avg_moc[0]
    elif sum(avg_moc) <= 0:
        moc = .000001
    else:
        moc = sum(avg_moc) / (math.sqrt(len(avg_moc)) - 1)

    moc = round(moc*100, 2)
    print(f"MoC = {moc}")

    return moc

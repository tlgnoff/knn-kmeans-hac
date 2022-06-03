import numpy as np
import matplotlib.pyplot as plt
import math


def euclidean_dist(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def single_linkage(active_set):
    overall_min = np.inf
    set_size = len(active_set)
    for idx_g_1 in range(set_size):
        g_1 = active_set[idx_g_1]
        for idx_g_2 in range(set_size):
            g_2 = active_set[idx_g_2]
            if not all(np.allclose(a, b) for a, b in zip(g_1, g_2)):
                pair_min = np.inf
                for x in g_1:
                    for y in g_2:
                        dist = euclidean_dist(x, y)
                        if dist < pair_min:
                            pair_min = dist
                            if pair_min < overall_min:
                                overall_min = pair_min
                                res1, res2, idx_res1, idx_res2 = g_1, g_2, idx_g_1, idx_g_2
    return res1, res2, idx_res1, idx_res2


def complete_linkage(active_set):
    overall_min = np.inf
    set_size = len(active_set)
    for idx_g_1 in range(set_size):
        g_1 = active_set[idx_g_1]
        for idx_g_2 in range(set_size):
            g_2 = active_set[idx_g_2]
            if not all(np.allclose(a, b) for a, b in zip(g_1, g_2)):
                pair_max = 0.
                for x in g_1:
                    for y in g_2:
                        dist = euclidean_dist(x, y)
                        if dist > pair_max:
                            pair_max = dist
                            if pair_max < overall_min:
                                overall_min = pair_max
                                res1, res2, idx_res1, idx_res2 = g_1, g_2, idx_g_1, idx_g_2
    return res1, res2, idx_res1, idx_res2


def average_linkage(active_set):
    overall_min = np.inf
    set_size = len(active_set)
    for idx_g_1 in range(set_size):
        g_1 = active_set[idx_g_1]
        for idx_g_2 in range(set_size):
            g_2 = active_set[idx_g_2]
            if not all(np.allclose(a, b) for a, b in zip(g_1, g_2)):
                group_dist = 0.
                for x in g_1:
                    for y in g_2:
                        group_dist += euclidean_dist(x, y)
                group_dist /= len(g_1) * len(g_2)
                if group_dist < overall_min:
                    overall_min = group_dist
                    res1, res2, idx_res1, idx_res2 = g_1, g_2, idx_g_1, idx_g_2
    return res1, res2, idx_res1, idx_res2


def centroid(active_set):
    overall_min = np.inf
    set_size = len(active_set)
    for idx_g_1 in range(set_size):
        g_1 = active_set[idx_g_1]
        for idx_g_2 in range(set_size):
            g_2 = active_set[idx_g_2]
            if not all(np.allclose(a, b) for a, b in zip(g_1, g_2)):
                dist_centroid = euclidean_dist(np.mean(g_1, axis=0), np.mean(g_2, axis=0))
                if dist_centroid < overall_min:
                    overall_min = dist_centroid
                    res1, res2, idx_res1, idx_res2 = g_1, g_2, idx_g_1, idx_g_2
    return res1, res2, idx_res1, idx_res2


if __name__ == '__main__':
    data1 = np.load('data/hac/data1.npy')
    data2 = np.load('data/hac/data2.npy')
    data3 = np.load('data/hac/data3.npy')
    data4 = np.load('data/hac/data4.npy')
    data_list = [data1, data2, data3, data4]
    criteria = [single_linkage, complete_linkage, average_linkage, centroid]
    for i in range(4):
        data = data_list[i]
        if i == 3:
            k = 4
        else:
            k = 2
        for j in range(4):
            active_set = []
            for datum in data:
                active_set.append([datum])
            distance = criteria[j]
            while len(active_set) > k:
                g_1, g_2, idx_g_1, idx_g_2 = distance(active_set)
                del active_set[idx_g_1]
                if idx_g_2 > idx_g_1:
                    idx_g_2 -= 1
                del active_set[idx_g_2]
                active_set.append(g_1 + g_2)
            active_set = np.asarray(active_set, dtype=object)
            for idx in range(k):
                active_set[idx] = np.asarray(active_set[idx], dtype=object)
                plt.scatter(active_set[idx][:, 0], active_set[idx][:, 1])
            plt.savefig('part3_plot_{}_{}'.format(i+1, j+1))
            plt.clf()

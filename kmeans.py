import numpy as np
import math
import copy
import matplotlib.pyplot as plt


def initialize_centers(k, cluster):
    means = np.empty((k, 2), dtype=object)
    means[:, 0] = np.random.uniform(min(cluster[:, 0]), max(cluster[:, 0]), k)
    means[:, 1] = np.random.uniform(min(cluster[:, 1]), max(cluster[:, 1]), k)
    return means


if __name__ == '__main__':
    clustering1 = np.load('data/kmeans/clustering1.npy')
    clustering2 = np.load('data/kmeans/clustering2.npy')
    clustering3 = np.load('data/kmeans/clustering3.npy')
    clustering4 = np.load('data/kmeans/clustering4.npy')
    clusterings = [clustering1, clustering2, clustering3, clustering4]
    temp = 1
    for clustering in clusterings:
        obj_funcs = []
        for k in range(1, 11):
            objective_func = 0.
            for restart in range(10):
                centers = initialize_centers(k, clustering)
                while True:
                    clusters = []
                    for i in range(k):
                        clusters.append([])
                    for data in clustering:
                        min_dist = np.inf
                        assign_to = 0
                        for j in range(k):
                            dist = math.sqrt((data[0] - centers[j, 0]) ** 2 + (data[1] - centers[j, 1]) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                assign_to = j
                        clusters[assign_to].append(data)
                    old_centers = copy.deepcopy(centers)
                    clusters = np.asarray(clusters, dtype=object)
                    for i in range(k):
                        clusters[i] = np.asarray(clusters[i], dtype=object)
                    for i in range(k):
                        if len(clusters[i]) > 0:
                            centers[i] = np.mean(clusters[i], axis=0)
                    if (old_centers == centers).all():
                        break
                for i in range(k):
                    for point in clusters[i]:
                        objective_func += (point[0] - centers[i][0]) ** 2 + (point[1] - centers[i][1]) ** 2
            objective_func /= 10
            obj_funcs.append(objective_func)
        plt.plot(range(1, 11), obj_funcs)
        plt.xlabel('k')
        plt.ylabel('Objective function')
        plt.savefig('kmeans_plot{}.png'.format(temp))
        plt.clf()
        temp += 1

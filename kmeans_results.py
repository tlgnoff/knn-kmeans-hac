import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from kmeans import initialize_centers


if __name__ == '__main__':
    clustering1 = np.load('data/kmeans/clustering1.npy')
    clustering2 = np.load('data/kmeans/clustering2.npy')
    clustering3 = np.load('data/kmeans/clustering3.npy')
    clustering4 = np.load('data/kmeans/clustering4.npy')
    clusterings = [clustering1, clustering2, clustering3, clustering4]
    k = 2
    for clustering in clusterings:
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
            plt.scatter(clusters[i][:, 0], clusters[i][:, 1], s=7, marker='.')
        plt.scatter(centers[:, 0], centers[:, 1], marker='s')
        if k == 4:
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2, 2)
        elif k == 2:
            plt.ylim(-1.5, 1.5)
        plt.savefig('plot_res{}.png'.format(k-1))
        plt.clf()
        k += 1

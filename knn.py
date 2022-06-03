import numpy as np
import math
import matplotlib.pyplot as plt


def distance(x_j, x_new):
    return math.sqrt((x_j[0] - x_new[0]) ** 2 + (x_j[1] - x_new[1]) ** 2 + (x_j[2] - x_new[2]) ** 2 + (x_j[3] - x_new[3]) ** 2)


def cross_validation(test, train, k):
    average_acc = 0.
    test_size = len(test)
    for i in range(test_size):
        neighbors = list()
        for fold in train:
            for j in range(25):
                neighbors.append((fold[j], distance(fold[j], test[i])))
        neighbors.sort(key=lambda neighbor: neighbor[1])
        neighbors_labels = [neighbor[0][4] for neighbor in neighbors[:k]]
        average_acc += (max(neighbors_labels, key=neighbors_labels.count) == test[i, 4])
    return average_acc/test_size


if __name__ == '__main__':
    train_data = np.load('data/knn/train_data.npy')
    train_labels = np.load('data/knn/train_labels.npy')
    test_data = np.load('data/knn/test_data.npy')
    test_labels = np.load('data/knn/test_labels.npy')
    train_data_size = len(train_labels)
    test_data_size = len(test_labels)
    data_label = np.zeros([train_data_size, 5])
    for i in range(train_data_size):
        data_label[i, 0] = train_data[i, 0]
        data_label[i, 1] = train_data[i, 1]
        data_label[i, 2] = train_data[i, 2]
        data_label[i, 3] = train_data[i, 3]
        data_label[i, 4] = train_labels[i]
    train_folds = np.split(data_label, 10)
    max_acc = 0
    best_k = 0
    accuracies = []
    for k in range(1, 200):
        average_acc = 0.
        for fold in range(10):
            test = train_folds[fold]
            train = train_folds[:fold] + train_folds[fold+1:]
            average_acc += cross_validation(test, train, k)
        average_acc /= 10
        accuracies.append(average_acc)
        if average_acc > max_acc:
            max_acc = average_acc
            best_k = k
    plt.plot(range(1,200), accuracies)
    plt.xlabel('k')
    plt.ylabel('Average accuracies')
    plt.savefig('part1.png')
    print('\nBest k: {}'.format(best_k))
    test_acc = 0.
    test_data_label = np.zeros([test_data_size, 5])
    for i in range(test_data_size):
        test_data_label[i, 0] = test_data[i, 0]
        test_data_label[i, 1] = test_data[i, 1]
        test_data_label[i, 2] = test_data[i, 2]
        test_data_label[i, 3] = test_data[i, 3]
        test_data_label[i, 4] = test_labels[i]
    result = cross_validation(test_data_label, train_folds, best_k)
    print('Accuracy of best k: {}'.format(result))


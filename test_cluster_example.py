# coding:utf-8
import time
import numpy as np
import model.cluster as cluster
import os
import collections
import time

c = collections.Counter()

_cur_dir = os.path.dirname(os.path.realpath(__file__))


class config():
    def __init__(self):
        self.result_dir = os.path.join(_cur_dir)
        self.input_file = os.path.join(_cur_dir, 'cluster_test_data30.txt')
        self.vector_len = 32


def train(config, X):
    print(X.shape)
    clst = cluster.AdaptiveAgglomerativeClustering(linkage='complete', affinity='cosine', threhold=0.8)
    print("Training...")
    labels = clst.fit_predict(X)
    n_clusters = clst.n_clusters
    return labels, n_clusters


if __name__ == '__main__':
    #load data
    config = config()
    with open(config.input_file, 'r') as fr:
        train_data = [r.strip().split('\t') for r in fr]
    train_data = [r[:-1]+[[float(num) for num in r[-1][1:-1].replace(' ','').split(',')]] for r in train_data if len(r) == 20]
    print("Done with data length:", len(train_data))

    # train
    feature = [r[-1] for r in train_data]
    feature = np.array(feature)
    begin = time.time()
    if len(feature) > 1:
        labels, n_clusters = train(config, feature)
    else:
        labels, n_clusters = [0], 1
    print("Time cost:", time.time() - begin)

    # write cluster result
    write_data = []
    for i, item in enumerate(train_data):
        item.append(labels[i])
        write_data.append(item)
    write_data.sort(key=lambda a: a[-1])
    write_data = [[str(r) for r in l] for l in write_data]
    write_data = ['\t'.join(l[-1:] + l[13:14] + l[:-1]) for l in write_data]
    fw = open(os.path.join(config.result_dir, 'cluster_test_result.txt'), 'w')
    fw.write('\n'.join(write_data))
    fw.close()

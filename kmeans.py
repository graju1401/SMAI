from sklearn.cluster import KMeans
import numpy as np
from gensim.models import KeyedVectors


def cluster_node_features():
    node_features_file = input("Enter feature vectors file: ")
    node_features = KeyedVectors.load_word2vec_format(node_features_file)

    N = len(node_features.vocab)
    X = []
    for i in range(1, N + 1):
        X.append(np.array(node_features[str(i)]))

    n_clusters = int(input("Enter no. of clusters: "))
    kmeans = KMeans(n_clusters=n_clusters).fit(X)

    clusters_output_file = input("Enter output file: ")
    with open(clusters_output_file, 'w') as out:
        cf = [str(N) + " " + str(n_clusters) + "\n"]
        for i in range(N):
            cl = [0] * n_clusters
            cl[kmeans.predict(X[i].reshape(1, -1))[0]] = 1
            cn = [i + 1] + cl
            cf.append(" ".join([str(n) for n in cn]) + "\n")
        out.writelines(cf)

if __name__ == "__main__":
    cluster_node_features()

import networkx as nx
from gensim.models import Word2Vec
import node2vec
from sklearn.cluster import KMeans
import numpy as np


def read_graph():

    edgelist_file = input("Enter graph edgelist filename: ")
    is_unweighted = input("Unweighted graph (Y/N): ")
    is_undirected = input("Undirected graph (Y/N): ")

    if is_unweighted == "Y":
        nx_graph = nx.read_edgelist(edgelist_file, nodetype=int, create_using=nx.DiGraph())
        for edge in nx_graph.edges():
            nx_graph[edge[0]][edge[1]]['weight'] = 1
    else:
        nx_graph = nx.read_edgelist(edgelist_file, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())

    if is_undirected == "Y":
        nx_graph = nx_graph.to_undirected()

    return nx_graph


def learn_node_features(walks, dim, window, epoch, output):
    emb_walks = [[str(n) for n in walk] for walk in walks]
    node_model = Word2Vec(emb_walks, size=dim, window=window, min_count=0, sg=1, workers=4, iter=epoch)
    node_model.wv.save_word2vec_format(output)


def cluster_node_features():
    node_features = Word2Vec.load_word2vec_format("test.output")
    N = len(node_features.vocab)
    X = []
    for i in range(1, N + 1):
        X.append(np.array(node_features[str(i)]))
    kmeans = KMeans(n_clusters=8).fit(X)
    for i in range(N):
        print(i + 1, kmeans.predict(X[i].reshape(1, -1))[0])


if __name__ == '__main__':
    print("OK")
    nx_graph = read_graph()

    print("Select Algorithm")
    print("1) Node2vec")
    print("2) DeepWalk")
    print("3) LINE")
    select = input("Enter option: ")

    if select == "1":
        print("Based on previous experiments the best in-out and return hyperparameters are {0.25, 0.50, 1, 2, 4}")
        P = float(input("Enter in-out parameter: "))
        Q = float(input("Enter return parameter: "))
        graph = node2vec.Graph(nx_graph, is_directed=nx.is_directed(nx_graph), p=P, q=Q)
        graph.preprocess_transition_probs()
        num_walks = int(input("Enter no. of walks to sample for each node: "))
        walk_length = int(input("Enter length of each walk: "))
        walks = graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)

        D = int(input("Enter dimensionality of the feature vectors: "))
        W = int(input("Enter window size: "))
        epoch = int(input("Enter number of iterations: "))
        output = input("Enter output file: ")
        learn_node_features(walks=walks, dim=D, window=W, epoch=epoch, output=output)

    elif select == "2":
        pass

    elif select == "3":
        pass

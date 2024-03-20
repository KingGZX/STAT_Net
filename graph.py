import numpy as np
from loadata import Config


class Graph:
    """
    build a human body graph
    """

    def __init__(self, max_hop=1):
        """_summary_

        Args:
            self.hop_dis: 两个node间连接的edge的数量
        """
        self.dialation = 1
        self.adjacency = None
        self.center = None
        self.num_node = None
        self.max_hop = max_hop
        self.get_edges()
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency()

    def get_edges(self):
        self.num_node = Config.joints
        self_link = [(i, i) for i in range(self.num_node)]

        # with spine
        neighbor_link = [(0, 4), (0, 14), (0, 18), (4, 5),
                         # shoulder -> upper arm  upper arm ->forearm
                         (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (12, 13),
                         # upper leg -> lower leg, lower leg -> foot,  foot -> toe
                         (14, 15), (15, 16), (16, 17), (18, 19), (19, 20), (20, 21), ]

        self.edge = self_link + neighbor_link
        self.center = 0  # use pelvis as the center of body

    def get_adjacency(self):
        self.adjacency = np.zeros((self.num_node, self.num_node))
        self.adjacency[self.hop_dis <= self.max_hop] = 1
        self.adjacency = undirected_graph_norm(self.adjacency)

        valid_hop = range(0, self.max_hop + 1, self.dialation)
        A = []
        A.append(self.adjacency)
        A = np.stack(A)
        self.A = A

        """
        spatial graph partition strategy
        利用的是距离 “重心” 的距离
        """
        # A = []
        # for hop in valid_hop:
        #     a_root = np.zeros((self.num_node, self.num_node))
        #     a_close = np.zeros((self.num_node, self.num_node))
        #     a_further = np.zeros((self.num_node, self.num_node))
        #     for i in range(self.num_node):
        #         for j in range(self.num_node):
        #             if self.hop_dis[j, i] == hop:
        #                 if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
        #                     a_root[j, i] = self.adjacency[j, i]
        #                 elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
        #                     a_close[j, i] = self.adjacency[j, i]
        #                 else:
        #                     a_further[j, i] = self.adjacency[j, i]
        #     if hop == 0:
        #         A.append(a_root)
        #     else:
        #         A.append(a_root + a_close)
        #         A.append(a_further)
        # A = np.stack(A)
        # self.A = A


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    """
    copied from st-gcn source code:
    link: https://github.com/yysijie/st-gcn

    My naive thinking of transfer matrix:
    when d = 0, it's identity matrix and it's simple to explain that the root node to the node itself has a distance 0
    when d = 1, it's just adjacency matrix.
    when d >= 2, use the concept of matrix multiplication row map, e.g., AA = C, 
    which means, the first row of C is just: sum(A[0,i] * A[i]) ,
    therefore we can update the distance of node1 to the other nodes especially 
    those node1 do not directly link to but node1's neighbor link to.

    by multiply again and again, we can explore all the nodes that one node can go to.
    """
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def undirected_graph_norm(adjacency):
    """
    :param adjacency:
        adjacency matrix of the graph
    :return:
        xxx after normalization

    Note: Graph Convolution Formula
    f = (D^-1/2 A D^-1/2 H W)
    """
    degree = np.sum(adjacency, axis=1)
    degree = np.sqrt(degree)
    deg_matrix = np.multiply(np.identity(len(degree)), degree)
    adjacency = np.dot(np.dot(deg_matrix, adjacency), adjacency)
    return adjacency

# code for debugging
# ins = Graph()
# b = 0

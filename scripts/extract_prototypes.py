# Unsupervised discretization algorithm for effect based action prototype generation.
import os, sys, torch, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from copy import copy, deepcopy
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
import statistics
from networkx import Graph
from tqdm import tqdm
import pickle, os
import pandas as pd
from einops import rearrange, repeat
from pandas import DataFrame

from rnn_policy.policy import NavNetPolicy, action_with_duration, mp42np, mp42np, read_json, sequence_forward
from rnn_policy.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_pack_info_from_episode_ids,
    build_rnn_build_seq_info,
    build_rnn_state_encoder,
)


class RgngGraph(Graph):
    ''''Class representing a graph for the RGNG algorithm
    '''
    def add_node(self, node_for_adding, **attr):
        for i in self.nodes():
            self.nodes[i]["prenode_ranking"] += 1
        return super().add_node(node_for_adding, prenode_ranking=0, **attr)

class RobustGrowingNeuralGas:

    def __init__(self, input_data, max_number_of_nodes, real_num_clusters=1, center=None):
        self.network = None
        self.optimal_network = None
        self.data = input_data
        self.units_created = 0
        self.beta_integral_mul = 2
        self.fsigma_weakening = 0.1
        self.e_bi = 0.1
        self.e_bf = 0.01
        self.e_ni = 0.005
        self.e_nf = 0.0005
        self.eta = 0.0001
        self.inputted_vectors = []
        self.outliers = []
        self.receptive_field = {}
        self.optimal_receptive_field = {}
        self.max_nodes = len(input_data)

        # Matlab
        self.stopcriteria = 0.00001
        self.init_centers = center

        self.prenumnode = max_number_of_nodes
        self.num_features = len(input_data[0])
        self.num_classes = real_num_clusters
        self.optimal_center = []

        self.old_prototypes = []

        self.units_created = 0
        self.epsilon = 10 ** (-5)

        # 0. start with two units a and b at random position w_a and w_b
        if self.init_centers is None:
            w_a = [
                np.random.uniform(
                    np.min(np.min(self.data, 0), 0), np.max(np.max(self.data, 1), 0)
                )
                for _ in range(np.shape(self.data)[1])
            ]
            w_b = [
                np.random.uniform(
                    np.min(np.min(self.data, 0), 0), np.max(np.max(self.data, 1), 0)
                )
                for _ in range(np.shape(self.data)[1])
            ]
        else:
            w_a = self.init_centers[0]
            w_b = self.init_centers[1]
        self.network = RgngGraph()
        self.network.add_node(self.units_created, vector=w_a, error=0, e_b=0, e_n=0)
        self.units_created += 1
        self.network.add_node(self.units_created, vector=w_b, error=0, e_b=0, e_n=0)
        self.units_created += 1

        for node in self.network.nodes():
            self.network.nodes[node]["prenode_ranking"] = 1

    def find_nearest_units(self, observation):
        distance = []
        for u, attributes in self.network.nodes(data=True):
            vector = attributes["vector"]
            dist = np.linalg.norm((observation - vector), ord=2) + self.epsilon
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        return ranking

    def prune_connections(self, a_max):
        nodes_to_remove = []
        for u, v, attributes in self.network.edges(data=True):
            if attributes["age"] > a_max:
                nodes_to_remove.append((u, v))
        for u, v in nodes_to_remove:
            self.network.remove_edge(u, v)

        nodes_to_remove = []
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                nodes_to_remove.append(u)
        for u in nodes_to_remove:
            self.network.remove_node(u)

    def fit_network(self, a_max, passes=10):
        # np.random.seed(1)
        self.epochspernode = passes

        nofirsttimeflag = 0
        stopflag = 1
        allmdlvalue = []
        previousmdlvalue = 999999999

        # 1. iterate through the data
        sequence = 0

        self.data_range = np.max(np.max(self.data, axis=0)) - np.min(np.min(self.data, axis=0))
        # np.random.shuffle(self.data)

        while len(self.network.nodes()) <= self.prenumnode and stopflag == 1:
            # print("Training when the number of the nodes in RGNG is: " + str(len(self.network.nodes())))
            flag = 1
            self.old_prototypes = []
            harmdist = []
            for i in self.network.nodes():
                self.old_prototypes.append([self.network.nodes[i]["vector"], i])
                temp = 0
                for x in self.data:
                    temp = temp + 1 / (
                        np.linalg.norm(x - self.network.nodes[i]["vector"], 2)
                        + self.epsilon
                    )
                harmdist.append((temp / len(self.data)))
                self.network.nodes[i]["e_b"] = self.e_bi * pow(
                    (self.e_bf / self.e_bi),
                    (self.network.nodes[i]["prenode_ranking"] - 1) / self.max_nodes,
                )
                self.network.nodes[i]["e_n"] = self.e_ni * pow(
                    (self.e_nf / self.e_ni),
                    (self.network.nodes[i]["prenode_ranking"] - 1) / self.max_nodes,
                )

            rand_state_count = 0
            for iter2 in range(self.epochspernode):
                tempvalue = 1 / np.array(harmdist)

                # In case a node gets deleted the node index still stays the same for the nodes above this dict keeps track of that
                NODE_TO_VALUE_CORRESPONDANCE = {}
                for i, node in enumerate(self.network.nodes()):
                    NODE_TO_VALUE_CORRESPONDANCE[node] = i

                if flag == 1:
                    iter1 = 0

                    np.random.shuffle(self.data)
                    workdata = list(deepcopy(self.data))
                    for observation in workdata:
                        iter1 = iter1 + 1
                        t = iter1 + iter2 * len(self.data)
                        # self.d_restr = { n_[0]: (np.linalg.norm( (observation - n_[1]['vector'])**2, ord=2) + self.epsilon) for n_ in self.network.nodes.items() }
                        self.d_restr = {
                            n_: (
                                np.linalg.norm(
                                    (observation - self.network.nodes[n_]["vector"])
                                    ** 2,
                                    ord=2,
                                )
                                + self.epsilon
                            )
                            for n_ in self.network.nodes()
                        }  # .items

                        # 2: Determine the winner S1 and the second nearest node S2
                        nearest_units = self.find_nearest_units(observation)
                        s_1 = nearest_units[0]
                        s_2 = nearest_units[1]

                        # 3:Set up or refresh connection relationship between S1 and S2
                        self.network.add_edge(s_1, s_2, age=0)

                        # 4: Adapt the reference vectors of S1 and its direct topological neighbours
                        tempv = 0
                        if (
                            self.d_restr[s_1]
                            > tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]]
                        ):
                            tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]] = 2 / (
                                (1 / self.d_restr[s_1])
                                + (1 / tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]])
                            )
                            tempv = tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]]
                        else:
                            tempv = self.d_restr[s_1]
                            tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]] = (
                                self.d_restr[s_1]
                                + tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]]
                            ) / 2

                        update_w_s_1 = (
                            self.network.nodes[s_1]["e_b"]
                            * tempv
                            * (
                                (observation - self.network.nodes[s_1]["vector"])
                                / self.d_restr[s_1]
                            )
                        )
                        self.network.nodes[s_1]["vector"] = np.add(
                            self.network.nodes[s_1]["vector"], update_w_s_1
                        )

                        # find neighbours of the winning node S1 and update them
                        avg_neighbor_dist = 0
                        if len(list(self.network.neighbors(s_1))) > 0:
                            avg_neighbor_dist = sum(
                                [
                                    np.linalg.norm(
                                        self.network.nodes[nb]["vector"]
                                        - self.network.nodes[s_1]["vector"],
                                        ord=2,
                                    )
                                    + self.epsilon
                                    for nb in self.network.neighbors(s_1)
                                ]
                            ) / len(list(self.network.neighbors(s_1)))

                        for neighbor in self.network.neighbors(s_1):
                            tempv = 0
                            if (
                                self.d_restr[neighbor]
                                > tempvalue[NODE_TO_VALUE_CORRESPONDANCE[neighbor]]
                            ):
                                tempvalue[NODE_TO_VALUE_CORRESPONDANCE[neighbor]] = (
                                    2
                                    / (
                                        (1 / self.d_restr[neighbor])
                                        + (
                                            1
                                            / tempvalue[
                                                NODE_TO_VALUE_CORRESPONDANCE[neighbor]
                                            ]
                                        )
                                    )
                                )
                                tempv = tempvalue[
                                    NODE_TO_VALUE_CORRESPONDANCE[neighbor]
                                ]
                            else:
                                tempv = self.d_restr[neighbor]
                                tempvalue[NODE_TO_VALUE_CORRESPONDANCE[neighbor]] = (
                                    self.d_restr[neighbor]
                                    + tempvalue[NODE_TO_VALUE_CORRESPONDANCE[neighbor]]
                                ) / 2

                            s1_to_neighbor = np.subtract(
                                self.network.nodes[neighbor]["vector"],
                                self.network.nodes[s_1]["vector"],
                            )

                            update_w_s_n = self.network.nodes[neighbor][
                                "e_n"
                            ] * tempv * (
                                (observation - self.network.nodes[neighbor]["vector"])
                                / self.d_restr[neighbor]
                            ) + np.exp(
                                -(
                                    np.linalg.norm(
                                        self.network.nodes[neighbor]["vector"]
                                        - self.network.nodes[s_1]["vector"],
                                        ord=2,
                                    )
                                    + self.epsilon
                                )
                                / self.fsigma_weakening
                            ) * self.beta_integral_mul * avg_neighbor_dist * (
                                s1_to_neighbor / np.linalg.norm(s1_to_neighbor, ord=2)
                                + self.epsilon
                            )
                            self.network.nodes[neighbor]["vector"] = np.add(
                                self.network.nodes[neighbor]["vector"], update_w_s_n
                            )

                        # 5: Increase the age of all edges emanating from S1
                        for u, v, attributes in self.network.edges(
                            data=True, nbunch=[s_1]
                        ):
                            self.network.add_edge(u, v, age=attributes["age"] + 1)

                        # 6: Removal of nodes
                        if nofirsttimeflag == 1:
                            self.prune_connections(a_max)
                    rand_state_count += 1
                    # Check if stopping criterion is meet
                    crit = 0
                    for vector, i in self.old_prototypes:
                        try:
                            crit += (
                                np.linalg.norm(
                                    vector - self.network.nodes[i]["vector"], ord=2
                                )
                                + self.epsilon
                            )
                        except:
                            crit += np.linalg.norm(vector, ord=2)
                    crit /= len(self.network.nodes())
                    if crit <= self.stopcriteria:
                        # print("stop")
                        flag = 0
                    else:
                        for i in self.network.nodes():
                            for x, [vector, j] in enumerate(self.old_prototypes):
                                if j == i:
                                    self.old_prototypes[x][0] = self.network.nodes[i][
                                        "vector"
                                    ]

                    harmdist = []
                    for i in self.network.nodes():
                        temp = 0
                        for x in self.data:
                            temp = temp + 1 / (
                                np.linalg.norm(x - self.network.nodes[i]["vector"], 2)
                                + self.epsilon
                            )
                        harmdist.append((temp / len(self.data)))

                    a = 1
            # End Epoch Loop
            # print("Epoch loop ended!!!")

            # Rebuiding the topology relationship among all current nodes
            self.d = {}  # np.zeros(len(self.network.nodes()))

            # Harmonic average distance for each node
            harmonic_average = {}
            self.receptive_field = {}
            for observation in self.data:
                self.d = {
                    n_: (
                        np.linalg.norm(
                            (observation - self.network.nodes[n_]["vector"]), ord=2
                        )
                    )
                    for n_ in self.network.nodes()
                }
                # for i in range(len(self.d)):
                #     self.d[i] = np.linalg.norm( (observation - self.network.nodes[i]['vector']), ord=2) + self.epsilon
                nearest_units = self.find_nearest_units(observation)
                s_1 = nearest_units[0]
                s_2 = nearest_units[1]
                self.network.add_edge(s_1, s_2, age=0)
                # Update receptive field of winner (optimal cluster size)
                if s_1 not in self.receptive_field.keys():
                    self.receptive_field[s_1] = {"input": [observation]}
                else:
                    self.receptive_field[s_1]["input"].append(observation)
                if s_1 not in harmonic_average.keys():
                    harmonic_average[s_1] = 1 / self.d[s_1]
                else:
                    harmonic_average[s_1] += 1 / self.d[s_1]

            for s in self.network.nodes():
                if s in self.receptive_field.keys():
                    harmonic_average[s] = (
                        len(self.receptive_field[s]["input"]) / harmonic_average[s]
                    )
            # print(harmonic_average)
            # Local Error in each node
            for u in self.network.nodes():
                self.network.nodes[u]["error"] = 0
            for obs in self.data:
                # d = { n_[0]: (np.linalg.norm( (obs - n_[1]['vector']), ord=2)) for n_ in self.network.nodes.items() }
                d = {
                    n_: (
                        np.linalg.norm((obs - self.network.nodes[n_]["vector"]), ord=2)
                    )
                    for n_ in self.network.nodes()
                }
                nearest_units = self.find_nearest_units(observation)
                s = nearest_units[0]
                self.network.nodes[u]["error"] = (
                    self.network.nodes[u]["error"]
                    + np.exp(-(d[s] / harmonic_average[s])) * d[s]
                )

            # MDL Calc
            prototypes = np.zeros(
                (len(self.network.nodes), np.shape(self.data)[1])
            )  # len(self.init_centers[0])
            for i, n in enumerate(self.network.nodes()):
                prototypes[i] = self.network.nodes[n]["vector"]

            prototypes = np.array(prototypes)
            # print("Prototypes: ")
            # print(prototypes)

            mdlvalue = self.outliertest(prototypes)
            if mdlvalue < previousmdlvalue:
                previousmdlvalue = mdlvalue
                self.optimal_center = prototypes
                self.optimal_receptive_field = deepcopy(self.receptive_field)
                self.optimal_network = deepcopy(self.network)
            # print(prototypes.shape[0])
            if prototypes.shape[0] == self.num_classes:
                actcenter = prototypes
            allmdlvalue.append(mdlvalue)

            # new node calc
            error_max = 0
            q = None
            for u in self.network.nodes():
                if self.network.nodes[u]["error"] > error_max:
                    error_max = self.network.nodes[u]["error"]
                    q = u
            # 8.b insert a new unit r halfway between q and its neighbor f with the largest error variable
            f = -1
            largest_error = -1
            for u in self.network.neighbors(q):
                if self.network.nodes[u]["error"] > largest_error:
                    largest_error = self.network.nodes[u]["error"]
                    f = u
            w_r = 0.5 * (
                np.add(self.network.nodes[q]["vector"], self.network.nodes[f]["vector"])
            )
            r = self.units_created
            self.units_created += 1
            # 10: Insert new node
            if len(self.network.nodes()) < self.prenumnode:
                self.network.add_node(r, vector=w_r, error=0, e_b=0, e_n=0)
            else:
                stopflag = 0

            nofirsttimeflag = 1
            self.network.add_edge(r, q, age=0)
            self.network.add_edge(r, f, age=0)
            self.network.remove_edge(q, f)

        return self.optimal_center  # , actcenter, allmdlvalue

    def outliertest(self, center):
        yeta = 0.000005  # data accuracy
        ki = 1.2  # Error balance coefficient
        rangevalue = self.data_range
        harmdist = np.zeros(len(self.network.nodes()))
        counter = np.zeros(center.shape[0])
        inderrorvector = []
        totalerrorvector = []

        NODE_TO_VALUE_CORRESPONDANCE = {}

        for i, node in enumerate(self.network.nodes()):
            inderrorvector.append(np.zeros_like(center[0]))
            NODE_TO_VALUE_CORRESPONDANCE[node] = i
        self.receptive_field = {}

        for observation in self.data:
            # rms error
            d = {
                n_[0]: (np.linalg.norm((observation - n_[1]["vector"]), ord=2))
                for n_ in self.network.nodes.items()
            }
            nearest_units = self.find_nearest_units(observation)
            s = nearest_units[0]
            s_index_in_matrix = NODE_TO_VALUE_CORRESPONDANCE[s]
            harmdist[s_index_in_matrix] = harmdist[s_index_in_matrix] + 1 / d[s]
            counter[s_index_in_matrix] = counter[s_index_in_matrix] + 1
            # Update receptive field of winner (optimal cluster size)
            if s not in self.receptive_field.keys():
                self.receptive_field[s] = {"input": [observation]}
            else:
                self.receptive_field[s]["input"].append(observation)

            inderrorvector[s_index_in_matrix] = inderrorvector[s_index_in_matrix] + d[s]
            totalerrorvector.append(d[s])

        for n in self.network.nodes():
            if n not in self.receptive_field.keys():
                harmdist[NODE_TO_VALUE_CORRESPONDANCE[n]] = 99999999
            else:
                harmdist[NODE_TO_VALUE_CORRESPONDANCE[n]] = harmdist[
                    NODE_TO_VALUE_CORRESPONDANCE[n]
                ] / len(self.receptive_field[n]["input"])

        disvector = np.zeros(len(self.data))
        for i, obs in enumerate(self.data):
            d = np.zeros(len(self.network.nodes()))
            for n in self.network.nodes():
                d[NODE_TO_VALUE_CORRESPONDANCE[n]] = np.linalg.norm(
                    (obs - self.network.nodes[n]["vector"]), ord=2
                )
            disvector[i] = sum(1 / (d * harmdist))

        outliercandidate = np.sort(disvector)
        outliercandidate_args = np.argsort(disvector)

        outdata = []
        errorvalue = 0
        protosize = center.shape[0]

        flagprototype = 0

        for i in range(len(outliercandidate)):
            d = np.zeros(len(self.network.nodes()))
            for j in range(center.shape[0]):
                d[j] = np.linalg.norm(
                    self.data[outliercandidate_args[i]] - center[j], 2
                )
            minval = np.min(d)
            s = np.argmin(d)
            erroradd = 0
            for h in np.arange(0, (self.data.shape[1])).reshape(
                -1
            ):  # range(self.data.shape[1]): #
                if np.abs(self.data[outliercandidate_args[i], h] - center[s, h]) != 0:
                    erroradd = erroradd + np.max(
                        np.log2(
                            np.abs(
                                self.data[outliercandidate_args[i], h] - center[s, h]
                            )
                            / yeta
                        )
                    )
                else:
                    erroradd = erroradd + 1
            errorvalue = errorvalue + erroradd

            a = 1

            if counter[s] >= 2:
                flagprototype = 0
                indexchange = np.log2(protosize)
            else:
                indexchange = np.log2(protosize) + (
                    len(self.data) - len(outdata) - 1
                ) * (np.log2(protosize) - np.log2(protosize - 1))
                protosize = protosize - 1
                flagprototype = 1

            if (ki * erroradd + indexchange) + flagprototype * center.shape[1] * (
                np.ceil(np.log2(rangevalue / yeta)) + 1
            ) > self.data.shape[1] * (np.ceil(np.log2(rangevalue / yeta)) + 1):
                outdata.append(
                    self.data[
                        outliercandidate_args[i], np.arange(0, self.data.shape[1])
                    ]
                )
                counter[s] = counter[s] - 1
                errorvalue = errorvalue - erroradd

        indexvalue = (len(self.data) - len(outdata)) * np.log2(protosize + 1)
        mdlvalue = (
            protosize * center.shape[1] * (np.ceil(np.log2(rangevalue / yeta)) + 1)
            + indexvalue
            + ki * errorvalue
            + len(outdata)
            * (self.data.shape[1])
            * (np.ceil(np.log2(rangevalue / yeta)) + 1)
        )
        # print("MDL Value: ", mdlvalue)

        return mdlvalue

    # ==================================
    # proto, obs, d(-1)
    # (self.network.nodes[s_1]['vector'], observation, self.d_restr_prev[s_1])
    def sigma_modulation(self, proto, observation, prev):
        # pseudo algo
        current_error = np.linalg.norm(observation - proto)
        if current_error < prev:
            return current_error
        else:
            return prev

    def update_restricting_dist(self, proto, observation, prev):
        current_error = np.linalg.norm(observation - proto)
        if current_error < prev:
            # arithmetic mean
            return 0.5 * (prev + current_error)
        else:
            # harmonic mean
            return statistics.harmonic_mean([prev, current_error])

    # array a
    def h_mean(self, a):
        return statistics.harmonic_mean(a)

    def number_of_clusters(self):
        return nx.number_connected_components(self.network)

    def cluster_data(self):
        color = ["r", "b", "g", "k", "m", "r", "b", "g", "k", "m"]
        clustered_data = []
        for key in self.optimal_receptive_field.keys():
            vectors = np.array(self.optimal_receptive_field[key]["input"])
            for obs in self.optimal_receptive_field[key]["input"]:
                clustered_data.append((obs, key))

        return clustered_data

    def reduce_dimension(self, clustered_data):
        transformed_clustered_data = []
        svd = decomposition.PCA(n_components=2)
        transformed_observations = svd.fit_transform(self.data)
        for i in range(len(clustered_data)):
            transformed_clustered_data.append(
                (transformed_observations[i], clustered_data[i][1])
            )
        return transformed_clustered_data

    def compute_global_error(self):
        global_error = 0
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s_1 = nearest_units[0]
            global_error += (
                spatial.distance.euclidean(
                    observation, self.network.nodes[s_1]["vector"]
                )
                ** 2
            )
        return global_error

class EffectActionPrototypes:
    """
    Facilitates unsupervised discretization algorithm for effect based action prototype generation.

    This class is designed to process (effect, motion) collections to identify
    effect based action prototypes. It utilizes clustering techniques to analyze motion
    samples effects and generate prototypes that represent common patterns within the effect space.
    The clustering method varies based on the dimensionality of the effect dimensions provided,
    using histograms for one-dimensional data
    and K-Means clustering for multi-dimensional effect spaces.

    Attributes:
        motion_samples (pandas.DataFrame): A DataFrame containing motion samples with each
            row representing a sample and each column a dimension of motion.
        motion_dimensions (List[str]): Specifies the columns in `motion_samples` that contain
            the relevant motion dimensions for analysis.
        action_prototypes (numpy.ndarray or None): Stores the generated action prototypes
            after processing. Initially set to None.
        m_samples_labeled (pandas.DataFrame or None): Similar to `motion_samples` but includes
            an additional column for cluster labels. Initially set to None.
        prototypes_per_label (Dict[int, numpy.ndarray] or None): Maps each cluster label to its
            corresponding action prototype(s). Initially set to None.
        cluster_labels (Set[int] or None): Contains the unique labels identifying the clusters
            found in the motion data. Initially set to None.
    """

    def __init__(
        self,
        motion_samples: pd.DataFrame,
        motion_dimensions: list,
        limit_prototypes_per_cluster=10,
    ) -> None:
        self.motion_samples = copy(motion_samples)
        self.motion_dimensions = motion_dimensions
        self.action_prototypes = None
        self.m_samples_labeled = None
        self.prototypes_per_label = None

        self.__pre_process = None
        self.__prototype_per_cluster_limit = limit_prototypes_per_cluster

    def generate(
        self,
        effect_dimensions: list,
    ) -> np.ndarray:
        """
        Starts prototype generation and returns prototypes. Depending on the number of
        effect dimensions histogram binning or K-Means clustering is used on the effect
        dimensions of the data to categorize the motion samples.
        """
        # Assert effect_dimensions list
        if len(effect_dimensions) == 1 and len(effect_dimensions[0]) == 1:
            cluster_labels = self.__bin_histogram_samples(effect_dimensions[0])
        else:
            cluster_labels = self.__kmeans_effect_clustering(effect_dimensions)

        self.__generate_prototypes(effect_dimensions, cluster_labels)
        return self.action_prototypes

    def my_generate_prototypes(self, cluster_labels,num_prototype=10):
        self.prototypes_per_label = {}
        self.__pre_process = StandardScaler()
        self.m_samples_labeled = deepcopy(self.motion_samples)
        self.m_samples_labeled["cluster_label"] = self.motion_samples['action_with_duration']
        scaled_m_samples = deepcopy(self.m_samples_labeled)
        scaled_m_samples[self.motion_dimensions] = self.__pre_process.fit_transform(scaled_m_samples[self.motion_dimensions])  # normalize
        max_prototypes_per_cluster = np.array([num_prototype] * len(np.unique(cluster_labels)))
        for cluster_label, num_prototypes in tqdm(zip(cluster_labels, max_prototypes_per_cluster)):
            self.__multi_prototypes(num_prototypes,scaled_m_samples[scaled_m_samples["cluster_label"] == cluster_label],cluster_label,)

        return self.action_prototypes

    def __generate_prototypes(
        self,
        effect_dimensions: list,
        cluster_labels: dict,
    ) -> None:
        effect_dimensions = effect_dimensions[0]
        # Dynamic prototypes per cluster
        mean_stds = []
        for i in cluster_labels:
            cluster_samples = self.m_samples_labeled[self.m_samples_labeled["cluster_label"] == i]
            mean_stds.append(self.__encode_mean_std([cluster_samples], effect_dimensions))

        mean_stds = np.stack(mean_stds)
        mean_std_all_dims = np.add.reduce(mean_stds, axis=1)
        mean_std_all_dims = mean_std_all_dims / np.max(mean_std_all_dims, axis=(0, 1))

        cv = mean_std_all_dims.T[1] / mean_std_all_dims.T[0]
        cv = np.array([min(x, 0.999999) for x in cv])
        max_prototypes_per_cluster = ((1 - cv) * self.__prototype_per_cluster_limit * mean_std_all_dims.T[1])
        max_prototypes_per_cluster = max_prototypes_per_cluster / np.min(max_prototypes_per_cluster)
        max_prototypes_per_cluster = np.floor(max_prototypes_per_cluster)
        max_prototypes_per_cluster = np.array([min(x, self.__prototype_per_cluster_limit)for x in max_prototypes_per_cluster])

        self.prototypes_per_label = {}
        self.__pre_process = StandardScaler()
        scaled_m_samples = deepcopy(self.m_samples_labeled)
        scaled_m_samples[self.motion_dimensions] = self.__pre_process.fit_transform(scaled_m_samples[self.motion_dimensions])

        for cluster_label, num_prototypes in zip(cluster_labels, max_prototypes_per_cluster):
            if num_prototypes == 1:
                self.__single_prototype_per_class(cluster_label, effect_dimensions)
            else:
                self.__multi_prototypes(num_prototypes,scaled_m_samples[scaled_m_samples["cluster_label"] == cluster_label],cluster_label,)

    def __single_prototype_per_class(self, cluster_label: dict, effect_dimensions: list) -> None:
        single_cluster_df = self.m_samples_labeled[self.m_samples_labeled["cluster_label"] == cluster_label]
        cluster_effect_means = single_cluster_df[effect_dimensions].mean().to_numpy()

        closest_motion_to_effect_mean = abs(single_cluster_df[effect_dimensions] - cluster_effect_means).sum(axis=1).argmin()

        prototype = single_cluster_df[self.motion_dimensions].iloc[closest_motion_to_effect_mean]

        if self.action_prototypes is None:  # first one to add
            self.action_prototypes = prototype
        else:
            self.action_prototypes = np.vstack((self.action_prototypes, prototype))
        self.prototypes_per_label[cluster_label] = prototype

    def __multi_prototypes(
        self,
        num_prototypes: int,
        cluster_data: pd.DataFrame,
        cluster_label: int,
    ) -> None:
        # RGNG
        data_np = cluster_data[self.motion_dimensions].to_numpy()
        rgng = RobustGrowingNeuralGas(input_data=data_np, max_number_of_nodes=num_prototypes)
        resulting_centers = rgng.fit_network(a_max=100, passes=20)
        local_prototype = self.__pre_process.inverse_transform(resulting_centers)  
        if self.action_prototypes is None:
            self.action_prototypes = local_prototype
            self.prototypes_per_label[cluster_label] = local_prototype
        else:
            self.action_prototypes = np.vstack((self.action_prototypes, local_prototype))
            self.prototypes_per_label[cluster_label] = local_prototype

    def __bin_histogram_samples(self, effect_dimensions: list) -> None:
        hist, bin_edges = np.histogram(self.motion_samples[effect_dimensions[0]])

        label = 0
        cluster_labels_with_edges = {}
        cluster_labels = []
        for i, count in enumerate(hist):
            if count != 0:
                cluster_labels_with_edges[label] = (bin_edges[i], bin_edges[i + 1])
                cluster_labels.append(label)
                label += 1

        self.m_samples_labeled = copy(self.motion_samples)
        self.m_samples_labeled["cluster_label"] = self.m_samples_labeled[effect_dimensions[0]].apply(lambda x: self.__find_position_hist(x, cluster_labels_with_edges))

        return set(cluster_labels)

    def __kmeans_effect_clustering(self, effect_dimensions: list) -> None:
        flatten_effect_dims = []
        for e in effect_dimensions:
            flatten_effect_dims = flatten_effect_dims + e

        kmeans_input = np.array(self.motion_samples[flatten_effect_dims])
        norm_factor = np.max(abs(kmeans_input), axis=0)
        kmeans_input = kmeans_input / norm_factor

        range_n_clusters = [3, 4, 5, 6, 7, 8]
        best_score = np.inf
        best_num_of_clusters = 0
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(kmeans_input)
            dbi = davies_bouldin_score(kmeans_input, kmeans.labels_)
            if dbi < best_score:
                best_score = dbi
                best_num_of_clusters = n_clusters
        kmeans = KMeans(n_clusters=best_num_of_clusters, random_state=0, n_init=10).fit(kmeans_input)

        self.m_samples_labeled = self.motion_samples
        self.m_samples_labeled.loc[:, ("cluster_label")] = kmeans.labels_
        return set(kmeans.labels_)

    def __encode_mean_std(
        self,
        dfs,
        effects,
    ) -> None:
        for df in dfs:
            df_effect = df[effects]
            effect_mean = df_effect.mean()
            effect_std = df_effect.std()
            effect_array = np.zeros((len(effects), 2))
            i = 0
            for mean, std in zip(effect_mean, effect_std):
                effect_array[i] = np.array([mean, std])
                i += 1

            return effect_array

    def __find_position_hist(self, value, dict_labels) -> None:
        for key, border in dict_labels.items():
            if border[0] <= value <= border[1]:
                return key
        raise ValueError("No key found in histogram bining for value" + str(value))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="offline-dataset/habitat-dataset/900/17DRP5sb8fy")#offline-dataset/robothor-dataset/900/FloorPlan_Train1_1
    parser.add_argument("--ckpt_file", type=str, default='')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scene_name = os.path.basename(args.src_path)
    policy = NavNetPolicy(domain=scene_name).to(device)

    if args.ckpt_file:
        policy.load_state_dict(torch.load(args.ckpt_file))
        print(f"Loaded policy from {args.ckpt_file}")
    
    # load video
    video_frames = mp42np(os.path.join(args.src_path, 'rgb_video.mp4'), way='ffmpeg')[:-1]  # drop the last frame corresponding to the "STOP"
    
    # load other data
    all_data = read_json(os.path.join(args.src_path,'data.json'))
    actions = all_data['pred_action_indices']  # action from flow rather than real action
    prev_actions = [-1] + actions[:-1]
    L = len(actions)

    # generate observation features
    prototype_dir = os.path.join(os.path.dirname(args.ckpt_file), 'prototype')
    feature_file = os.path.join(prototype_dir, f'rgb_features.csv')
    if not os.path.exists(feature_file):
        os.makedirs(prototype_dir, exist_ok=True)
        
        t = all_data['objects'][0]['timestep']  # we do not care
        obs_features = sequence_forward(policy, video_frames, prev_actions, t)[1]
        actions_wd = actions if policy.n == 3 else action_with_duration(actions)
        df_data = pd.DataFrame(np.hstack([obs_features.cpu().numpy(), np.array(actions_wd,dtype=int).reshape(-1,1)]), columns=[f'f{i}' for i in range(obs_features.shape[1])]+['action_with_duration'])
        df_data.to_csv(feature_file, index=False)
    else:
        df_data = pd.read_csv(feature_file)
    motion_dims = [f'f{i}' for i in range(512)]
    action_labels = np.arange(0, policy.action_space.n)

    # generate prototypes
    num_prototype = 3
    prototype_file = os.path.join(prototype_dir, f"prototypes_{num_prototype}.csv")

    prototype_generator = EffectActionPrototypes(df_data, motion_dims)
    prototypes = prototype_generator.my_generate_prototypes(action_labels, num_prototype)
    print(prototypes.shape)
    all_proto_df = pd.DataFrame(columns=motion_dims + ['prototype'])
    for k, v in prototype_generator.prototypes_per_label.items():
        proto_df = pd.DataFrame(np.hstack((v, np.ones((v.shape[0], 1)) * k)), columns=motion_dims + ['prototype'])
        all_proto_df = pd.concat([all_proto_df, proto_df])
    all_proto_df.to_csv(prototype_file, index=False)           

    # visualize prototypes
    # all_proto_df = pd.read_csv(prototype_file, index_col=False, header=0)
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    dr = PCA(n_components=2)
    dr.fit(np.vstack((df_data[motion_dims].values, all_proto_df[motion_dims].values)))
    X = dr.transform(df_data[motion_dims].values)
    P = dr.transform(all_proto_df[motion_dims].values)
    plt.scatter(X[:, 0], X[:, 1], c=df_data['action_with_duration'],cmap='tab10',)
    print(np.unique(df_data['action_with_duration']))
    # plt.scatter(P[:, 0], P[:, 1], c=all_proto_df['prototype'], marker='^',s=200,cmap='tab10',edgecolors='black')
    # given each color which uses to plot with corresponding label
    plt.colorbar()
    plt.savefig(prototype_file.replace('.csv', '.png'))
    plt.show()
    plt.clf()

import numpy as np
import warnings
from oass.StaticDirectedAcyclicGraph import StaticDirectedAcyclicGraph


class GradientCalculator:
    """The gradient calculater.
    
    This part is implemented completely based on ``numpy``, so it naturally supports ``PyTorch``, ``TensorFlow`` and other deep learning frameworks.

    Attributes:
        gamma (float): The discount factor. Default: 1.
        baseline_strategy (str): The calculation method of baseline value. It can be ``"random"``, ``"self"`` or ``"zero"``. Default: ``"random"``.
        extra_function (str): The extra function applied on ``action_prob``. It can be ``"none"`` or ``"log"``. Default: ``"none"``.
    """
    def __init__(self, gamma=1, baseline_strategy="random", extra_function="None"):
        self.gamma = gamma
        self.baseline_strategy = baseline_strategy
        self.extra_function = extra_function

    def calculate_gradient(self, G: StaticDirectedAcyclicGraph, action_prob, node_reward, edge_reward):
        """Calculate gradient for a DAG.
        
        Args:
            G (StaticDirectedAcyclicGraph): The DAG.
            action_prob (array-like type): The probabilities that the agent chooses each edge. The way it is saved needs to be consistent with ``G.edge``. At each node u, ``action_prob[u]`` is a :math:`|N(u)|` x batch_size matrix.
            node_reward (array-like type): The reward when arriving at each node. It is an array that contains :math:`|V|` reward values.
            edge_reward (array-like type): The reward when passing each edge. At each node u, ``edge_reward[u]`` is an array that contains :math:`|N(u)|` reward values.

        Returns:
            E (np.array): The mathematical expectation of the subsequent rewards when starting with each node.
            D (list(np.array)): The gradient value for updating ``action_prob``.

        .. note::
            For ease of use, ``action_prob`` may contain additional values, but must cover all the edges.
        """
        G.topological_sort()
        batch_size = action_prob[0].shape[-1]
        E = np.zeros((G.n, batch_size))
        D = [np.zeros(len(G.edge[u])) for u in range(G.n)]
        for u in G.topological_node[::-1]:
            if len(G.edge[u]) > 0:
                D[u] = np.stack(
                    [r+(node_reward[v]+E[v])*self.gamma for r,
                        v in zip(edge_reward[u], G.edge[u])]
                )
                E[u] = np.sum(action_prob[u]*D[u], axis=-2)
                if self.baseline_strategy == "random":
                    baseline = D[u].mean(axis=0, keepdims=True)
                elif self.baseline_strategy == "self":
                    baseline = E[u]
                    baseline = np.tile(baseline, D[u].shape[0]).reshape(
                        (D[u].shape[0], baseline.shape[0]))
                elif self.baseline_strategy == "zero":
                    pass
                else:
                    warnings.warn("unknown baseline strategy")
                D[u] -= baseline
                if self.extra_function=="log":
                    D[u] /= action_prob[u].clip(0.01, 1.0)
        return E, D

    def get_path(self, G: StaticDirectedAcyclicGraph, action_prob, start_node, strategy="argmax"):
        """Choose the edge with the highest probability to get a path.

        Args:
            G (StaticDirectedAcyclicGraph): The DAG.
            action_prob (array-like type): The probabilities that the agent chooses each edge. The way it is saved needs to be consistent with ``G.edge``. At each node u, ``action_prob[u]`` is a :math:`|N(u)|` dimension vector.
            start_node (any hashable type): The starting node.
            strategy (str): The strategy to choose actions. It can be ``argmax`` or ``probability``.

        Returns:
            path (list): The path determined by ``action_prob``.
        """
        u = G.node2index[start_node]
        path = [start_node]
        while len(G.edge[u]) > 0:
            if strategy == "argmax":
                action = action_prob[u].argmax()
            else:
                action = np.random.choice(
                    np.arange(action_prob[u].shape[0]),
                    p=action_prob[u]
                )
            u = G.edge[u][action]
            path.append(G.node[u])
        return path

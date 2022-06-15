import warnings


class StaticDirectedAcyclicGraph:
    """Static Directed Acyclic Graph (DAG).
    
    OASS is able to perform path planning on directed acyclic graphs. You can use this class to build a directed acyclic graph.

    Attributes:
        node2index (dict): The ``dict`` that maps each node name to node index.
        node (list): The node name of each node.
        edge (list(list(int))): All directed edges.
        n (int): The number of nodes.
    """
    def __init__(self):
        self.node2index = {}
        self.node = []
        self.edge = []
        self.n = 0

    def add_node(self, node):
        """Add a node to the graph.
        
        Args:
            node (any hashable type): The name of this node.

        .. note::
            If a node with the same name already exists in the graph, nothing will happen.
        """
        if node not in self.node2index:
            self.node2index[node] = len(self.node2index)
            self.node.append(node)
            self.edge.append([])
            self.n += 1

    def add_edge(self, u, v):
        """Add a directed edge <u, v> to the graph.
        
        Args:
            u (any hashable type): The start node of this edge.
            v (any hashable type): The end node of this edge.

        .. note::
            You must ensure that both nodes u and v exist in the graph.
        """
        u_index = self.node2index[u]
        v_index = self.node2index[v]
        self.edge[u_index].append(v_index)

    def topological_sort(self):
        """Calculate the topological ordering of nodes.
        
        Returns:
            topological_node (list): The topological sorting result of nodes.
        
        .. note::
            1. Please make sure that the constructed graph is acyclic, otherwise this function will give a warning.
            2. If the topological order of the nodes is not unique, we will sort the nodes in the order they are added.
        """
        indeg = [0]*self.n
        for u in range(self.n):
            for v in self.edge[u]:
                indeg[v] += 1
        start_node = [u for u in range(self.n) if indeg[u] == 0]
        self.topological_node = []
        while len(start_node) > 0:
            u = start_node.pop(-1)
            self.topological_node.append(u)
            for v in self.edge[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    start_node.append(v)
        if len(self.topological_node) != self.n:
            warnings.warn("This graph is not a DAG.")
        return self.topological_node

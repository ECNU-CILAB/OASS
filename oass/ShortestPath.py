from oass.StaticDirectedAcyclicGraph import StaticDirectedAcyclicGraph
import torch


class EdgeProbModel(torch.nn.Module):
    """
    A simple model designed for path planing.

    Attributes:
        G (StaticDirectedAcyclicGraph): The graph. This model will create a parameter representing the probability for each edge in G.
    """
    def __init__(self, G: StaticDirectedAcyclicGraph):
        super(EdgeProbModel, self).__init__()
        self.section = []
        edge_num = 0
        for u in range(G.n):
            neighbor_num = len(G.edge[u])
            self.section.append((edge_num, edge_num+neighbor_num))
            edge_num += neighbor_num
        self.edge_embedding = torch.nn.Parameter(
            torch.rand(edge_num, requires_grad=True)
        )

    def forward(self):
        """
        This model has no input layers.
        """
        action_prob = []
        for l, r in self.section:
            action_prob.append(torch.softmax(self.edge_embedding[l:r], dim=-1))
        return action_prob

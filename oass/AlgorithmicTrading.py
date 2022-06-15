from oass.StaticDirectedAcyclicGraph import StaticDirectedAcyclicGraph
from oass.GradientCalculator import GradientCalculator
import torch


class DecisionMakerOASS():
    '''The decision maker of OASS for algorithmic trading.
    
    Args:
        head_mask (int): ``head_mask``.
        tail_mask (int): ``tail_mask``.
        device (torch.device): The computing device.
    '''
    def __init__(self, head_mask, tail_mask, device=torch.device("cuda")):
        self.head_mask = head_mask
        self.tail_mask = tail_mask
        self.device = device

    def __call__(self, model, input_data):
        '''Get the decisions of a model.
    
        Args:
            model (torch.nn.Module): The neural network model.
            input_data (torch.tensor): A batch of sequencial environment states. The shape is ``(batch_size, sequence_length, input_size)``.
        
        Returns:
            signal_series (torch.tensor): A batch of sequencial actions.
        '''
        model.eval()
        input_data = torch.tensor(input_data).to(torch.float32).to(self.device)
        signal_data = []
        with torch.no_grad():
            output_data_batch = model(input_data)
            output_data_batch = torch.argmax(output_data_batch, axis=-1)
            output_data_batch = output_data_batch.tolist()
        for output_data_sequence in output_data_batch:
            signal_sequence = []
            signal = 1
            for i, action in enumerate(output_data_sequence):
                if i < self.head_mask or i >= self.tail_mask:
                    signal = 1
                else:
                    signal = action[signal]
                signal_sequence.append(signal)
            signal_data.append(signal_sequence)
        return torch.tensor(signal_data).to(torch.int64)


class TrainerOASS():
    '''The trainer of OASS for algorithmic trading.
    
    Args:
        head_mask (int): ``head_mask``.
        tail_mask (int): ``tail_mask``.
        sequence_length (int): The length of each sequence.
        buy_cost_pct (float): The cost ratio when the agent chooses to buy.
        sell_cost_pct (float): The cost ratio when the agent chooses to sell.
        gamma (float): The discount factor in the reward value function.
        device (torch.device): The computing device.

    Attributes:
        difficulty (float): ``difficulty``.
    '''

    def __init__(
        self,
        head_mask,
        tail_mask,
        sequence_length,
        buy_cost_pct=0.0001,
        sell_cost_pct=0.0001,
        gamma=1.0,
        device=torch.device("cuda")
    ):
        self.head_mask = head_mask
        self.tail_mask = tail_mask
        self.sequence_length = sequence_length
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma
        self.device = device
        self.build_graph()
        self.gradient_calculator = GradientCalculator(gamma=self.gamma)
        self.difficulty = 0.0

    def build_graph(self):
        """Build a DAG.
        
        Assuming that the ``sequence_length`` is :math:`n`, then there are :math:`3n` nodes in the graph. A node is represented as :math:`(i, j)`, where :math:`i\in\{0,1,\dots,n-1\}` represents the time step and :math:`j\in\{-1,0,1\}` represents the number of stocks held.
        """
        self.G = StaticDirectedAcyclicGraph()
        for i in range(self.sequence_length):
            for si in range(-1, 2):
                self.G.add_node((i, si))
        for i in range(self.sequence_length-1):
            for si in range(-1, 2):
                for si_ in range(-1, 2):
                    self.G.add_edge((i, si), (i+1, si_))

    def calculate_reward(self, buy_price_batch, sell_price_batch, buy_cost_pct, sell_cost_pct):
        r"""Calculate the node rewards and edge rewards.
        
        In this problem, these rewards are dynamic. The node reward represents the corresponding stock value at the last time step, and is zero at other time steps. The edge rewards represent the change of money. For example, the edge reward of :math:`\langle(i,0), (i+1,1)\rangle` is :math:`-p(1+\alpha)`, where :math:`p` is the price and :math:`\alpha` is the transaction cost ratio.

        Args:
            buy_price_batch (torch.Tensor): A batch of buy price. The shape is ``(batch_size, sequence_length)``.
            sell_price_batch (torch.Tensor): A batch of sell price. The shape is ``(batch_size, sequence_length)``.
            buy_cost_pct (float): The cost ratio when the agent chooses to buy.
            sell_cost_pct (float): The cost ratio when the agent chooses to sell.

        Returns:
            node_reward (torch.Tensor): The reward when arriving at each node. The shape is ``(sequence_length, 3, batch_size)``.
            edge_reward (torch.Tensor): The reward when passing each edge. The shape is ``(sequence_length, 3, batch_size)``.
        """
        sequence_length = buy_price_batch.shape[1]
        batch_size = buy_price_batch.shape[0]
        # price
        initial_price = ((buy_price_batch + sell_price_batch) /
                         2)[:, 0].repeat((sequence_length, 1)).T
        buy_price_batch = buy_price_batch/initial_price
        sell_price_batch = sell_price_batch/initial_price
        mid_price = (buy_price_batch + sell_price_batch)/2
        buy_price = mid_price + (buy_price_batch - mid_price)*self.difficulty
        sell_price = mid_price + (sell_price_batch - mid_price)*self.difficulty
        # node reward
        node_reward = torch.zeros(batch_size, sequence_length, 3)
        node_reward[:, -1, 0] = -buy_price[:, -1]
        node_reward[:, -1, 2] = sell_price[:, -1]
        node_reward = node_reward.reshape(batch_size, sequence_length*3)
        node_reward = torch.moveaxis(node_reward, 0, -1)
        # edge reward
        edge_reward = torch.zeros(batch_size, sequence_length, 3, 3)
        edge_reward[:, :, 0, 1] = -buy_price*(1+buy_cost_pct*self.difficulty)
        edge_reward[:, :, 1, 2] = -buy_price*(1+buy_cost_pct*self.difficulty)
        edge_reward[:, :, 0, 2] = -2*buy_price*(1+buy_cost_pct*self.difficulty)
        edge_reward[:, :, 1, 0] = sell_price*(1-sell_cost_pct*self.difficulty)
        edge_reward[:, :, 2, 1] = sell_price*(1-sell_cost_pct*self.difficulty)
        edge_reward[:, :, 2, 0] = 2*sell_price*(1-sell_cost_pct*self.difficulty)
        edge_reward = edge_reward.reshape(batch_size, sequence_length*3, 3)
        edge_reward = torch.moveaxis(edge_reward, 0, -1)
        return node_reward, edge_reward

    def train_epoch(self, model, data_loader, optimizer, **kwargs):
        '''Train the model for an epoch.
    
        Args:
            model (torch.nn.Module): The neural network model.
            data_loader (torch.utils.data.DataLoader): The data loader which contains the ``train`` dataset.
            optimizer (torch.optim.Adam): The optimizer provided by PyTorch.

        Returns:
            sum_loss, sum_reward (float, float): The average loss and reward of each sequence in the data loader.
        '''
        model.train()
        sum_loss, sum_reward, data_amount = 0, 0, 0
        for batch_id, data_batch in enumerate(data_loader):
            # get data from data_loader
            observation_batch, buy_price_batch, sell_price_batch = data_batch
            observation_batch = observation_batch.to(self.device)
            # action probability
            action_prob = model(observation_batch)
            action_prob = action_prob.reshape((
                action_prob.shape[0], action_prob.shape[1]*3, 3
            ))
            action_prob = torch.moveaxis(action_prob, 0, -1)
            # reward
            node_reward, edge_reward = self.calculate_reward(
                buy_price_batch, sell_price_batch, self.buy_cost_pct, self.sell_cost_pct
            )
            # gradient
            E, D = self.gradient_calculator.calculate_gradient(
                self.G,
                action_prob.cpu().detach().numpy(),
                node_reward.numpy(),
                edge_reward.numpy()
            )
            # loss
            loss = 0
            for d, p in zip(D, action_prob):
                if d.shape[0] != 0:
                    loss += torch.sum(-torch.tensor(d).to(self.device)*p)
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            sum_loss += loss.tolist()
            sum_reward += E[self.G.node2index[(0, 0)]].sum()
            data_amount += observation_batch.shape[0]
        sum_loss /= data_amount
        sum_reward /= data_amount
        return sum_loss, sum_reward

    def test_epoch(self, model, data_loader, **kwargs):
        '''Test the model for an epoch.
    
        Args:
            model (torch.nn.Module): The neural network model.
            data_loader (torch.utils.data.DataLoader): The data loader which contains the ``dev`` (or ``test``) dataset.

        Returns:
            sum_loss, sum_reward (float, float): The average loss and reward of each sequence in the data loader.
        '''
        model.eval()
        sum_loss, sum_reward, data_amount = 0, 0, 0
        for batch_id, data_batch in enumerate(data_loader):
            with torch.no_grad():
                # get data from data_loader
                observation_batch, buy_price_batch, sell_price_batch = data_batch
                observation_batch = observation_batch.to(self.device)
                # action probability
                action_prob = model(observation_batch)
                action_prob = action_prob.reshape((
                    action_prob.shape[0], action_prob.shape[1]*3, 3
                ))
                action_prob = torch.moveaxis(action_prob, 0, -1)
                # reward
                node_reward, edge_reward = self.calculate_reward(
                    buy_price_batch, sell_price_batch, self.buy_cost_pct, self.sell_cost_pct
                )
                # gradient
                E, D = self.gradient_calculator.calculate_gradient(
                    self.G,
                    action_prob.cpu().detach().numpy(),
                    node_reward.numpy(),
                    edge_reward.numpy()
                )
                # loss
                loss = 0
                for i, d, p in zip(range(len(D)), D, action_prob):
                    # head_mask and tail_mask
                    if i<self.head_mask*3 or i>=self.tail_mask*3:
                        continue
                    if d.shape[0] != 0:
                        loss += torch.sum(-torch.tensor(d).to(self.device)*p)
                # log
                sum_loss += loss.tolist()
                sum_reward += E[self.G.node2index[(0, 0)]].sum()
                data_amount += observation_batch.shape[0]
        sum_loss /= data_amount
        sum_reward /= data_amount
        return sum_loss, sum_reward

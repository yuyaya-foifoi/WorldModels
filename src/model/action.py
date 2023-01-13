import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import functional as F


class ActionModel(nn.Module):
    """
    低次元の状態表現(state_dim + rnn_hidden_dim)から行動を計算するクラス
    """

    def __init__(
        self,
        state_dim,
        rnn_hidden_dim,
        action_dim,
        hidden_dim=400,
        act=F.elu,
        min_stddev=1e-4,
        init_stddev=5.0,
    ):
        super(ActionModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_stddev = nn.Linear(hidden_dim, action_dim)
        self.act = act
        self.min_stddev = min_stddev
        self.init_stddev = np.log(np.exp(init_stddev) - 1)

    def forward(self, state, rnn_hidden, training=True):
        """
        training=Trueなら, NNのパラメータに関して微分可能な形の行動のサンプル（Reparametrizationによる）を返します
        training=Falseなら, 行動の確率分布の平均値を返します
        """
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        hidden = self.act(self.fc4(hidden))

        # Dreamerの実装に合わせて少し平均と分散に対する簡単な変換が入っています
        mean = self.fc_mean(hidden)
        mean = 5.0 * torch.tanh(mean / 5.0)
        stddev = self.fc_stddev(hidden)
        stddev = F.softplus(stddev + self.init_stddev) + self.min_stddev

        if training:
            action = torch.tanh(
                Normal(mean, stddev).rsample()
            )  # 微分可能にするためrsample()
        else:
            action = torch.tanh(mean)
        return action

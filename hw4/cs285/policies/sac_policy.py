from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: get this from previous HW
        entropy = self.log_alpha.exp()
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: get this from previous HW
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        
        observation = ptu.from_numpy(observation)
        batch_mean = self.mean_net(observation)
        # out = self.mean_log_std_net(observation)
        # batch_mean, batch_scale = torch.split(out, self.ac_dim, dim=1)
        batch_scale = torch.clamp(self.logstd, self.log_std_bounds[0], self.log_std_bounds[1])
        
        if sample:
            batch_scale = torch.exp(self.logstd)
            action = sac_utils.SquashedNormal(
                batch_mean,
                batch_scale,
            ).sample()
            return ptu.to_numpy(action)
        else:
            return ptu.to_numpy(batch_mean)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from previous HW
        batch_mean = self.mean_net(observation)
        batch_scale = torch.clamp(self.logstd, self.log_std_bounds[0], self.log_std_bounds[1])
        action_distribution = sac_utils.SquashedNormal(
            batch_mean,
            batch_scale.exp()
        )
        action = action_distribution.rsample()
        log_prob = action_distribution.log_prob(action).sum(dim=1)
        return action, log_prob

    def update(self, obs, critic):
        # TODO: get this from previous HW_alpha_optimizer.step()
        obs = ptu.from_numpy(obs)
        sampled_action, log_prob = self.forward(obs)
        q1_values, q2_values = critic.forward(obs, sampled_action)
        min_q_values = torch.min(q1_values, q2_values)
        actor_loss = - min_q_values + self.alpha * log_prob
        actor_loss = actor_loss.mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        alpha_loss = - self.alpha * (log_prob + self.target_entropy).detach()
        alpha_loss = alpha_loss.mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return actor_loss, alpha_loss, self.alpha
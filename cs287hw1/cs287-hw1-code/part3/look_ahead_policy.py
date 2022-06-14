import numpy as np
from gym import spaces
from part2.look_ahead_policy import LookAheadPolicy as BaseLookAheadPolicy


class LookAheadPolicy(BaseLookAheadPolicy):
    """
    Look ahead policy

    -- UTILS VARIABLES FOR RUNNING THE CODE --
    * look_ahead_type (str): Type of look ahead policy to use

    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
    * self.horizon (int): Horizon for the look ahead policy

    * act_dim (int): Dimension of the state space

    * self.num_elites (int): number of best actions to pick for the cross-entropy method

    * self.value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states

    * self.get_returns_state(state): It is the same that you implemented in the previous part
    """
    def __init__(self,
                 env,
                 value_fun,
                 horizon,
                 look_ahead_type='tabular',
                 num_acts=20,
                 cem_itrs=10,
                 precent_elites=0.25,
                 ):
        self.env = env
        self.discount = self.env.discount
        self._value_fun = value_fun
        self.horizon = horizon
        self.num_acts = num_acts
        self.cem_itrs = cem_itrs
        self.num_elites = int(num_acts * precent_elites)
        assert self.num_elites > 0
        self.look_ahead_type = look_ahead_type

    def get_action(self, state):
        if self.look_ahead_type == 'tabular':
            action = self.get_action_tabular(state)
        elif self.look_ahead_type == 'rs':
            action = self.get_action_rs(state)
        elif self.look_ahead_type == 'cem':
            action = self.get_action_cem(state)
        else:
            raise NotImplementedError
        return action

    def get_action_cem(self, state):
        """
        Do lookahead in the continous and discrete case with the cross-entropy method..
        :param state (int if discrete np.ndarray if continous)
        :return: best_action (int if discrete np.ndarray if continous)
        """
        num_acts = self.num_acts
        """ INSERT YOUR CODE HERE"""
        if isinstance(self.env.action_space, spaces.Discrete):
            act_dim = self.env.action_space.n
            actions = np.arange(act_dim)
            act_seqs = np.random.choice(actions, size=(self.horizon, num_acts))
            for _ in range(self.cem_itrs):
                act_stack = []
                returns = self.get_returns(state, act_seqs)
                idx_elites = returns.argsort()[::-1][:self.num_elites]
                for i in range(self.horizon):
                    acts = np.random.choice(act_seqs[i][idx_elites], size=(num_acts,))
                    act_stack.append(acts)
                act_seqs = np.array(act_stack)
            best_action = np.random.choice(act_seqs[0])
        else:
            act_low, act_high = self.env.action_space.low, self.env.action_space.high
            act_dim = act_low.shape[0]
            act_seqs = np.random.uniform(act_low, act_high, size=(self.horizon, num_acts, act_dim))
            for _ in range(self.cem_itrs):
                act_stack = []
                returns = self.get_returns(state, act_seqs)
                idx_elites = returns.argsort()[::-1][:self.num_elites]
                acts_mean = act_seqs[:, idx_elites].mean(axis=1)
                acts_std = act_seqs[:, idx_elites].std(axis=1)
                for i in range(self.horizon):
                    acts = np.random.normal(acts_mean[i], acts_std[i], size=(num_acts, act_dim))
                    act_stack.append(acts)
                act_seqs = np.array(act_stack)
            best_action = acts_mean[0]
                # raise NotImplementedError
            """ Your code ends here """
        return best_action

    def get_action_rs(self, state):
        """
        Do lookahead in the continous and discrete case with random shooting..
        :param state (int if discrete np.ndarray if continous)
        :return: best_action (int if discrete np.ndarray if continous)
        """
        num_acts = self.num_acts
        """ INSERT YOUR CODE HERE """
        if isinstance(self.env.action_space, spaces.Discrete):
            act_dim = self.env.action_space.n
            act_seqs = np.random.choice(np.arange(act_dim), size=(self.horizon, num_acts))
            returns = self.get_returns(state, act_seqs)
            best_idx = np.argmax(returns)
            best_action = act_seqs[0][best_idx]
            # raise NotImplementedError
        else:
            assert num_acts is not None
            act_dim = self.env.action_space.low.shape[0]
            act_seqs = np.random.uniform(self.env.action_space.low, self.env.action_space.high, size=(self.horizon, num_acts, act_dim))
            returns = self.get_returns(state, act_seqs)
            best_idx = np.argmax(returns)
            best_action = act_seqs[0][best_idx]
            # raise NotImplementedError

        return best_action

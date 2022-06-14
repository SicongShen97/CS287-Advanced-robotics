import numpy as np
from gym import spaces


class LookAheadPolicy(object):
    """
    Look ahead policy

    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
    * self.horizon (int): Horizon for the look ahead policy

    * act_dim (int): Dimension of the state space

    * value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states
    * env (Env):
                - vec_set_state(states): vectorized (multiple environments in parallel) version of reseting the
                environment to a state for a batch of states.
                - vec_step(actions): vectorized (multiple environments in parallel) version of stepping through the
                environment for a batch of actions. Returns the next observations, rewards, dones signals, env infos
                (last not used).
    """
    def __init__(self,
                 env,
                 value_fun,
                 horizon,
                 ):
        self.env = env
        self.discount = env.discount
        self._value_fun = value_fun
        self.horizon = horizon

    def get_action(self, state):
        """
        Get the best action by doing look ahead, covering actions for the specified horizon.
        HINT: use np.meshgrid to compute all the possible action sequences.
        :param state:
        :return: best_action (int)
           """
        assert isinstance(self.env.action_space, spaces.Discrete)
        act_dim = self.env.action_space.n
        """ INSERT YOUR CODE HERE"""
        actions = np.tile(np.arange(act_dim), (self.horizon, 1))
        actions_comb = np.array(np.meshgrid(*actions)).T.reshape(-1, self.horizon)

        returns = self.get_returns(state, actions_comb.T)
        idx = np.argmax(returns)
        best_action = actions_comb[idx][0]

        return best_action

    def get_returns(self, state, actions):
        """
        :param state: current state of the policy
        :param actions: array of actions of shape [horizon, num_acts]
        :return: returns for the specified horizon + self.discount ^ H value_fun
        HINT: Make sure to take the discounting and done into acount!
        """
        assert self.env.vectorized
        """ INSERT YOUR CODE HERE"""
        n_comb = actions.shape[1]
        returns = np.zeros(n_comb)
        states = np.array([state for _ in range(n_comb)])
        for i in range(self.horizon):
            self.env.vec_set_state(states)
            states, rewards, dones, infos = self.env.vec_step(actions[i])
            returns += self.discount**i*rewards
        returns += self.discount**self.horizon*self._value_fun.get_values(states)*(1-dones)

        return returns

    def update(self, actions):
        pass

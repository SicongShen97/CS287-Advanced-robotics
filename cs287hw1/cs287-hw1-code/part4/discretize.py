import numpy as np
from utils.utils import DiscretizeWrapper


class Discretize(DiscretizeWrapper):
    """
    Discretize class: Discretizes a continous gym environment


    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
        * self.state_points (np.ndarray): grid that contains the real values of the discretization

        * self.obs_n (int): number of discrete points

        * self.transitions (np.ndarray): transition matrix of size (S+1, A, S+1). The last state corresponds to the sink
                                         state
        * self.rewards (np.ndarray): reward matrix of size (S+1, A, S+1). The last state corresponds to the sink state

        * self.get_id_from_coordinates(coordinate_vector) returns the id of the coordinate_vector

        * self.get_state_from_id(id_s): get the continuous state associated to that state id

        * self.get_action_from_id(id_a): get the contiouns action associated to that action id

        * env.set_state(s): resets the environment to the continous state s

        * env.step(a): applies the action a to the environment. Returns next_state, reward, done, env_infos. The
                            last element is not used.
    """

    def vec_get_discrete_state_from_cont_state(self, cont_state):
        """
        Get discrete state from continuous state.
            * self.mode (str): specifies if the discretization is to the nearest-neighbour (nn) or n-linear (linear).

        :param cont_state (np.ndarray): of shape env.observation_space.shape
        :return: A tuple of (states, probs). states is a np.ndarray of shape (1,) if mode=='nn'
                and (2 ^ obs_dim,) if mode=='linear'. probs is the probabability of going to such states,
                it has the same size than states.
        """
        """INSERT YOUR CODE HERE"""
        cont_state = np.expand_dims(cont_state, axis=-1)
        obs_dim = cont_state.shape[0]
        if self.mode == 'nn':
            idx = np.argmin(np.abs(self.state_points - cont_state), axis=1)
            states = np.array([self.get_id_from_coordinates(idx)])
            probs = np.array([1])
            # raise NotImplementedError

        elif self.mode == 'linear':
            upper_index = np.argmax(self.state_points > cont_state, axis=1)
            upper_index[np.sum(self.state_points > cont_state) == 0] = obs_dim
            lower_index = upper_index - 1

            too_small_i = np.sum(self.state_points < cont_state, axis=1) == 0
            too_big_i = np.sum(self.state_points > cont_state, axis=1) == 0

            lower_states = self.state_points[np.arange(obs_dim), lower_index]
            upper_states = self.state_points[np.arange(obs_dim), upper_index]

            lower_p = (upper_states - cont_state[:, 0]) / (upper_states - lower_states)
            upper_p = 1 - lower_p

            comb_index = np.column_stack([lower_index, upper_index])
            comb_p = np.column_stack([lower_p, upper_p])
            comb_p[too_small_i] = [0, 1]
            comb_p[too_big_i] = [1, 0]

            indexs = np.array(np.meshgrid(*comb_index)).T.reshape(-1, obs_dim)
            ps = np.array(np.meshgrid(*comb_p)).T.reshape(-1, obs_dim)

            states = self.get_id_from_coordinates(indexs)
            probs = np.prod(ps, axis=1)
            # raise NotImplementedError
        else:
            raise NotImplementedError
        return states, probs

    def vec_add_transition(self, id_s, id_a):
        """
        Populates transition and reward matrix (self.transition and self.reward)
        :param id_s (int): discrete index of the the state
        :param id_a (int): discrete index of the the action

        """
        env = self._wrapped_env
        obs_n = self.obs_n

        states = self.get_state_from_id(id_s)
        actions = self.get_action_from_id(id_a)
        for i in range(len(states)):
            env.set_state(states[i])
            next_state, reward, done, env_infos = env.step(actions[i])
            if done:
                self.transitions[id_s[i], id_a[i], obs_n] = 1
                self.rewards[id_s[i], id_a[i], obs_n] = reward
            else:
                next_state_idxs, action_probs = self.vec_get_discrete_state_from_cont_state(next_state)
                for j in range(len(next_state_idxs)):
                    self.transitions[id_s[i], id_a[i], next_state_idxs[j]] = action_probs[j]
                    self.rewards[id_s[i], id_a[i], next_state_idxs[j]] = reward
        """INSERT YOUR CODE HERE"""
        # raise NotImplementedError

    def add_done_transitions(self):
        """
        Populates transition and reward matrix for the sink state (self.transition and self.reward). The sink state
        corresponds to the last state (self.obs_n or -1).
        """
        """INSERT YOUR CODE HERE"""
        self.transitions[-1, :, -1] = 1
        self.rewards[-1, :, -1] = 0
        # raise NotImplementedError




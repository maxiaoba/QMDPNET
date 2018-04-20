import os, random, shutil
import numpy as np, scipy.sparse

from utils import dijkstra
from utils.qmdp import QMDP

# try:
#     import ipdb as pdb
# except Exception:
#     import pdb
import pdb

import zmq
import gym
from rllab.envs.base import Env
from rllab.spaces import Box, Discrete
from rllab.envs.base import Step
import numpy as np
from rllab.core.serializable import Serializable

from matplotlib import pyplot
import matplotlib as mpl
from rllab.misc.overrides import overrides

FREESTATE = 0.0
OBSTACLE = 1.0


class GridBase(Env):
    def __init__(self, params, grid=None, b0=None, start_state=None, goal_state=None):
        """
        Initialize domain simulator
        :param params: domain descriptor dotdict
        :param db: pytable database file
        """
        self.params = params

        self.N = params['grid_n']
        self.M = params['grid_m']
        self.grid_shape = [self.N, self.M]
        self.moves = params['moves']
        self.observe_directions = params['observe_directions']

        self.num_action = params['num_action']
        self.num_obs = params['num_obs']
        self.obs_len = len(self.observe_directions)
        self.num_state = self.N * self.M

        self.grid = grid
        self.b0 = b0
        self.start_state = start_state
        self.goal_state = goal_state
        if grid is not None:
            self.gen_pomdp()

        self.generate_grid = False
        self.generate_b0_start_goal = False

        self.act = params['stayaction']
        # self.fig = None
    def render(self):
        # pass
        pyplot.clf()
        env_img = self.env_img
        goal_img = self.goal_img
        b0_img = self.b0_img
        state = self.state

        show_img = np.copy(env_img)
        start_coord = self.state_lin_to_bin(self.start_state)
        show_img[start_coord[0]][start_coord[1]] = 2

        show_img = show_img + 3 * goal_img

        current_coord = self.state_lin_to_bin(state)
        show_img[current_coord[0]][current_coord[1]] = 4
        # make a color map of fixed colors
        cmap = mpl.colors.ListedColormap(['white','black','red','blue','yellow'])
        bounds=[-0.5,0.5,1.5,2.5,3.5,4.5]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # tell imshow about color map so that only set colors are used
        # pyplot.ion()
        fig = pyplot.figure(1)
        ax = fig.add_subplot(111)
        img = ax.imshow(show_img,interpolation='nearest',
                            cmap = cmap,norm=norm)

        # make a color bar
        pyplot.colorbar(img,cmap=cmap,
                        norm=norm,boundaries=bounds,ticks=[0,1,2,3,4])
        ax.set_title('step: '+str(self.step_count)+' action: '+str(self.act))
        # print(show_img)
        # self.fig.savefig('step'+str(self.step_count)+'.png')

    @property
    def observation_space(self):
        return Box(low=-np.ones(self.params['obs_len']), high=np.ones(self.params['obs_len']))

    @property
    def action_space(self):
        return Discrete(self.num_action)
        # return Box(low=np.zeros(1), high=np.ones(1)*(self.num_action-1))

    def close(self):
        pass

    def reset(self):
        done = False
        while not done:
            if self.generate_grid:
                self.grid = self.random_grid(self.params['grid_n'], self.params['grid_m'], self.params['Pobst'])
                self.gen_pomdp()  # generates pomdp model, self.T, self.Z, self.R
            if self.generate_grid or self.generate_b0_start_goal:
                while True:
                    # sample initial belief, start, goal
                    b0, start_state, goal_state = self.gen_start_and_goal()
                    if b0 is None:
                        assert generate_grid
                        break  # regenerate obstacles
                    self.b0 = b0
                    self.start_state = start_state
                    self.goal_state = goal_state
                    goal_states = [self.goal_state]
                    # reject if start == goal
                    if start_state in goal_states:
                        continue
                    else:
                        done = True
                        break
            else:
                break

        self.state = self.start_state
        # create qmdp
        goal_states = [self.goal_state]
        self.qmdp = self.get_qmdp(goal_states)  # makes soft copies from self.T{R,Z}simple
        # it will also convert to csr sparse, and set qmdp.issparse=True

        obs = self.qmdp.random_obs(self.state, self.params['stayaction'])
        obs = self.obs_lin_to_bin(obs)
        self.env_img = np.squeeze(self.grid[None], axis=0)
        self.goal_img = self.process_goals(self.goal_state)
        self.b0_img = self.process_beliefs(self.b0)
        self.step_count = 0
        return obs

    def step(self, act):
        self.state, r = self.qmdp.transition(self.state, act)
        obs = self.qmdp.random_obs(self.state, act)
        obs = self.obs_lin_to_bin(obs)
        done = False
        self.act = act
        self.step_count = self.step_count + 1
        # if np.isclose(r, self.params['R_obst']) or self.step_count > self.params['traj_limit']:
        current_coord = self.state_lin_to_bin(self.state)
        if self.step_count > self.params['traj_limit'] or self.goal_img[current_coord[0]][current_coord[1]]==1:
            done = True
        return Step(observation=obs, reward=r, done=done)

    @overrides
    def get_param_values(self):
        return vars(self)

    @overrides
    def set_param_values(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def gen_pomdp(self):
        # construct all POMDP model(R, T, Z)
        self.Z = self.build_Z()
        self.T, Tml, self.R = self.build_TR()

        # transform into graph with opposite directional actions, so we can compute path from goal
        G = {i: {} for i in range(self.num_state)}
        for a in range(self.num_action):
            for s in range(self.num_state):
                snext = Tml[s, a]
                if s != snext:
                    G[snext][s] = 1  # edge with distance 1
        self.graph = G

    def build_Z(self):
        params = self.params

        Pobs_succ = params['Pobs_succ']

        Z = np.zeros([self.num_action, self.num_state, self.num_obs], 'f')

        for i in range(self.N):
            for j in range(self.M):
                state_coord = np.array([i, j])
                state = self.state_bin_to_lin(state_coord)

                # first build observation
                obs = np.zeros([self.obs_len])  # 1 or 0 in four directions
                for direction in range(self.obs_len):
                    neighb = self.apply_move(state_coord, np.array(self.observe_directions[direction]))
                    if self.check_free(neighb):
                        obs[direction] = 0
                    else:
                        obs[direction] = 1

                # add all observations with their probabilities
                for obs_i in range(self.num_obs):
                    dist = np.abs(self.obs_lin_to_bin(obs_i) - obs).sum()
                    prob = np.power(1.0 - Pobs_succ, dist) * np.power(Pobs_succ, self.obs_len - dist)
                    Z[:, state, obs_i] = prob

                # sanity check
                assert np.isclose(1.0, Z[0, state, :].sum())

        return Z

    def build_TR(self):
        """
        Builds transition (T) and reward (R) model for a grid.
        The model does not capture goal states, which must be incorporated later.
        :return: transition model T, maximum likely transitions Tml, reward model R
        """
        params = self.params
        Pmove_succ = params['Pmove_succ']

        # T, R does not capture goal state, it must be incorporated later
        T = [scipy.sparse.lil_matrix((self.num_state, self.num_state), dtype='f')
             for x in range(self.num_action)]  # probability of transition with a0 from s1 to s2
        R = [scipy.sparse.lil_matrix((self.num_state, self.num_state), dtype='f')
             for x in range(self.num_action)]  # probability of transition with a0 from s1 to s2
        # goal will be defined as a terminal state, all actions remain in goal with 0 reward

        # maximum likely versions
        Tml = np.zeros([self.num_state, self.num_action], 'i')  # Tml[s, a] --> next state
        Rml = np.zeros([self.num_state, self.num_action], 'f')  # Rml[s, a] --> reward after executing a in s

        for i in range(self.N):
            for j in range(self.M):
                state_coord = np.array([i, j])
                state = self.state_bin_to_lin(state_coord)

                # build T and R
                for act in range(self.num_action):
                    neighbor_coord = self.apply_move(state_coord, np.array(self.moves[act]))
                    if self.check_free(neighbor_coord):
                        Rml[state, act] = params['R_step'][act]
                    else:
                        neighbor_coord[:2] = [i, j]  # dont move if obstacle or edge of world
                        # alternative: neighbor_coord = state_coord
                        Rml[state, act] = params['R_obst']

                    neighbor = self.state_bin_to_lin(neighbor_coord)
                    Tml[state, act] = neighbor
                    if state == neighbor:
                        # shortcut if didnt move
                        R[act][state, state] = Rml[state, act]
                        T[act][state, state] = 1.0
                    else:
                        R[act][state, state] = params['R_step'][act]
                        # cost if transition fails (might be lucky and avoid wall)
                        R[act][state, neighbor] = Rml[state, act]
                        T[act][state, state] = 1.0 - Pmove_succ
                        T[act][state, neighbor] = Pmove_succ

        return T, Tml, R

    def gen_start_and_goal(self, maxtrials=1000):
        """
        Pick an initial belief, initial state and goal state randomly
        """
        free_states = np.nonzero((self.grid == FREESTATE).flatten())[0]
        freespace_size = len(free_states)

        for trial in range(maxtrials):
            b0sizes = np.floor(freespace_size / np.power(2.0, np.arange(20)))
            b0sizes = b0sizes[:np.nonzero(b0sizes < 1)[0][0]]
            b0size = int(np.random.choice(b0sizes))

            b0ind = np.random.choice(free_states, b0size, replace=False)
            b0 = np.zeros([self.num_state])
            b0[b0ind] = 1.0 / b0size  # uniform distribution over sampled states

            # sanity check
            for state in b0ind:
                coord = self.state_lin_to_bin(state)
                assert self.check_free(coord)

            # sample initial state from initial belief
            start_state = np.random.choice(self.num_state, p=b0)

            # sample goal uniformly from free space
            goal_state = np.random.choice(free_states)

            # check if path exists from start to goal, if not, pick a new set
            D, path_pointers = dijkstra.Dijkstra(self.graph, goal_state)  # map of distances and predecessors
            if start_state in D:
                break
        else:
            # never succeeded
            raise ValueError

        return b0, start_state, goal_state

    def get_qmdp(self, goal_states):
        qmdp = QMDP(self.params)

        qmdp.processT(self.T)  # this will make a hard copy
        qmdp.processR(self.R)
        qmdp.processZ(self.Z)

        qmdp.set_terminals(goal_states, reward=self.params['R_goal'])

        qmdp.transfer_all_sparse()
        return qmdp

    @staticmethod
    def sample_free_state(map):
        """
        Return the coordinates of a random free state from the 2D input map
        """
        while True:
            coord = [random.randrange(map.shape[0]), random.randrange(map.shape[1])]
            if map[coord[0],coord[1],0] == FREESTATE:
                return coord

    @staticmethod
    def outofbounds(map, coord):
        return (coord[0] < 0 or coord[0] >= map.shape[0] or coord[1] < 0 or coord[1] >= map.shape[1])

    @staticmethod
    def apply_move(coord_in, move):
        coord = coord_in.copy()
        coord[:2] += move[:2]
        return coord

    def check_free(self, coord):
        return (not GridBase.outofbounds(self.grid, coord) and self.grid[coord[0], coord[1]] != OBSTACLE)

    @staticmethod
    def random_grid(N, M, Pobst):
        grid = np.zeros([N, M])

        # borders
        grid[0, :] = OBSTACLE
        grid[-1, :] = OBSTACLE
        grid[:, 0] = OBSTACLE
        grid[:, -1] = OBSTACLE

        rand_field = np.random.rand(N, M)
        grid = np.array(np.logical_or(grid, (rand_field < Pobst)), 'i')
        return grid

    def obs_lin_to_bin(self, obs_lin):
        obs = np.array(np.unravel_index(obs_lin, [2,2,2,2]), 'i')
        if obs.ndim > 2:
            raise NotImplementedError
        elif obs.ndim > 1:
            obs = np.transpose(obs, [1,0])
        return obs

    def obs_bin_to_lin(self, obs_bin):
        return np.ravel_multi_index(obs_bin, [2,2,2,2])

    def state_lin_to_bin(self, state_lin):
        return np.unravel_index(state_lin, self.grid_shape)

    def state_bin_to_lin(self, state_coord):
        return np.ravel_multi_index(state_coord, self.grid_shape)

    def process_goals(self, goal_state):
        """
        :param goal_state: linear goal state
        :return: goal image, same size as grid
        """
        goal_img = np.zeros([self.N, self.M], 'i')
        goalidx = np.unravel_index(goal_state, [self.N, self.M])

        goal_img[goalidx[0], goalidx[1]] = 1

        return goal_img

    def process_beliefs(self, linear_belief):
        """
        :param linear_belief: belief in linear space
        :return: belief reshaped to grid size
        """
        b = linear_belief.reshape([self.params['grid_n'], self.params['grid_m'], ])
        if b.dtype != np.float:
            return b.astype('f')

        return b

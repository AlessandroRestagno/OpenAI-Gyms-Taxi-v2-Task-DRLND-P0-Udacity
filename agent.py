import numpy as np
import random
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        #self.Q1 = defaultdict(lambda: np.zeros(self.nA))
        #self.Q2 = defaultdict(lambda: np.zeros(self.nA))

        self.epsilon = 0.001
        self.alpha = 0.2
        self.gamma = 1.0
        self.n = 8
        #self.episode = 0

        print('Epsilon: {}, Alpha = {}'.format(self.epsilon ,self.alpha) )

    def epsilon_greedy_probs(self, Q_s, epsilon):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """

        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        '''
        if self.episode > 1000:
            self.epsilon = 0.001
        self.episode = self.episode + 1
        print('Epsilon: {}, Alpha = {}, Episode = {}'.format(self.epsilon ,self.alpha,self.episode) )
        '''

        state_policy = self.epsilon_greedy_probs(self.Q[state] , self.epsilon)
        #Double expected SARSA
        #state_policy = self.epsilon_greedy_probs((self.Q1[state] + self.Q2[state]) , self.epsilon)

        action = np.random.choice(np.arange(self.nA), p=state_policy)

        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        #Q-learning (sarsamax) or Expected SARSA
        previous_Q = self.Q[state][action]
        #previous_Q1 = self.Q1[state][action]
        #previous_Q2 = self.Q2[state][action]


        # Sarsamax or Q-learning
        #self.Q[state][action] = old_Q + (self.alpha * (reward + (self.gamma * np.max(self.Q[next_state]) - old_Q)))
        # Expected SARSA
        self.Q[state][action] = previous_Q + (self.alpha * (reward + (self.gamma * ((1 - self.epsilon) * np.max(self.Q[next_state]) + (self.epsilon / self.nA * sum(self.Q[next_state][action] for action in range (self.nA))))) - previous_Q))

        '''
        #Double Expected SARSA
        r = random.random()
        if r < 0:
            self.Q1[state][action] = previous_Q1 + (self.alpha * (reward + (self.gamma * ((1 - self.epsilon) * np.max(self.Q1[next_state]) + (self.epsilon / self.nA * sum(self.Q1[next_state][action] for action in range (self.nA))))) - previous_Q1))
        else:
            self.Q2[state][action] = previous_Q2 + (self.alpha * (reward + (self.gamma * ((1 - self.epsilon) * np.max(self.Q2[next_state]) + (self.epsilon / self.nA * sum(self.Q2[next_state][action] for action in range (self.nA))))) - previous_Q2))
        '''

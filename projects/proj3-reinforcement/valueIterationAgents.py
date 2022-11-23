# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        for i in range(self.iterations):
            updatedValues = self.values.copy()  # to use batch-version of MDP , hard copy the values
            for state in mdp.getStates():
                if mdp.isTerminal(state):
                    continue
                action = self.computeActionFromValues((state))
                updatedValues[state] = self.computeQValueFromValues(state, action)
            self.values = updatedValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        nextStateInfo = mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0
        for nextState, prob in nextStateInfo:
            reward = mdp.getReward(state, action, nextState)
            qValue += prob*(reward + self.discount*self.values[nextState])

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        actions = mdp.getPossibleActions(state)
        bestAction = None
        maxQ = float("-Inf")

        for action in actions:
            qValue = self.computeQValueFromValues(state, action)
            if qValue > maxQ:
                maxQ = qValue
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        states = mdp.getStates()
        for i in range(self.iterations):
            state = states[i % len(states)]
            if mdp.isTerminal(state):
                continue
            action = self.computeActionFromValues((state))
            self.values[state] = self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue()
        mdp = self.mdp
        states = mdp.getStates()
        for state in states:
            if not mdp.isTerminal(state):
                maxQ = self.findMaxQ(state)
                diff = abs(self.values[state] - maxQ)
                pq.push(state, -1*diff)

        for _ in range(self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()
            action = self.computeActionFromValues(state)
            self.values[state] = self.computeQValueFromValues(state, action)
            predecessor = self.findPre(state)
            for p in predecessor:
                if not mdp.isTerminal(p):
                    maxQ = self.findMaxQ(p)
                    diff = abs(self.values[p]-maxQ)
                    if diff > self.theta:
                        pq.update(p, -1*diff)

    def findPre(self, state):
        res = set()
        mdp = self.mdp
        states = mdp.getStates()
        for s in states:
            if mdp.isTerminal(s):
                continue
            actions = mdp.getPossibleActions(s)
            for action in actions:
                nextSInfo = mdp.getTransitionStatesAndProbs(s, action)
                for nextS, prob in nextSInfo:
                    if nextS == state:
                        res.add(s)
                        break
                else:
                    continue
        return res

    def findMaxQ(self, state):
        actions = self.mdp.getPossibleActions(state)
        maxQ = float("-Inf")
        for action in actions:
            qval = self.computeQValueFromValues(state, action)
            maxQ = max(maxQ, qval)
        return maxQ
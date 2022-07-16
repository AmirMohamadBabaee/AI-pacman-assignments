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
from sys import maxsize

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
        for _ in range(self.iterations):
            values_copy = self.values.copy()
            for state in self.mdp.getStates():
                legalActions = self.mdp.getPossibleActions(state)
                max_value = -maxsize
                for action in legalActions:
                    current_value = self.computeQValueFromValues(state, action)
                    if current_value > max_value:
                        max_value = current_value
                if max_value > -maxsize:
                    values_copy[state] = max_value
            self.values = values_copy
            

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
        # print(f'{self.mdp.getTransitionStatesAndProbs(state, action)=}')
        nextStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        sum = 0
        for nextState, prob in nextStatesAndProbs:
            sum += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))
        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # print(f'{state=}')
        if not self.mdp.isTerminal(state):
            legalActions = self.mdp.getPossibleActions(state)
            # print(f'{legalActions=}')
            max_value = -maxsize
            max_action = None
            for action in legalActions:
                current_value = self.computeQValueFromValues(state, action)
                if current_value > max_value:
                    max_value = current_value
                    max_action = action

            return max_action


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
        state_num = len(self.mdp.getStates())
        for i in range(self.iterations):
            turn = i % state_num
            # values_copy = self.values.copy()
            state = self.mdp.getStates()[turn]
            if not self.mdp.isTerminal(state):
                legalActions = self.mdp.getPossibleActions(state)
                max_value = -maxsize
                for action in legalActions:
                    current_value = self.computeQValueFromValues(state, action)
                    if current_value > max_value:
                        max_value = current_value
                if max_value > -maxsize:
                    self.values[state] = max_value

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
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = self.findPredecessors(state)

        priorityQueue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                bestAction = self.computeActionFromValues(state)
                theHighestQ = self.computeQValueFromValues(state, bestAction)
                diff = abs(self.values[state] - theHighestQ)
                priorityQueue.update(state, -diff)

        for _ in range(self.iterations):
            if not priorityQueue.isEmpty():
                state = priorityQueue.pop()
                
                if not self.mdp.isTerminal(state):
                    legalActions = self.mdp.getPossibleActions(state)
                    max_value = -maxsize
                    
                    for action in legalActions:
                        current_value = self.computeQValueFromValues(state, action)
                        if current_value > max_value:
                            max_value = current_value
                    
                    if max_value > -maxsize:
                        self.values[state] = max_value

                    for prevState in list(predecessors[state]):
                        if not self.mdp.isTerminal(prevState):
                            bestAction = self.computeActionFromValues(prevState)
                            theHighestQ = self.computeQValueFromValues(prevState, bestAction)
                            diff = abs(self.values[prevState] - theHighestQ)
                            if diff > self.theta:
                                priorityQueue.update(prevState, -diff)

            else:
                break


    def findPredecessors(self, currentState):

        predecessors = set()

        if not self.mdp.isTerminal(currentState):
            for state in self.mdp.getStates():
                legalActions = self.mdp.getPossibleActions(state)
                for action in legalActions:
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for nextState, prob in transitions:
                        if nextState == currentState and prob > 0:
                            predecessors.add(state)

        return predecessors
                    



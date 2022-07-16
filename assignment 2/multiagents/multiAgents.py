# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        currentFood = currentGameState.getFood()
        currentPos = currentGameState.getPacmanPosition()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        from sys import maxsize
        currentFoodList = currentFood.asList()
        newFoodList = newFood.asList()
        scaredTimesSum = sum(newScaredTimes)
        foodsDistance = 0
        minFoodsDistance = maxsize
        for food in currentFoodList:
            foodsDistance = max(foodsDistance, manhattanDistance(newPos, food))
            minFoodsDistance = min(minFoodsDistance, manhattanDistance(newPos, food))

        ghostScaredDistance = maxsize
        ghostActiveDistance = maxsize
        for ghost in newGhostStates:
            if ghost.scaredTimer > 0:
                ghostScaredDistance = min(ghostScaredDistance, util.manhattanDistance(ghost.getPosition(), newPos))
            else:
                ghostActiveDistance = min(ghostActiveDistance, util.manhattanDistance(ghost.getPosition(), newPos))
        return -1 * minFoodsDistance if ghostActiveDistance > 1 else -1 * maxsize




def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        from sys import maxsize
        AgentsNum = gameState.getNumAgents()
        def Minimax(currentGameState, depth, agentIndex):
            if agentIndex == 0:
                depth -= 1
            if depth == 0 or currentGameState.isLose() or currentGameState.isWin():
                return None, self.evaluationFunction(currentGameState)

            legalActions = currentGameState.getLegalActions(agentIndex)
            bestAction = legalActions[0]

            if agentIndex == 0: # Pac Man
                max_value = -1 * maxsize
                nextAgentIndex = agentIndex + 1
                for action in legalActions:
                    nextGameState = currentGameState.generateSuccessor(agentIndex, action)
                    value = Minimax(nextGameState, depth=depth, agentIndex=nextAgentIndex)[1]
                    if value > max_value:
                        max_value = value
                        bestAction = action
                return bestAction, max_value

            else:               # Ghosts
                min_value = maxsize
                nextAgentIndex = (agentIndex + 1) % AgentsNum
                for action in legalActions:
                    nextGameState = currentGameState.generateSuccessor(agentIndex, action)
                    value = Minimax(nextGameState, depth=depth, agentIndex=nextAgentIndex)[1]
                    if value < min_value:
                        min_value = value
                        bestAction = action
                return bestAction, min_value

        bestAction, bestValue = Minimax(gameState, self.depth+1, agentIndex=0)
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        from sys import maxsize
        AgentsNum = gameState.getNumAgents()
        def Minimax(currentGameState, depth, alpha, beta, agentIndex):
            if agentIndex == 0:
                depth -= 1
            if depth == 0 or currentGameState.isLose() or currentGameState.isWin():
                return None, self.evaluationFunction(currentGameState)

            legalActions = currentGameState.getLegalActions(agentIndex)
            bestAction = legalActions[0]

            if agentIndex == 0: # Pac Man
                max_value = -1 * maxsize
                nextAgentIndex = agentIndex + 1
                for action in legalActions:
                    nextGameState = currentGameState.generateSuccessor(agentIndex, action)
                    value = Minimax(nextGameState, depth, alpha, beta, agentIndex=nextAgentIndex)[1]
                    if value > max_value:
                        max_value = value
                        bestAction = action
                    alpha = max(alpha, value)
                    if beta < alpha:
                        break
                return bestAction, max_value

            else:               # Ghosts
                min_value = maxsize
                nextAgentIndex = (agentIndex + 1) % AgentsNum
                for action in legalActions:
                    nextGameState = currentGameState.generateSuccessor(agentIndex, action)
                    value = Minimax(nextGameState, depth, alpha, beta, agentIndex=nextAgentIndex)[1]
                    if value < min_value:
                        min_value = value
                        bestAction = action
                    beta = min(beta, value)
                    if beta < alpha:
                        break
                return bestAction, min_value

        bestAction, bestValue = Minimax(gameState, self.depth+1, -1*maxsize, maxsize, agentIndex=0)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        from sys import maxsize
        AgentsNum = gameState.getNumAgents()
        def Expectimax(currentGameState, depth, agentIndex):
            if agentIndex == 0:
                depth -= 1
            if depth == 0 or currentGameState.isLose() or currentGameState.isWin():
                return None, self.evaluationFunction(currentGameState)

            legalActions = currentGameState.getLegalActions(agentIndex)
            bestAction = legalActions[0]

            if agentIndex == 0: # Pac Man
                max_value = 0
                nextAgentIndex = agentIndex + 1
                for action in legalActions:
                    nextGameState = currentGameState.generateSuccessor(agentIndex, action)
                    value = Expectimax(nextGameState, depth=depth, agentIndex=nextAgentIndex)[1]
                    if value > max_value:
                        max_value = value
                        bestAction = action
                return bestAction, max_value

            else:               # Ghosts
                sum_value = 0
                nextAgentIndex = (agentIndex + 1) % AgentsNum
                for action in legalActions:
                    nextGameState = currentGameState.generateSuccessor(agentIndex, action)
                    value = Expectimax(nextGameState, depth=depth, agentIndex=nextAgentIndex)[1]
                    sum_value += value
                expected_value = sum_value/len(legalActions)
                return bestAction, expected_value

        bestAction, bestValue = Expectimax(gameState, self.depth+1, 0)
        return bestAction
        


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentFood = currentGameState.getFood()
    currentPos = currentGameState.getPacmanPosition()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    currentCapsules = currentGameState.getCapsules()

    "*** YOUR CODE HERE ***"
    from sys import maxsize
    currentFoodList = currentFood.asList()
    scaredTimesSum = sum(currentScaredTimes)
    foodsDistance = 0
    minFoodsDistance = maxsize
    for food in currentFoodList:
        foodsDistance = max(foodsDistance, manhattanDistance(currentPos, food))
        minFoodsDistance = min(minFoodsDistance, manhattanDistance(currentPos, food))

    minCapsuleDistance = maxsize
    for capsule in currentCapsules:
        minCapsuleDistance = min(minCapsuleDistance, manhattanDistance(currentPos, capsule))

    ghostScaredDistance = maxsize
    ghostActiveDistance = maxsize
    for ghost in currentGhostStates:
        if ghost.scaredTimer > 0:
            ghostScaredDistance = min(ghostScaredDistance, util.manhattanDistance(ghost.getPosition(), currentPos))
        else:
            ghostActiveDistance = min(ghostActiveDistance, util.manhattanDistance(ghost.getPosition(), currentPos))
    if scaredTimesSum > 0:
        return -15 * ghostScaredDistance -5 * scaredTimesSum
    elif ghostActiveDistance == 1 and minCapsuleDistance == 1:
        return -15 * minCapsuleDistance
    elif ghostActiveDistance < 4:
        return 15 * ghostActiveDistance
    elif ghostActiveDistance >= 4:
        return -15 * (1/ghostActiveDistance) * minFoodsDistance -5 * (1/ghostActiveDistance) * minCapsuleDistance + ghostActiveDistance
    else:
        return -1 * maxsize
    # return -1 * minFoodsDistance if ghostActiveDistance > 1 else -1 * maxsize


# Abbreviation
better = betterEvaluationFunction

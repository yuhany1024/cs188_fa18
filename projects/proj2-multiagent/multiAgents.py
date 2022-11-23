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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPos = [ghostState.getPosition() for ghostState in newGhostStates]
        scared = min(newScaredTimes) > 0
        "*** YOUR CODE HERE ***"
        if not scared and newPos in newGhostPos:
            return -1
        if scared and newPos in newGhostPos:
            return 100
        if newPos in currentGameState.getFood().asList():
            return 1

        food_count = 0
        closest_food = float("Inf")
        for food_pos in newFood.asList():
            d = util.manhattanDistance(newPos, food_pos)
            closest_food = min(d, closest_food)
            food_count += 1

        closest_ghost = float("Inf")
        for i, ghostPos in enumerate(newGhostPos):
            d = util.manhattanDistance(newPos, ghostPos)
            if d < closest_ghost:
                closest_ghost = d
                scared = 1 if newScaredTimes[i] > 0 else -1

        score = 1/closest_food+scared/closest_ghost
        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        nAgent = gameState.getNumAgents()
        actionList = gameState.getLegalActions(0) # consider the pacman's action list
        resAction = None
        maxValue = float("-Inf")
        for action in actionList:
            nextState = gameState.generateSuccessor(0, action)
            nextAgent = 1 if nAgent>1 else 0
            value = self.get_value(nextState, agentId=nextAgent, depth=1)
            if value > maxValue:
                maxValue = value
                resAction = action

        return resAction

    def get_value(self, gameState, agentId, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        fn = max if agentId == 0 else min
        v = float("-Inf") if agentId == 0 else float("Inf")
        nAgent = gameState.getNumAgents()
        actionList = gameState.getLegalActions(agentId)
        nextAgent = agentId + 1 if agentId + 1 < nAgent else 0
        nextDepth = depth if agentId + 1 < nAgent else depth + 1
        for action in actionList:
            nextState = gameState.generateSuccessor(agentId, action)
            if nextDepth > self.depth:
                v0 = self.evaluationFunction(nextState)
            else:
                v0 = self.get_value(nextState, nextAgent, nextDepth)

            v = fn(v, v0)
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        nAgent = gameState.getNumAgents()
        actionList = gameState.getLegalActions(0) # consider the pacman's action list
        resAction = None
        maxValue = float("-Inf")
        for action in actionList:
            nextState = gameState.generateSuccessor(0, action)
            nextAgent = 1 if nAgent>1 else 0
            value = self.get_value(nextState, agentId=nextAgent, depth=1, alpha=maxValue, beta=float("Inf"))
            if value > maxValue:
                maxValue = value
                resAction = action

        return resAction

    def get_value(self, gameState, agentId, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        fn = max if agentId == 0 else min
        v = float("-Inf") if agentId == 0 else float("Inf")
        nAgent = gameState.getNumAgents()
        actionList = gameState.getLegalActions(agentId)
        nextAgent = agentId + 1 if agentId + 1 < nAgent else 0
        nextDepth = depth if agentId + 1 < nAgent else depth + 1
        for action in actionList:
            nextState = gameState.generateSuccessor(agentId, action)
            if nextDepth > self.depth:
                v0 = self.evaluationFunction(nextState)
            else:
                v0 = self.get_value(nextState, nextAgent, nextDepth, alpha, beta)

            v = fn(v, v0)
            if agentId == 0:
                if v > beta:
                    return v
                alpha = fn(alpha, v)
            else:
                if v < alpha:
                    return v
                beta = fn(beta, v)


        return v

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
        nAgent = gameState.getNumAgents()
        actionList = gameState.getLegalActions(0)  # consider the pacman's action list
        resAction = None
        maxValue = float("-Inf")
        for action in actionList:
            nextState = gameState.generateSuccessor(0, action)
            if nAgent > 1:
                nextAgent = 1
                value = self.get_expect(nextState, agentId=nextAgent, depth=1)
            else:
                nextAgent = 0
                value = self.get_max(nextState, agentId=nextAgent, depth=1)

            if value > maxValue:
                maxValue = value
                resAction = action

        return resAction

    def get_max(self, gameState, agentId, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        v = float("-Inf")
        nAgent = gameState.getNumAgents()
        actionList = gameState.getLegalActions(agentId)
        nextAgent = agentId + 1 if agentId + 1 < nAgent else 0
        nextDepth = depth if agentId + 1 < nAgent else depth + 1
        for action in actionList:
            nextState = gameState.generateSuccessor(agentId, action)
            if nextDepth > self.depth:
                v0 = self.evaluationFunction(nextState)
            else:
                if nextAgent == 0:
                    v0 = self.get_max(nextState, nextAgent, nextDepth)
                else:
                    v0 = self.get_expect(nextState, nextAgent, nextDepth)

            v = max(v, v0)
        return v

    def get_expect(self, gameState, agentId, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        v = 0
        nAgent = gameState.getNumAgents()
        actionList = gameState.getLegalActions(agentId)
        nextAgent = agentId + 1 if agentId + 1 < nAgent else 0
        nextDepth = depth if agentId + 1 < nAgent else depth + 1
        for action in actionList:
            nextState = gameState.generateSuccessor(agentId, action)
            if nextDepth > self.depth:
                v0 = self.evaluationFunction(nextState)
            else:
                if nextAgent == 0:
                    v0 = self.get_max(nextState, nextAgent, nextDepth)
                else:
                    v0 = self.get_expect(nextState, nextAgent, nextDepth)

            v += v0

        return v/len(actionList)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    GhostPos = [ghostState.getPosition() for ghostState in GhostStates]

    if currentGameState.isLose() or pacmanPos in GhostPos:
        return float('-inf')

    closest_food = float("Inf")
    for food_pos in Food.asList():
        d = util.manhattanDistance(pacmanPos, food_pos)
        closest_food = min(d, closest_food)

    closest_ghost = float("Inf")
    for i, ghostPos in enumerate(GhostPos):
        d = util.manhattanDistance(pacmanPos, ghostPos)
        if d < closest_ghost:
            closest_ghost = d
            scared = 1 if ScaredTimes[i] > 0 else -1

    return scoreEvaluationFunction(currentGameState) + 1/closest_food + scared*1/closest_ghost


# Abbreviation
better = betterEvaluationFunction

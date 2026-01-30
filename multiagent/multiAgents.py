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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        closestFood = float('inf')
        for x in range(0, successorGameState.data.layout.width):
            for y in range(0, successorGameState.data.layout.height):
                if (newFood[x][y] == True):
                    distance = manhattanDistance((x, y), newPos)
                    if (distance < closestFood):
                       closestFood = distance
        foodScore = (1 / (closestFood * 2))
        ghostDistances = [manhattanDistance(ghostState.getPosition(), newPos) for ghostState in newGhostStates]
        minGhostDistance = min(ghostDistances)
        ghostScore = 0
        if (newScaredTimes[0] > 0):
            ghostScore = (1 / minGhostDistance)
        else:
            ghostScore = minGhostDistance
        return successorGameState.getScore() + (foodScore * ghostScore)

def scoreEvaluationFunction(currentGameState: GameState): 
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

    def getAction(self, gameState: GameState):
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
        score, move = self.minimax(gameState, self.depth, 0)
        return move

    def minimax(self, gameState: GameState, depth: int, agentIndex: int):
        """
        Helper function to recursively compute the minimax search algorithm
        """
        if (agentIndex == gameState.getNumAgents()):
            agentIndex = 0
            depth -= 1
        if ((depth == 0) or (not gameState.getLegalActions(agentIndex))):
            return self.evaluationFunction(gameState), ""
        
        if agentIndex == 0:
            pacmanLegalMoves = gameState.getLegalActions(agentIndex)
            alpha = float('-inf')
            action = ""
            for move in pacmanLegalMoves:
                state = gameState.generateSuccessor(agentIndex, move)
                evaluation = self.minimax(state, depth, agentIndex + 1)[0]
                if (evaluation > alpha):
                    alpha = evaluation
                    action = move
            return alpha, action
        else:
            ghostLegalMoves = gameState.getLegalActions(agentIndex)
            beta = float('inf')
            action = ""
            for move in ghostLegalMoves:
                state = gameState.generateSuccessor(agentIndex, move)
                evaluation = self.minimax(state, depth, agentIndex + 1)[0]
                if (evaluation < beta):
                    beta = evaluation
                    action = move
            return beta, action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        score, move = self.minimax(gameState, self.depth, 0, float('-inf'), float('inf'))
        return move
    def minimax(self, gameState: GameState, depth: int, agentIndex: int, alpha: int, beta: int):
        """
        Helper function to recursively compute the minimax search algorithm
        """
        if (agentIndex == gameState.getNumAgents()):
            agentIndex = 0
            depth -= 1
        if ((depth == 0) or (not gameState.getLegalActions(agentIndex))):
            return self.evaluationFunction(gameState), ""
        
        if agentIndex == 0:
            pacmanLegalMoves = gameState.getLegalActions(agentIndex)
            val_max = float('-inf')
            action = ""
            for move in pacmanLegalMoves:
                state = gameState.generateSuccessor(agentIndex, move)
                evaluation = self.minimax(state, depth, agentIndex + 1, alpha, beta)[0]
                if (evaluation > val_max):
                    val_max = evaluation
                    action = move
                alpha = max(alpha, val_max)
                if (alpha > beta):
                    return val_max, action
            return val_max, action
        else:
            ghostLegalMoves = gameState.getLegalActions(agentIndex)
            val_min = float('inf')
            action = ""
            for move in ghostLegalMoves:
                state = gameState.generateSuccessor(agentIndex, move)
                evaluation = self.minimax(state, depth, agentIndex + 1, alpha, beta)[0]
                if (evaluation < val_min):
                    val_min = evaluation
                    action = move
                beta = min(beta, val_min)
                if (alpha > beta):
                    return val_min, action
            return val_min, action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        score, move = self.expectimax(gameState, self.depth, 0)
        return move

    def expectimax(self, gameState: GameState, depth: int, agentIndex: int):
        """
        Helper function to recursively compute the expecimax search algorithm
        """
        if (agentIndex == gameState.getNumAgents()):
            agentIndex = 0
            depth -= 1
        if ((depth == 0) or (not gameState.getLegalActions(agentIndex))):
            return self.evaluationFunction(gameState), ""
        
        if agentIndex == 0:
            pacmanLegalMoves = gameState.getLegalActions(agentIndex)
            alpha = float('-inf')
            action = ""
            for move in pacmanLegalMoves:
                state = gameState.generateSuccessor(agentIndex, move)
                evaluation = self.expectimax(state, depth, agentIndex + 1)[0]
                if (evaluation > alpha):
                    alpha = evaluation
                    action = move
            return alpha, action
        else:
            ghostLegalMoves = gameState.getLegalActions(agentIndex)
            action = ""
            avg_score = 0
            for move in ghostLegalMoves:
                state = gameState.generateSuccessor(agentIndex, move)
                avg_score += self.expectimax(state, depth, agentIndex + 1)[0]
            avg_score /= len(ghostLegalMoves)
            return avg_score, action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    
    pacPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    food = currentGameState.getFood()
    closestFood = float('inf')
    for x in range(0, currentGameState.data.layout.width):
        for y in range(0, currentGameState.data.layout.height):
            if (food[x][y] == True):
                distance = manhattanDistance((x, y), pacPos)
                if (distance < closestFood):
                    closestFood = distance
    foodScore = (1 / (closestFood * 5))
    ghostDistances = [manhattanDistance(ghostState.getPosition(), pacPos) for ghostState in ghostStates]
    minGhostDistance = min(ghostDistances)
    ghostScore = 0
    if (scaredTimes[0] > 0):
        ghostScore = (1 / minGhostDistance)
    else:
        ghostScore = minGhostDistance

    capsules = currentGameState.getCapsules()
    capsuleScore = 0
    if (capsules):
        capsuleDistance = [manhattanDistance(pacPos, capsule) for capsule in capsules]
        capsuleScore = (1 / min(capsuleDistance))

    return currentGameState.getScore() + (foodScore * ghostScore) + (capsuleScore * 10)

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

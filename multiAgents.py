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


import random
import sys

import util
from game import Agent
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


        #INICIALIZAR LISTAS
        dF = []
        dC = []
        por_comer = []
        score = successorGameState.getScore()

        # A�ADIR COMIDAS Y C�PSULAS
        for x, fila in enumerate(newFood):
            for y, comida in enumerate(fila):
                if comida:
                    por_comer.append((x, y))
        for capsule in currentGameState.getCapsules():
                por_comer.append(capsule)

        for comida in por_comer:
            dC.append(abs(newPos[0] - comida[0]) + abs(newPos[1] - comida[1]))

        # A�ADIR FANTASMAS
        for i, fantasma in enumerate(newGhostStates):
            if newScaredTimes[i]==0:
                dF.append(abs(newPos[0] - fantasma.configuration.pos[0]) + abs(newPos[1] - fantasma.configuration.pos[1]))
            else: # En caso de que el fantasma est� huyendo se considera comida
                dC.append(abs(newPos[0] - fantasma.configuration.pos[0]) + abs(newPos[1] - fantasma.configuration.pos[1]))

        # SACAR EL M�NIMO DE CADA UNO (EL M�S CERCANO)
        if len(dF) != 0: distFan = min(dF)
        else: distFan = sys.maxsize

        if len(dC) == 0: distCom = 0
        else: distCom = min(dC)

        # CONDICIONES
        if distCom == 0:
            ema = sys.maxsize
        elif distFan == 0:
            ema = -sys.maxsize
        elif distCom == distFan:
            ema = -1 / distFan
        elif distCom < distFan:
            ema = 1 / distCom
        else:
            ema = -distCom

        return ema + score


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
        "*** YOUR CODE FROM HERE ***"
        agentIndex = self.index

        actionValue = {}

        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            succesorGameState = gameState.generateSuccessor(agentIndex, action)
            value = self.value(succesorGameState, agentIndex + 1, 0)
            actionValue[action] = value
        
        maxValue = max(actionValue.values())
        for action, value in actionValue.items():
            if value == maxValue:
                return action

    def value(self, state, agent, layer):
        # Si ha pasado por todos los agentes:
        if agent == state.getNumAgents():
            agent = 0
            layer += 1
        if self.isTerminal(state, layer): return self.evaluationFunction(state)
        if agent == 0: return self.maxValue(state, layer)
        else: return self.minValue(state, agent, layer)

    def maxValue(self, state, layer):
        v = -sys.maxsize
        legalActions = state.getLegalActions(self.index)
        for action in legalActions:
            succesorGameState = state.generateSuccessor(self.index, action)
            v = max(v, self.value(succesorGameState, 1, layer))
        return v

    def minValue(self, state, agent, layer):
        v = sys.maxsize
        legalActions = state.getLegalActions(agent)
        for action in legalActions:
            succesorGameState = state.generateSuccessor(agent, action)
            v = min(v, self.value(succesorGameState, agent + 1, layer))
        return v
    
    def isTerminal(self, state, layer):
        return state.isWin() or state.isLose() or layer == self.depth

        "*** YOUR CODE TO HERE ***"
 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE FROM HERE ***"

        agentIndex = self.index

        actionValue = {}

        alfa = -sys.maxsize
        beta = sys.maxsize

        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            succesorGameState = gameState.generateSuccessor(agentIndex, action)
            value = self.value(succesorGameState, agentIndex + 1, 0, alfa, beta)
            actionValue[action] = value
            alfa = max(alfa, value)

        maxValue = max(actionValue.values())
        for action, value in actionValue.items():
            if value == maxValue:
                return action

    def value(self, state, agent, layer, alfa, beta):
        # Si ha pasado por todos los agentes:
        if agent == state.getNumAgents():
            agent = 0
            layer += 1
        if self.isTerminal(state, layer): return self.evaluationFunction(state)
        if agent == 0: return self.maxValue(state, layer, alfa, beta)
        else: return self.minValue(state, agent, layer, alfa, beta)

    def maxValue(self, state, layer, alfa, beta):
        v = -sys.maxsize
        legalActions = state.getLegalActions(self.index)
        for action in legalActions:
            succesorGameState = state.generateSuccessor(self.index, action)
            v = max(v, self.value(succesorGameState, 1, layer, alfa, beta))
            if v > beta: return v
            alfa = max(alfa, v)
        return v

    def minValue(self, state, agent, layer, alfa, beta):
        v = sys.maxsize
        legalActions = state.getLegalActions(agent)
        for action in legalActions:
            succesorGameState = state.generateSuccessor(agent, action)
            v = min(v, self.value(succesorGameState, agent + 1, layer, alfa, beta))
            if v < alfa: return v
            beta = min(beta, v)
        return v

    def isTerminal(self, state, layer):
        return state.isWin() or state.isLose() or layer == self.depth
    
        "*** YOUR CODE TO HERE ***"


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
        "*** YOUR CODE FROM HERE ***"
        agentIndex = self.index

        actionValue = {}

        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            succesorGameState = gameState.generateSuccessor(agentIndex, action)
            value = self.value(succesorGameState, agentIndex + 1, 0)
            actionValue[action] = value
        
        maxValue = max(actionValue.values())
        for action, value in actionValue.items():
            if value == maxValue:
                return action

    def value(self, state, agent, layer):
        # Si ha pasado por todos los agentes:
        if agent == state.getNumAgents():
            agent = 0
            layer += 1
        if self.isTerminal(state, layer): return self.evaluationFunction(state)
        if agent == 0: return self.maxValue(state, layer)
        else: return self.exp(state, agent, layer)

    def maxValue(self, state, layer):
        v = -sys.maxsize
        legalActions = state.getLegalActions(self.index)
        for action in legalActions:
            succesorGameState = state.generateSuccessor(self.index, action)
            v = max(v, self.value(succesorGameState, 1, layer))
        return v

    def exp(self, state, agent, layer):
        v = 0
        legalActions = state.getLegalActions(agent)
        p = 1/len(legalActions)
        for action in legalActions:
            succesorGameState = state.generateSuccessor(agent, action)
            v += p * self.value(succesorGameState, agent + 1, layer)
        return v
    
    def isTerminal(self, state, layer):
        return state.isWin() or state.isLose() or layer == self.depth
        "*** YOUR CODE TO HERE ***"

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    ghostStates = currentGameState.getGhostStates()
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    #INICIALIZAR LISTAS
    dF = []
    dC = []
    por_comer = []
    score = currentGameState.getScore()

    # A�ADIR COMIDAS Y C�PSULAS
    for x, fila in enumerate(food):
        for y, comida in enumerate(fila):
            if comida:
                por_comer.append((x, y))
    for capsule in currentGameState.getCapsules():
            por_comer.append(capsule)

    for comida in por_comer:
        dC.append(abs(pos[0] - comida[0]) + abs(pos[1] - comida[1]))

    # A�ADIR FANTASMAS
    for i, fantasma in enumerate(ghostStates):
        if scaredTimes[i]==0:
            dF.append(abs(pos[0] - fantasma.configuration.pos[0]) + abs(pos[1] - fantasma.configuration.pos[1]))
        else: # En caso de que el fantasma est� huyendo se considera comida
            dC.append(abs(pos[0] - fantasma.configuration.pos[0]) + abs(pos[1] - fantasma.configuration.pos[1]))

    # SACAR EL M�NIMO DE CADA UNO (EL M�S CERCANO)
    if len(dF) != 0: distFan = min(dF)
    else: distFan = sys.maxsize

    if len(dC) == 0: distCom = 0
    else: distCom = min(dC)

    # CONDICIONES
    if distCom == 0:
        ema = sys.maxsize
    elif distFan == 0:
        ema = -sys.maxsize
    elif distCom == distFan:
        ema = -1 / distFan
    elif distCom < distFan:
        ema = 1 / distCom
    else:
        ema = -distCom

    return ema + score

# Abbreviation
better = betterEvaluationFunction

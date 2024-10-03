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
import random, util, itertools

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

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            closestFood = min(foodDistances)
        else:
            closestFood = 1

        ghostDis = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        ghostPen = 0
        for i, ghostDist in enumerate(ghostDis):
            if newScaredTimes[i] > 0:
                ghostPen += 200 / (ghostDist + 1)
            else:
                ghostPen -= 30 / (ghostDist + 1)

        foodReward = 8 / closestFood

        return successorGameState.getScore() + foodReward + ghostPen


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minmaxSearch(self, gameState: GameState, dep: int, cur: int):
        if gameState.isWin() or gameState.isLose() or dep == 0:
            return self.evaluationFunction(gameState)

        n = gameState.getNumAgents()  # anz. agenten
        nxt = (cur + 1) % n

        return (max if cur == 0 else min)(
            map(
                lambda act: self.minmaxSearch(
                    gameState.generateSuccessor(cur, act),
                    dep - 1 if nxt == 0 else dep,
                    nxt,
                ),
                gameState.getLegalActions(cur),
            ),
            default=float("inf") * (-1 if cur == 0 else 1),
        )

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
        "*** YOUR CODE HERE ***"

        actions = gameState.getLegalActions(0)
        if len(actions) == 0:
            return Directions.STOP

        act = Directions.STOP

        scr = float("-inf")

        for action in actions:
            succ = gameState.generateSuccessor(0, action)
            score = self.minmaxSearch(succ, self.depth, 1)

            if score > scr:
                scr = score
                act = action

        return act
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    # min max value helper
    def minmaxValue(
        self, state: GameState, dep: int, cur: int, a: int, b: int, maxf: bool
    ):
        if state.isWin() or state.isLose() or dep == 0:
            return self.evaluationFunction(state)
        res = float("-inf") if maxf else float("inf")
        n = state.getNumAgents()
        nxt = (cur + 1) % n

        for action in state.getLegalActions(cur):
            res = (max if maxf else min)(
                res,
                self.minmaxValue(
                    state.generateSuccessor(cur, action),
                    dep - 1 if nxt == 0 else dep,
                    nxt,
                    a,
                    b,
                    nxt == 0,
                ),
            )
            if maxf:
                if res > b:
                    return res
                a = max(res, a)
            else:
                if res < a:
                    return res
                b = min(res, b)
        return res

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        actions = gameState.getLegalActions()
        if len(actions) == 0:
            return Directions.STOP
        a = float("-inf")
        b = float("inf")
        act = Directions.STOP
        val = float("-inf")

        for action in actions:
            res = self.minmaxValue(
                gameState.generateSuccessor(0, action), self.depth, 1, a, b, False
            )
            if res > val:
                val = res
                act = action
            a = max(a, val)

        return act
        # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    #expectimax helper
    def expectimax(self, n: int, dep: int, state: GameState):
        if n == state.getNumAgents():
            if dep == self.depth:
                return self.evaluationFunction(state)
            else:
                return self.expectimax(0, dep + 1, state)
        else:
            actions = state.getLegalActions(n)

            if len(actions) == 0:
                return self.evaluationFunction(state)

            # computes minimax for every possible action, increment agent by 1 to make ghost
            succ = [
                self.expectimax(n + 1, dep, state.generateSuccessor(n, action))
                for action in actions
            ]

            if n == 0:
                return max(succ)
            else:
                return sum(succ) / len(succ)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # minimax each legal action at next level of agent

        return max(
            gameState.getLegalActions(0),
            key=lambda action: self.expectimax(
                1, 1, gameState.generateSuccessor(0, action)
            ),
        )
        # util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Intuitively, nearer food, farther ghosts when not scared and nearer ghosts when scared is better.
                 We use Chebyshev Distance, since it's very easy to compute.
    """
    "*** YOUR CODE HERE ***"
    
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    def cDistance(a,b):
        return max(abs(a[0]-b[0]),abs(a[1]-b[1]))

    foodDistances = [cDistance(newPos, food) for food in newFood.asList()]
    if foodDistances:
        closestFood = min(foodDistances)
    else:
        closestFood = 1

    ghostDis = [cDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    ghostPen = 0
    for i, ghostDist in enumerate(ghostDis):
        if newScaredTimes[i] > 0:
            ghostPen += 100 / (ghostDist + 1)
        else:
            ghostPen -= 10 / (ghostDist + 1)

    foodReward = 12 / closestFood

    return successorGameState.getScore() + foodReward + ghostPen


# Abbreviation
better = betterEvaluationFunction

# multiagentTestClasses.py
# ------------------------
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


import json
import random
import time
from collections import defaultdict
from pprint import PrettyPrinter

import layout
import pacman
import testClasses
from game import Agent
from ghostAgents import *
from pacman import GameState

# import grading


VERBOSE = False


class MultiagentTreeState(object):
    # A minimax tree which interfaces like gameState
    #     state.getNumAgents()
    #     state.isWin()
    #     state.isLose()
    #     state.generateSuccessor(agentIndex, action)
    #     state.getScore()
    #           used by multiAgents.scoreEvaluationFunction, which is the default
    #
    def __init__(self, problem, state):
        self.problem = problem
        self.state = state

    def generateSuccessor(self, agentIndex, action):
        if VERBOSE: print(f"generateSuccessor({self.state}, {agentIndex}, {action}) -> {self.problem.stateToSuccessorMap[self.state][action]}")
        
        successor = self.problem.stateToSuccessorMap[self.state][action]
        self.problem.generatedStates.add(successor)
        
        return MultiagentTreeState(self.problem, successor)

    def getScore(self):
        if VERBOSE: print(f"getScore({self.state}) -> {self.problem.evaluation[self.state]}")
        
        if self.state not in self.problem.evaluation:
            raise Exception('getScore() called on non-terminal state or before maximum depth achieved.')
        
        return float(self.problem.evaluation[self.state])

    def getLegalActions(self, agentIndex=0):
        if VERBOSE: print(f"getLegalActions({self.state}) -> {self.problem.stateToActions[self.state]}")
        # if len(self.problem.stateToActions[self.state]) == 0:
        #    print "WARNING: getLegalActions called on leaf state {%s}" % (self.state,)
        return list(self.problem.stateToActions[self.state])

    def isWin(self):
        if VERBOSE: print(f"isWin({self.state}) -> {self.state in self.problem.winStates}")
        return self.state in self.problem.winStates

    def isLose(self):
        if VERBOSE: print(f"isLose({self.state}) -> {self.state in self.problem.loseStates}")
        return self.state in self.problem.loseStates

    def getNumAgents(self):
        if VERBOSE: print(f"getNumAgents({self.state}) -> {self.problem.numAgents}")
        return self.problem.numAgents


class MultiagentTreeProblem(object):
    def __init__(self, numAgents, startState, winStates, loseStates, successors, evaluation):
        self.startState = MultiagentTreeState(self, startState)

        self.numAgents = numAgents
        self.winStates = winStates
        self.loseStates = loseStates
        self.evaluation = evaluation
        self.successors = successors

        self.generatedStates = None
        self.reset()

        self.stateToSuccessorMap = defaultdict(dict)
        self.stateToActions = defaultdict(list)
        for state, action, nextState in successors:
            self.stateToActions[state].append(action)
            self.stateToSuccessorMap[state][action] = nextState

    def reset(self):
        self.generatedStates = {self.startState.state}


def parseTreeProblem(testDict):
    numAgents = int(testDict["num_agents"])
    startState = testDict["start_state"]
    winStates = set(testDict["win_states"].split(" "))
    loseStates = set(testDict["lose_states"].split(" "))
    successors = []

    evaluation = {}
    for line in testDict["evaluation"].split('\n'):
        tokens = line.split()
        if len(tokens) == 2:
            state, value = tokens
            evaluation[state] = float(value)
        else:
            raise Exception(f"[parseTree] Bad evaluation line: |{line}|")

    for line in testDict["successors"].split('\n'):
        tokens = line.split()
        if len(tokens) == 3:
            state, action, nextState = tokens
            successors.append((state, action, nextState))
        else:
            raise Exception(f"[parseTree] Bad successor line: |{line}|")

    return MultiagentTreeProblem(numAgents, startState, winStates, loseStates, successors, evaluation)


def run(lay, layName, pac, ghosts, disp, nGames=1, name='games'):
    """
    Runs a few games and outputs their statistics.
    """
    starttime = time.time()
    
    print(f'*** Running {name} on {layName} {nGames} time(s).')
    
    games = pacman.runGames(lay, pac, ghosts, disp, nGames, False, catchExceptions=True, timeout=120)
    
    print(f'*** Finished running {name} on {layName} after {time.time() - starttime} seconds.')
    stats = {'time': time.time() - starttime,
             'wins': [game.state.isWin() for game in games].count(True),
             'games': games,
             'scores': [game.state.getScore() for game in games],
             'timeouts': [game.agentTimeout for game in games].count(True),
             'crashes': [game.agentCrashed for game in games].count(True)}
    print(f'*** Won {stats["wins"]} out of {len(games)} games. Average score: {sum(stats["scores"]) * 1.0 / len(games)} ***')
    
    return stats


class GradingAgent(Agent):
    def __init__(self, seed, studentAgent, optimalActions, altDepthActions, partialPlyBugActions):
        super().__init__()
        # save student agent and actions of refernce agents
        self.studentAgent = studentAgent
        self.optimalActions = optimalActions
        self.altDepthActions = altDepthActions
        self.partialPlyBugActions = partialPlyBugActions
        # create fields for storing specific wrong actions
        self.suboptimalMoves = []
        self.wrongStatesExplored = -1
        # boolean vectors represent types of implementation the student could have
        self.actionsConsistentWithOptimal = [True] * len(optimalActions[0])
        self.actionsConsistentWithAlternativeDepth = [True] * len(altDepthActions[0])
        self.actionsConsistentWithPartialPlyBug = [True] * len(partialPlyBugActions[0])
        # keep track of elapsed moves
        self.stepCount = 0
        self.seed = seed

    def registerInitialState(self, state):
        if 'registerInitialState' in dir(self.studentAgent):
            self.studentAgent.registerInitialState(state)

        random.seed(self.seed)

    def getAction(self, state):
        GameState.getAndResetExplored()

        studentAction = (self.studentAgent.getAction(state), len(GameState.getAndResetExplored()))
        optimalActions = self.optimalActions[self.stepCount]
        altDepthActions = self.altDepthActions[self.stepCount]
        partialPlyBugActions = self.partialPlyBugActions[self.stepCount]

        studentOptimalAction = False
        curRightStatesExplored = False

        for i in range(len(optimalActions)):
            if studentAction[0] in optimalActions[i][0]:
                studentOptimalAction = True
            else:
                self.actionsConsistentWithOptimal[i] = False
            if studentAction[1] == int(optimalActions[i][1]):
                curRightStatesExplored = True

        if not curRightStatesExplored and self.wrongStatesExplored < 0:
            self.wrongStatesExplored = 1

        for i in range(len(altDepthActions)):
            if studentAction[0] not in altDepthActions[i]:
                self.actionsConsistentWithAlternativeDepth[i] = False

        for i in range(len(partialPlyBugActions)):
            if studentAction[0] not in partialPlyBugActions[i]:
                self.actionsConsistentWithPartialPlyBug[i] = False

        if not studentOptimalAction:
            self.suboptimalMoves.append((state, studentAction[0], optimalActions[0][0][0]))

        self.stepCount += 1
        random.seed(self.seed + self.stepCount)

        return optimalActions[0][0][0]

    def getSuboptimalMoves(self):
        return self.suboptimalMoves

    def getWrongStatesExplored(self):
        return self.wrongStatesExplored

    def checkFailure(self):
        """
        Return +n if have n suboptimal moves.
        Return -1 if have only off by one depth moves.
        Return 0 otherwise.
        """
        if self.wrongStatesExplored > 0:
            return -3
        if self.actionsConsistentWithOptimal.count(True) > 0:
            return 0
        elif self.actionsConsistentWithPartialPlyBug.count(True) > 0:
            return -2
        elif self.actionsConsistentWithAlternativeDepth.count(True) > 0:
            return -1
        else:
            return len(self.suboptimalMoves)


class PolyAgent(Agent):
    def __init__(self, seed, multiAgents, ourPacOptions, depth):
        super().__init__()

        # prepare our pacman agents
        solutionAgents, alternativeDepthAgents, partialPlyBugAgents = self.construct_our_pacs(multiAgents, ourPacOptions)

        for p in solutionAgents:
            p.depth = depth
        for p in partialPlyBugAgents:
            p.depth = depth
        for p in alternativeDepthAgents[:2]:
            p.depth = max(1, depth - 1)
        for p in alternativeDepthAgents[2:]:
            p.depth = depth + 1

        self.solutionAgents = solutionAgents
        self.alternativeDepthAgents = alternativeDepthAgents
        self.partialPlyBugAgents = partialPlyBugAgents
        # prepare fields for storing the results
        self.optimalActionLists = []
        self.alternativeDepthLists = []
        self.partialPlyBugLists = []
        self.seed = seed
        self.stepCount = 0

    @staticmethod
    def select(original_list, indices):
        """
        Return a sublist of elements given by indices in list.
        """
        return [original_list[i] for i in indices]

    def construct_our_pacs(self, multiAgents, keyword_dict):
        pacs_without_stop = [multiAgents.StaffMultiAgentSearchAgent(**keyword_dict) for _ in range(3)]

        keyword_dict['keepStop'] = 'True'
        pacs_with_stop = [multiAgents.StaffMultiAgentSearchAgent(**keyword_dict) for _ in range(3)]

        keyword_dict['usePartialPlyBug'] = 'True'
        partial_ply_bug_pacs = [multiAgents.StaffMultiAgentSearchAgent(**keyword_dict)]

        keyword_dict['keepStop'] = 'False'
        partial_ply_bug_pacs += [multiAgents.StaffMultiAgentSearchAgent(**keyword_dict)]

        for pac in pacs_with_stop + pacs_without_stop + partial_ply_bug_pacs:
            pac.verbose = False

        ourpac = [pacs_with_stop[0], pacs_without_stop[0]]
        alternative_depth_pacs = self.select(pacs_with_stop + pacs_without_stop, [1, 4, 2, 5])

        return ourpac, alternative_depth_pacs, partial_ply_bug_pacs

    def registerInitialState(self, state):
        for agent in self.solutionAgents + self.alternativeDepthAgents:
            if 'registerInitialState' in dir(agent):
                agent.registerInitialState(state)
        random.seed(self.seed)

    def getAction(self, state):
        # survey agents
        GameState.getAndResetExplored()
        optimalActionLists = []

        for agent in self.solutionAgents:
            optimalActionLists.append((agent.getBestPacmanActions(state)[0], len(GameState.getAndResetExplored())))

        alternativeDepthLists = [agent.getBestPacmanActions(state)[0] for agent in self.alternativeDepthAgents]
        partialPlyBugLists = [agent.getBestPacmanActions(state)[0] for agent in self.partialPlyBugAgents]

        # record responses
        self.optimalActionLists.append(optimalActionLists)
        self.alternativeDepthLists.append(alternativeDepthLists)
        self.partialPlyBugLists.append(partialPlyBugLists)
        self.stepCount += 1

        random.seed(self.seed + self.stepCount)
        return optimalActionLists[0][0][0]

    def getTraces(self):
        # return traces from individual agents
        return self.optimalActionLists, self.alternativeDepthLists, self.partialPlyBugLists


class PacmanGameTreeTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(PacmanGameTreeTest, self).__init__(question, testDict)
        self.seed = int(self.testDict['seed'])
        self.alg = self.testDict['alg']
        self.layout_text = self.testDict['layout']
        self.layout_name = self.testDict['layoutName']
        self.depth = int(self.testDict['depth'])
        self.max_points = int(self.testDict['max_points'])

    def execute(self, grades, moduleDict, solutionDict):
        # load student code and staff code solutions
        multiAgents = moduleDict['multiAgents']
        studentAgent = getattr(multiAgents, self.alg)(depth=self.depth)
        allActions = [json.loads(x) for x in solutionDict['optimalActions'].split('\n')]
        altDepthActions = [json.loads(x) for x in solutionDict['altDepthActions'].split('\n')]
        partialPlyBugActions = [json.loads(x) for x in solutionDict['partialPlyBugActions'].split('\n')]

        # set up game state and play a game
        random.seed(self.seed)
        lay = layout.Layout([line.strip() for line in self.layout_text.split('\n')])
        pac = GradingAgent(self.seed, studentAgent, allActions, altDepthActions, partialPlyBugActions)

        # check return codes and assign grades
        disp = self.question.getDisplay()
        stats = run(lay, self.layout_name, pac, [DirectionalGhost(i + 1) for i in range(2)], disp, name=self.alg)

        if stats['timeouts'] > 0:
            self.addMessage('Agent timed out on smallClassic.  No credit')
            return self.testFail(grades)

        if stats['crashes'] > 0:
            self.addMessage('Agent crashed on smallClassic.  No credit')
            return self.testFail(grades)

        code = pac.checkFailure()
        if code == 0:
            return self.testPass(grades)
        elif code == -3:
            if pac.getWrongStatesExplored() >= 0:
                self.addMessage('Bug: Wrong number of states expanded.')
                return self.testFail(grades)
            else:
                return self.testPass(grades)
        elif code == -2:
            self.addMessage('Bug: Partial Ply Bug')
            return self.testFail(grades)
        elif code == -1:
            self.addMessage('Bug: Search depth off by 1')
            return self.testFail(grades)
        elif code > 0:
            moves = pac.getSuboptimalMoves()
            state, studentMove, optMove = random.choice(moves)
            self.addMessage('Bug: Suboptimal moves')
            self.addMessage(f'State:{state}\nStudent Move:{studentMove}\nOptimal Move:{optMove}')
            return self.testFail(grades)

    @staticmethod
    def writeList(handle, name, item_list):
        handle.write(f'{name}: """\n')
        for item in item_list:
            handle.write(f'{json.dumps(item)}\n')
        handle.write('"""\n')

    def writeSolution(self, moduleDict, filePath):
        # load module, set seed, create ghosts and macman, run game
        multiAgents = moduleDict['multiAgents']
        random.seed(self.seed)
        lay = layout.Layout([line.strip() for line in self.layout_text.split('\n')])
        if self.alg == 'ExpectimaxAgent':
            ourPacOptions = {'expectimax': 'True'}
        elif self.alg == 'AlphaBetaAgent':
            ourPacOptions = {'alphabeta': 'True'}
        else:
            ourPacOptions = {}
        pac = PolyAgent(self.seed, multiAgents, ourPacOptions, self.depth)
        disp = self.question.getDisplay()
        run(lay, self.layout_name, pac, [DirectionalGhost(i + 1) for i in range(2)], disp, name=self.alg)
        (optimalActions, altDepthActions, partialPlyBugActions) = pac.getTraces()
        # recover traces and record to file
        handle = open(filePath, 'w')
        self.writeList(handle, 'optimalActions', optimalActions)
        self.writeList(handle, 'altDepthActions', altDepthActions)
        self.writeList(handle, 'partialPlyBugActions', partialPlyBugActions)
        handle.close()


class GraphGameTreeTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GraphGameTreeTest, self).__init__(question, testDict)
        self.problem = parseTreeProblem(testDict)
        self.alg = self.testDict['alg']
        self.diagram = self.testDict['diagram'].split('\n')
        self.depth = int(self.testDict['depth'])

    def solveProblem(self, multiAgents):
        self.problem.reset()
        studentAgent = getattr(multiAgents, self.alg)(depth=self.depth)
        action = studentAgent.getAction(self.problem.startState)
        generated = self.problem.generatedStates
        return action, " ".join([str(s) for s in sorted(generated)])

    def addDiagram(self):
        self.addMessage('Tree:')
        for line in self.diagram:
            self.addMessage(line)

    def execute(self, grades, moduleDict, solutionDict):
        multiAgents = moduleDict['multiAgents']
        goldAction = solutionDict['action']
        goldGenerated = solutionDict['generated']
        action, generated = self.solveProblem(multiAgents)

        fail = False
        if action != goldAction:
            self.addMessage(f'Incorrect move for depth={self.depth}')
            self.addMessage(f'    Student move: {action}\n    Optimal move: {goldAction}')
            fail = True

        if generated != goldGenerated:
            self.addMessage(f'Incorrect generated nodes for depth={self.depth}')
            self.addMessage(f'    Student generated nodes: {generated}\n    Correct generated nodes: {goldGenerated}')
            fail = True

        if fail:
            self.addDiagram()
            return self.testFail(grades)
        else:
            return self.testPass(grades)

    def writeSolution(self, moduleDict, filePath):
        multiAgents = moduleDict['multiAgents']
        action, generated = self.solveProblem(multiAgents)
        with open(filePath, 'w') as handle:
            handle.write(f'# This is the solution file for {self.path}.\n')
            handle.write(f'action: "{action}"\n')
            handle.write(f'generated: "{generated}"\n')
        return True


class EvalAgentTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(EvalAgentTest, self).__init__(question, testDict)
        self.layoutName = testDict['layoutName']
        self.agentName = testDict['agentName']
        self.ghosts = eval(testDict['ghosts'])
        self.maxTime = int(testDict['maxTime'])
        self.seed = int(testDict['randomSeed'])
        self.numGames = int(testDict['numGames'])

        self.scoreMinimum = int(testDict['scoreMinimum']) if 'scoreMinimum' in testDict else None
        self.nonTimeoutMinimum = int(testDict['nonTimeoutMinimum']) if 'nonTimeoutMinimum' in testDict else None
        self.winsMinimum = int(testDict['winsMinimum']) if 'winsMinimum' in testDict else None

        self.scoreThresholds = [int(s) for s in testDict.get('scoreThresholds', '').split()]
        self.nonTimeoutThresholds = [int(s) for s in testDict.get('nonTimeoutThresholds', '').split()]
        self.winsThresholds = [int(s) for s in testDict.get('winsThresholds', '').split()]

        self.maxPoints = sum([len(t) for t in [
            self.scoreThresholds, self.nonTimeoutThresholds, self.winsThresholds]])
        self.agentArgs = testDict.get('agentArgs', '')

    def execute(self, grades, moduleDict, solutionDict):
        startTime = time.time()

        agentType = getattr(moduleDict['multiAgents'], self.agentName)
        agentOpts = pacman.parseAgentArgs(self.agentArgs) if self.agentArgs != '' else {}
        agent = agentType(**agentOpts)

        lay = layout.getLayout(self.layoutName, 3)

        disp = self.question.getDisplay()

        random.seed(self.seed)
        games = pacman.runGames(lay, agent, self.ghosts, disp, self.numGames,
                                False, catchExceptions=True, timeout=self.maxTime)
        totalTime = time.time() - startTime

        stats = {'time': totalTime,
                 'wins': [game.state.isWin() for game in games].count(True),
                 'games': games,
                 'scores': [game.state.getScore() for game in games],
                 'timeouts': [game.agentTimeout for game in games].count(True),
                 'crashes': [game.agentCrashed for game in games].count(True)}

        averageScore = sum(stats['scores']) / float(len(stats['scores']))
        nonTimeouts = self.numGames - stats['timeouts']
        wins = stats['wins']

        def gradeThreshold(value, minimum, thresholds, name):
            points = 0
            passed = (minimum is None) or (value >= minimum)
            if passed:
                for t in thresholds:
                    if value >= t:
                        points += 1
            return passed, points, value, minimum, thresholds, name

        results = [gradeThreshold(averageScore, self.scoreMinimum, self.scoreThresholds, "average score"),
                   gradeThreshold(nonTimeouts, self.nonTimeoutMinimum, self.nonTimeoutThresholds, "games not timed out"),
                   gradeThreshold(wins, self.winsMinimum, self.winsThresholds, "wins")]

        totalPoints = 0
        for passed, points, value, minimum, thresholds, name in results:
            if minimum is None and len(thresholds) == 0:
                continue

            # print passed, points, value, minimum, thresholds, name
            totalPoints += points
            if not passed:
                assert points == 0
                self.addMessage(f"{value} {name} (fail: below minimum value {minimum})")
            else:
                self.addMessage(f"{value} {name} ({points} of {len(thresholds)} points)")

            if minimum is not None:
                self.addMessage("    Grading scheme:")
                self.addMessage(f"     < {minimum}:  fail")
                if len(thresholds) == 0 or minimum != thresholds[0]:
                    self.addMessage(f"    >= {minimum}:  0 points")
                for idx, threshold in enumerate(thresholds):
                    self.addMessage(f"    >= {threshold}:  {idx + 1} points")
            elif len(thresholds) > 0:
                self.addMessage("    Grading scheme:")
                self.addMessage(f"     < {thresholds[0]}:  0 points")
                for idx, threshold in enumerate(thresholds):
                    self.addMessage(f"    >= {threshold}:  {idx + 1} points")

        if any([not passed for passed, _, _, _, _, _ in results]):
            totalPoints = 0

        return self.testPartial(grades, totalPoints, self.maxPoints)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write(f'# This is the solution file for {self.path}.\n')
        handle.write('# File intentionally blank.\n')
        handle.close()
        return True

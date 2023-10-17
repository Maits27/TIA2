# testParser.py
# -------------
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


import re
import sys


class TestParser(object):

    def __init__(self, path):
        # save the path to the test file
        self.path = path

    @staticmethod
    def removeComments(rawlines):
        # remove any portion of a line following a '#' symbol
        fixed_lines = []
        for line in rawlines:
            idx = line.find('#')
            if idx == -1:
                fixed_lines.append(line)
            else:
                fixed_lines.append(line[0:idx])
        return '\n'.join(fixed_lines)

    def parse(self):
        # read in the test case and remove comments
        test = {}
        with open(self.path) as handle:
            raw_lines = handle.read().split('\n')

        test_text = self.removeComments(raw_lines)
        test['__raw_lines__'] = raw_lines
        test['path'] = self.path
        test['__emit__'] = []
        lines = test_text.split('\n')
        i = 0
        # read a property in each loop cycle
        while i < len(lines):
            # skip blank lines
            if re.match(r'\A\s*\Z', lines[i]):
                test['__emit__'].append(("raw", raw_lines[i]))
                i += 1
                continue
            m = re.match(r'\A([^"]*?):\s*"([^"]*)"\s*\Z', lines[i])
            if m:
                test[m.group(1)] = m.group(2)
                test['__emit__'].append(("oneline", m.group(1)))
                i += 1
                continue
            m = re.match(r'\A([^"]*?):\s*"""\s*\Z', lines[i])
            if m:
                msg = []
                i += 1
                while not re.match(r'\A\s*"""\s*\Z', lines[i]):
                    msg.append(raw_lines[i])
                    i += 1
                test[m.group(1)] = '\n'.join(msg)
                test['__emit__'].append(("multiline", m.group(1)))
                i += 1
                continue
            print(f'error parsing test file: {self.path}')
            sys.exit(1)
        return test


def emitTestDict(testDict, handle):
    for kind, data in testDict['__emit__']:
        if kind == "raw":
            handle.write(data + "\n")
        elif kind == "oneline":
            handle.write(f'{data}: "{testDict[data]}"\n')
        elif kind == "multiline":
            handle.write(f'{data}: """\n{testDict[data]}\n"""\n')
        else:
            raise Exception("Bad __emit__")

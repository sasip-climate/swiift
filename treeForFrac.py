import numpy as np
import matplotlib.pyplot as plt


class NodeForFloes:
    """ class for the fracture history
    Attributes:
        floe (class Floe): represented floe
        time (float): time of creation of the floe (ie of the fracture event)
        gen (int): generation of the floe (the right most floe is always of generation 0)
        children (list of Floe): if the floe has broken, the list contains its children
                                 (2 for a single fracture for instance)
            note: the children are in order from left to right
    """

    def __init__(self, floe, time, gen=0):
        self.floe = floe
        self.children = []
        self.time = time
        self.gen = gen

    def __str__(self):
        return(f'{self.floe.__repr__()} created at time {self.time} (gen {self.gen}),'
               f' with {len(self.children)} children')

    def findDirection(self, searchedFloe):
        """ Find the branch in which to look for the searched floe
        """
        x0, L = searchedFloe.x0, searchedFloe.L
        children = self.children
        nChildren = len(children)

        for k in range(nChildren):
            if children[k].floe.x0 <= x0 and x0 + L <= children[k].floe.x0 + children[k].floe.L:
                return k
        raise ValueError("Direction not found")

    def genChild(self, newFloe):
        """ Computes the generation of the newFloe resulting from the fracture of self.floe
            Note: The right most floe is always of generation 0 (it is considered as the icepack)
        """
        genOld = self.gen
        if genOld > 0:
            return genOld + 1
        elif np.abs(newFloe.x0 + newFloe.L - self.floe.x0 - self.floe.L) < 1e-6:
            return 0  # The new floe is the right most one
        else:
            return 1  # The floe is of a detached part of the ice sheet

    def addChildren(self, brokenFloe, resultingFloes, time):
        """ Search for the position in the tree to add the new resulting floes and add them
        """
        currentNode = self
        while not (abs(currentNode.floe.x0 - brokenFloe.x0) < 1e-6 and abs(currentNode.floe.L - brokenFloe.L) < 1e-6):
            direction = currentNode.findDirection(brokenFloe)
            currentNode = currentNode.children[direction]

        # At this point, the current node corresponds to the broken floe
        # Make sure children are in the left to right order
        resultingFloes.sort(key=(lambda floe: floe.x0))
        currentNode.children = [NodeForFloes(newfloe, time, gen=currentNode.genChild(newfloe))
                                for newfloe in resultingFloes]

    # Functions to get the generations of the resulting floes

    def allFloesByGeneration(self):
        """ Returns the list of all floes by generation
        """
        generations = [[]]

        def deepFirstSearch(node):
            if len(generations) <= node.gen:
                assert len(generations) == node.gen
                generations.append([])
            generations[node.gen].append(node.floe)

            if node.children != []:
                for child in node.children:
                    deepFirstSearch(child)

        deepFirstSearch(self)
        return generations

    def existingFloesByGeneration(self):
        """ Returns the list the list of existing floes by generation
        """
        generations = [[]]

        def deepFirstSearch(node):
            if len(generations) <= node.gen:
                assert len(generations) == node.gen
                generations.append([])

            if node.children != []:
                for child in node.children:
                    deepFirstSearch(child)
            else:
                generations[node.gen].append(node.floe)

        deepFirstSearch(self)
        return generations

    def maxGen(self):
        """ Computes the maximal generation of the tree
        """
        def deepMax(node, maxOld):
            if node.children == []:
                return max(maxOld, node.gen)
            else:
                maxChildren = max([deepMax(child, maxOld) for child in node.children])
                return max(maxOld, maxChildren)

        return deepMax(self, 0)

    def plotGeneration(self, onlyExisting=False, filename=''):
        """ Plots the graph of generations of fracture (ie graphical fracture history)
        """
        plt.figure()
        delta = 2
        heigth = 1
        maxgen = self.maxGen()

        def plotSides(floe, zPos, color='k'):
            # Plot floes as a rectangle
            x0 = floe.x0
            L = floe.L
            plt.plot([x0, x0], [zPos, zPos - heigth], c=color)
            plt.plot([x0 + L, x0 + L], [zPos, zPos - heigth], c=color)

        def plotFrames():
            for k in range(0, maxgen + 1):
                x0 = self.floe.x0
                L = self.floe.L
                zPos = - k * delta

                plt.plot([x0, x0], [zPos, zPos - heigth], 'k')
                plt.plot([x0 + L, x0 + L], [zPos, zPos - heigth], 'k')
                plt.plot([x0, x0 + L], [zPos - heigth, zPos - heigth], 'k')
                plt.plot([x0, x0 + L], [zPos, zPos], 'k')

        def deepFirstSearch(node):
            maxRange = node.gen if onlyExisting else maxgen
            minRange = max(node.gen, 1)
            for k in range(minRange, maxRange + 1):
                zPos = - k * delta
                plotSides(node.floe, zPos)

            # Continue search
            for child in node.children:
                deepFirstSearch(child)

        plotFrames()
        deepFirstSearch(self)

        if len(filename) > 0:
            plt.savefig(filename, dpi=150)
            plt.close()
        else:
            plt.show()

    def modifyLengthDomain(self, DeltaL):
        """ Adapt the tree if the length of the domain is modified
        """
        self.floe.L += DeltaL

        if self.children != []:
            self.children[-1].modifyLengthDomain(DeltaL)

    def asArray(self):
        # Returns an array, each row corresponding to a floe which has existed
        # Each row contains the following informations (in columns)
        # L, x0, generation, time of creation, existing boolean, index of the parent in this array
        existing = (self.children == [])
        listOfFloes = [[self.floe.L, self.floe.x0, 0, 0., existing, -1]]

        # Parcours en profondeur
        def deepFirstSearch(node, indexParent):
            for child in node.children:
                existing = (child.children == [])
                listOfFloes.append([child.floe.L, child.floe.x0, child.gen, child.time, existing, indexParent])
                deepFirstSearch(child, len(listOfFloes) - 1)

        deepFirstSearch(self, 0)
        return np.array(listOfFloes)

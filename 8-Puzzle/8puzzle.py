# coding: utf-8
# This program solves the 8-Puzzle game following the A * algorithm and the manhattam distance heuristic.
#Programmer: Matheus E. Santana
#03/March/2018


steps = 0 # Will save the number of steps to soluction
class No(object):
	"""docstring for No"""
	def __init__(self):
		self.predecessor = None
		self.state = []	
		self.distanceOfManhattam = 0

class Number():
	def __init__(self):
		self.x = 0			#refering to line
		self.y = 0			#refering to column 
		self.value = 0		#number.

def printMatriz(matriz):
	for i in range(3):
		for j in range(3):
			print("[", matriz[i][j],"]", end="")
		print("")	
	print("\n--------------------------")	

def findZero(matriz):
	positionZero = [0,0]
	for i in range(3):
		for j in range(3):
			if matriz[i][j] == 0:
				positionZero[0] = i
				positionZero[1] = j
				return positionZero


def right(matriz):
	positionZero = findZero(matriz)
	matriz[positionZero[0]][positionZero[1]] = matriz[positionZero[0]][positionZero[1]+1] 
	matriz[positionZero[0]][positionZero[1]+1] = 0
	return matriz	
def left(matriz):
	positionZero = findZero(matriz)
	matriz[positionZero[0]][positionZero[1]] = matriz[positionZero[0]][positionZero[1]-1] 
	matriz[positionZero[0]][positionZero[1]-1] = 0
	return matriz

def up(matriz):
	positionZero = findZero(matriz)
	matriz[positionZero[0]][positionZero[1]] = matriz[positionZero[0]-1][positionZero[1]] 
	matriz[positionZero[0]-1][positionZero[1]] = 0
	return matriz

def down(matriz):
	positionZero = findZero(matriz)
	matriz[positionZero[0]][positionZero[1]] = matriz[positionZero[0]+1][positionZero[1]] 
	matriz[positionZero[0]+1][positionZero[1]] = 0
	return matriz

def findSoluction(initialState, soluction, arrayNumbers):
	arrayStates = []						#Will save the open states
	arrayStates.append(initialState)		#Add first state.
	checkeds = []							#will save the checkeds states
	counter = 0								#Count the number of attempts 
	
	while len(arrayStates) != 0:
		counter+=1
		
		#Sorting elements to choose the minumum value state by distance of manhattam.
		arrayStates.sort(key=lambda a: a.distanceOfManhattam)	
		
		#Choose the minumum state.
		node = arrayStates.pop(0)
		
		#adding state to the checks 
		checkeds.append(node.state)
		
		#if expected.
		if node.state == soluction:
			print("Soluction Found!")
			printSoluction(node)
			print("Attempts:", counter)
			print("Steps:", steps)
			break
		
		localizationOfZero = findZero(node.state)

		#Generating states and adding to the open states if it has not been opened yet.
		if localizationOfZero[0] !=0:
			childNode = No() 
			childNode.state = copyState(node.state)
			childNode.state = up(childNode.state)
			if not childNode.state in checkeds:
				childNode.distanceOfManhattam = distanceOfManhattam(arrayNumbers, childNode.state)
				childNode.predecessor=node
				arrayStates.append(childNode)		
		
		
		if localizationOfZero[0] !=2:
			childNode = No() 
			childNode.state = copyState(node.state)
			childNode.state = down(childNode.state)
			
			if not childNode.state in checkeds:
				childNode.distanceOfManhattam = distanceOfManhattam(arrayNumbers, childNode.state)
				childNode.predecessor=node
				arrayStates.append(childNode)		
		
			childNode.predecessor=node
			arrayStates.append(childNode)		

		if localizationOfZero[1] !=0:
			childNode = No() 
			childNode.state =copyState(node.state)
			childNode.state = left(childNode.state)
			if not childNode.state in checkeds:
				childNode.distanceOfManhattam = distanceOfManhattam(arrayNumbers, childNode.state)
				childNode.predecessor=node
				arrayStates.append(childNode)		

		if localizationOfZero[1] !=2:
			childNode = No() 
			childNode.state = copyState(node.state)
			childNode.state = right(childNode.state)
			if not childNode.state in checkeds:
				childNode.predecessor=node
				childNode.distanceOfManhattam = distanceOfManhattam(arrayNumbers, childNode.state)
				arrayStates.append(childNode)		
		
#Print all step to the soluction 
def printSoluction(no):
	global steps
	steps = steps + 1
	if no.predecessor != None:
		printSoluction(no.predecessor)
	printMatriz(no.state)	
		

#Just copy state.
def copyState(matriz):
	temp = []
	for x in matriz:
		temp.append(x[:])
	return temp		

#Make the coordinates refering to the soluction case.
def makeCoordinates(initialState, arrayNumbers):
	for i in range(3):
		for j in range(3):
			number = Number()
			number.x = i
			number.y = j
			number.number = initialState[i][j]
			arrayNumbers[number.number] = number

#Calculate the distance of manhattam refering to the soluction case.
def distanceOfManhattam(arrayNumbers, state):
	total = 0
	for i in range(3):
		for j in range(3):
			number = state[i][j]
			total += abs(i - arrayNumbers[number].x) + abs(j - arrayNumbers[number].y)
	return total		  



soluction = [[1,2,3], [4,5,6],[7,8,0]]

initialNode = No()
initialNode.state = [[4,1,3], [2,6,8],[7,5,0]]

arrayNumbers = [None] *9

#Building array of positions. 
makeCoordinates(soluction, arrayNumbers)

#Building soluction.
findSoluction(initialNode ,soluction, arrayNumbers)

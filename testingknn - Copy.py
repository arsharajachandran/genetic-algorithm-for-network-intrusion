detectors=[]
import csv
with open('F:/GECBH/internet explorer/artificial intelligence/fwdlinkstoieeepapers/2ndcodeunder constrn/csvzzz/detectors.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
            print ("",row)
            detectors.append(row)

        
#print("\n",detectors)

import math
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(float(instance1[x]) - float(instance2[x]))), 2)
	return math.sqrt(distance)

import operator 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	print("len(testInstance)",len(testInstance))
	length = len(testInstance)   ############euclidean distance changed length not included attack or normal
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	print("\n test neighbors",neighbors) 
	return neighbors

##########################
# -----------------------------------------------------------euclidean  and for testing only 
def euclideanDistance1(instance1,instance2):    
        
        ret = []
        distance=0
        len(instance2)
        for i in range(len(instance2)):
           
           distance= distance+pow(float(float(instance1[i])-float(instance2[i])),2) 
                #ret.append(math.sqrt(distance))
        return math.sqrt(distance)



def testing(knnDistance):
   
   if knnDistance < .90 :
      result='attack'
      print("attack")
   else:
      result='normal'     
      print("normal")
   return result


###-----------------------------------------------

result=[]
testInstance=[]

with open('F:/GECBH/internet explorer/artificial intelligence/fwdlinkstoieeepapers/2ndcodeunder constrn/testinst.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
            print ("",row)
            testInstance.append(row)
#print(len(testInstance))
for i in range(len(testInstance)):
        #print(type(testInstance[i]))
        neighbors = getNeighbors(detectors, testInstance[i], 1)
        print("testInstance[i]",*testInstance[i], sep=',')
        knnDistance=euclideanDistance1(neighbors[0],testInstance[i])
        print("\n knnDistance",knnDistance)
        result.append(testing(knnDistance))


print("\n",*result,sep='\n')

























        
        

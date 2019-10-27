import numpy as np
import heapq 


def heuristicDistance(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal):
    #Deux politiques différentes sont possibles, l'une ou l'on considère les mouvements diagonaux, l'autre non
    neighbours = [(0,1),(0,-1),(1,0),(-1,0)]
    neighbours = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristicDistance(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbours:
            neighbour = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristicDistance(current, neighbour)
            if 0 <= neighbour[0] < len(array):
                if 0 <= neighbour[1] < len(array[0]):                
                    if array[neighbour[0]][neighbour[1]] == 1:
                        continue
                else:
                    #On a atteint la limite en y
                    continue
            else:
                #On a atteint la limite en x
                continue
                
            if neighbour in close_set and tentative_g_score >= gscore.get(neighbour, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbour, 0) or neighbour not in [i[1]for i in oheap]:
                came_from[neighbour] = current
                gscore[neighbour] = tentative_g_score
                fscore[neighbour] = tentative_g_score + heuristicDistance(neighbour, goal)
                heapq.heappush(oheap, (fscore[neighbour], neighbour))
                
    return False

#Exemple d'utilisation
    
#Dessiner un labyrinthe sur paint (bordures en noir, fond en blanc) et l'algorithme trouvera le meilleur chemin entre start et end, s'il existe
    
from matplotlib import pyplot as plt
img=plt.imread('maze.jpg')
plt.imshow(img)


def findStartEnd(img): 
    indiceMaxRed=(0,0)
    indiceMaxGreen=(0,0)
    for i in range (0,len(img)):
        for j in range (0,len(img[0])): 
            if [img[i,j][0],img[i,j][1],img[i,j][2]]!=[0,0,0]:
                if img[i,j,0]/(sum(img[i,j])+0.00001)>img[indiceMaxRed[0],indiceMaxRed[1]][0]/sum(img[indiceMaxRed[0],indiceMaxRed[1]]+0.00001):
                    indiceMaxRed=(i,j)
                if img[i,j,1]/(sum(img[i,j])+0.0001)>img[indiceMaxGreen[0],indiceMaxGreen[1]][1]/sum(img[indiceMaxGreen[0],indiceMaxGreen[1]]+0.00001):
                    indiceMaxGreen=(i,j)
    return(indiceMaxGreen,indiceMaxRed) 

def removeRedGreenPatches(img, start, end): 
    newimage=[]
    greenpixel=img[start[0],start[1]]
    redpixel=img[end[0],end[1]]
    for i in range(0,len(img)):
        ligne=[]
        for j in range(0,len(img[0])):
            if [img[i,j][0],img[i,j][1],img[i,j][2]]==[greenpixel[0],greenpixel[1],greenpixel[2]]:
                ligne.append([255,255,255])
            elif [img[i,j][0],img[i,j][1],img[i,j][2]]==[redpixel[0],redpixel[1],redpixel[2]]:
                ligne.append([255,255,255])
            else: 
                ligne.append([img[i,j,0],img[i,j,1],img[i,j,2]])
        newimage.append(ligne)
    return(newimage)            
    
    
start,end = findStartEnd(img)  
img=np.uint8(removeRedGreenPatches(img,start,end))

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2990 * r + 0.5870 * g + 0.1140 * b

    return gray

gray=rgb2gray(img)

def rescale(gray):
    rescaled=gray[:]/255
    return(rescaled)

gray=rescale(gray)

def invert(gray):
    inverted=[]
    for i in range (0,len(gray)):
        row=[]
        for j in range(0,len(gray[0])):
            if 1-gray[i,j]>0.7:
                value=1
            else: 
                value=0
            row.append(value)
        inverted.append(row)
    return(inverted)
#plt.imshow(numpy.array(invert(gray)))
maze=numpy.array(invert(gray))
listvisitedpoint=astar(maze, start, (end[0],end[1]))
print(listvisitedpoint)
if listvisitedpoint != False: 
    for visitedpoint in listvisitedpoint: 
        img=numpy.array(img)
        img[visitedpoint[0]][visitedpoint[1]]=[0,0,255]
        img[visitedpoint[0]+1][visitedpoint[1]]=[0,0,255]
        img[visitedpoint[0]-1][visitedpoint[1]]=[0,0,255]
        img[visitedpoint[0]][visitedpoint[1]+1]=[0,0,255]
        img[visitedpoint[0]][visitedpoint[1]-1]=[0,0,255]
    plt.imshow(img)
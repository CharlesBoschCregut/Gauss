import numpy as np
import math
from time import perf_counter

print("Pour une matrice carrée de taille nxn. Entrez n : ")
# size = int(input())
size = 3
np.random.seed()
a = np.random.choice(size*size, size*size, replace = False).reshape((size, size))
id = np.identity(size)
a = a.astype(float)
print("A origine: ")
print(a)
print()
print(" [A | Id]")
aid = np.concatenate((a, id), axis=1)
print(aid)

n = int(math.sqrt(a.size))
m = n * 2

start = perf_counter()
r = -1

#Pour j de 1 a n
for j in range(n):
    #Rechercher max(|A[i,j]|, r+1 <= i <= n)
    max = np.abs(aid).max(0)[j]

    #On note K l'index de ligne du maximum
    k = np.abs(aid).argmax(0)[j]

    #Si A[k,j] != 0 Alors
    if(aid[k,j] != 0):
        r += 1
        #Diviser la ligne k par A[k,j]
        div = float(max)
        aid[k] = aid[k] / aid[k,j]

        #échanger ligne k et r
        if(k != r):
            aid[[k, r]] = aid[[r, k]]

        #Pour i de 1 a m
        for i in range(n):
            #Si i != r Alors
            if(i != r):
                #Soustraire a la ligne i la ligne r multipliée par A[i,j]
                factor = float(aid[i,j])
                for l in range(m):
                    aid[i,l] -= factor * aid[r,l]

#UI
end = perf_counter()
print("Avec cet algp A =")
print(np.hsplit(aid, 2)[1])
print("En utilisant une calculette A^-1 = ")
print(np.linalg.inv(a))

res = end - start
print("Pour une matrice de taille n = " + str(size))
print('CPU Execution time:', res, 's')
print('CPU Execution time:', res * 1000, 'ms')
print('CPU Execution time:', res * 1000000, 'us')


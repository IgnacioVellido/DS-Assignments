import random
import time

N = 100000

def generar_lista(n, lim1=1, lim2=-1) : 
    if (lim2 == -1) :
        lim2 = n
    l = []
    for i in range(n) :
        l.append(random.randint(lim1,lim2))
    
    return l

if __name__ == "__main__" :
    l = generar_lista(100000)
    print (345232 in l)

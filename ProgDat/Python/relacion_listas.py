# ------------------------------------------------------------------------------
# Ignacio Vellido Expósito
# Ejercicios sobre listas
# ------------------------------------------------------------------------------

from generar_lista import generar_lista

# ------------------------------------------------------------------------------

# 1. Escribe una función sum_nums_lista(numeros) que sume todos los números de una lista.
# Compara el tiempo entre usar o no range

def sum_nums_lista(numeros):
    # Más lento
    # s = 0
    # for x in numeros:
    #     s += 0
        
    return sum(numeros)

# ------------------------------------------------------------------------------

# 2. Escribe una función contar_numeros_impares(numeros) que cuente la cantidad de número
# impares que hay en una lista.

def contar_numeros_impares(numeros):
    return len([x for x in numeros if x%2])

# ------------------------------------------------------------------------------

# 3. Escribe una función numeros_pares(numeros) que devuelva los números pares que hay en
# una lista.

def numeros_pares(numeros):
    return [x for x in numeros if not x%2]

# ------------------------------------------------------------------------------

# 4. Escribe una función combinar_listas(l1, l2) que devuelva una lista que esté formada por
# todos los elementos de l1 y a continuación todos los de l2. Por ejemplo combinar_listas([1,
# 2, 8] , [5, 10]) devolvería [1, 2, 8, 5, 10]

def combinar_listas(l1,l2):
    res = l1.copy()
    res.extend(l2)
    return res

# ------------------------------------------------------------------------------

# 5. Escribe una función mezclar(la, lb) que dadas dos listas ordenadas devuelva una lista
# conteniendo los elementos de ambas listas ordenados de forma ascendente.

def mezclar(la,lb):
    return sorted(combinar_listas(la,lb))

# ------------------------------------------------------------------------------

# 6. La traspuesta de una matriz se obtiene intercambiado filas y columna. Escribe una función
# que devuelva la traspuesta de una matriz.

def trans(m):
    # El operador * desempaqueta una lista
    # Si hacemos zip sobre *m cap en un loop capturamos cada columna

    t = m   # Creamos copia de la matriz

    for fil, col in enumerate(zip(*m)):
        t[fil] = list(col)
    
    return t

# ------------------------------------------------------------------------------

# 7. Escribe una función contar_letras(palabra) que tome una palabra como argumento y
# devuelva una lista de pares en la que aparece cada letra junto con el número de veces que
# aparece esa letra en la palabra. Por ejemplo, contar_letras('patata') devuelve [('a', 3), ('p', 1), ('t', 2)].

def contar_letras(palabra):
    letras = set(palabra)

    return [(l, palabra.count(l)) for l in letras]

# ------------------------------------------------------------------------------

# 8. Escribe una función eliminar(l1, l2) que dadas dos listas devuelva una lista en la que estén
# todos los elementos de l1 que no están en l2.

def eliminar(l1,l2):
    return list(set(l1).difference(set(l2)))

# ------------------------------------------------------------------------------

# 9. Escribe una función suma_acumulada(numeros) a la que se le pase una lista de números y
# devuelva una lista en la que el elemento i-ésimo se obtiene como la suma de los elementos
# de la lista entre las posiciones 0 e i. Por ejemplo, para [1, 2. 3] sería [1, 3, 6]

def suma_acumulada(numeros):
    return [sum(numeros[0:i+1],0) for i in range(len(numeros))]

# ------------------------------------------------------------------------------

# 10. Escribe una función parejas(lista) que calcule las parejas distintas de valores que aparecen
# en una lista.
# [1,2,3] -> [11, 12, 13, 21, 22, 23, 31, 32, 33]

def parejas(lista):
    par = []

    for x in lista:
        for y in lista:
            par.append((x,y))

    return par

# ------------------------------------------------------------------------------

# 11. Escribe una función cadena_mas_larga(cadenas) a la que se pasa una lista de palabras y que
# devuelva la palabra más larga.

def cadena_mas_larga(cadenas):
    return sorted(cadenas, key=len)[-1]

# ------------------------------------------------------------------------------

# 12. Escribe una función suma_primer_digito(numeros) que devuelva la suma de los primeros
# dígitos de todos los números de la lista que se pasa como argumento.

def suma_primer_digito(numeros):
    return sum([int(str(x)[0]) for x in numeros])

# ------------------------------------------------------------------------------

# 13. Un vector disperso es aquel que tiene muchos elementos nulos. Para ese tipo de vectores, la
# representación más adecuada es guardar únicamente los elementos no nulos. Escribe una
# función dispersa(v) a la que se le pase una lista representando un vector disperso y que
# devuelva el número de elementos del vector junto con una lista de pares (pos, elem) con
# cada una de las posiciones en las que hay un elemento no nulo y el elemento. Ejemplo:
# (1,0,0, 5, 4, 0, 0, 0) sería ([(0,1), (3,5), (4,4)], 8)

def dispersa(v):
    pos = [(i,x) for i,x in enumerate(v) if x]
    return (pos,len(v))

# ------------------------------------------------------------------------------

# 14. Escribe una función que saque de forma aleatoria todas las cartas de una baraja hasta que
# quede vacía. Para ello debe usar una lista que tenga inicialmente todas las cartas.

from random import sample

def sacar_carta(baraja):
    num = len(baraja)
    orden = sample(range(num), num)

    for carta in orden:
        print("Carta nº {} - {}".format(carta, baraja[carta]))

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

lista = generar_lista(3)
lista2 = generar_lista(2)
lista3 = generar_lista(4, 10, 20)
matriz = [generar_lista(3) for _ in range(3)]

print("Lista: {}".format(lista))
print("Lista2: {}".format(lista2))
print("Lista2: {}".format(lista3))
print("Matriz: {}".format(matriz))
print("",
    sum_nums_lista(lista),
    contar_numeros_impares(lista),
    numeros_pares(lista),
    combinar_listas(lista, lista),
    mezclar(lista, lista),
    trans(matriz),
    contar_letras("dfghjkkjyuhfjka"),
    eliminar(lista, lista2),
    suma_acumulada(lista),
    parejas(lista),
    cadena_mas_larga(["dfgh","jk","kjyu","h","fjka"]),
    suma_primer_digito(lista3),
    dispersa([0,0,1,2,0,1,0,4,0]),
    sep = "\n--- "
)
sacar_carta(lista)
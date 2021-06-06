# ------------------------------------------------------------------------------
# Ignacio Vellido Expósito
# Ejercicios sobre cadenas
# ------------------------------------------------------------------------------

# Escribe una función contar_letras(palabra, letra) que devuelva el número de veces que aparece una letra en una palabra.

def contar_letras(palabra, letras):
    return palabra.count(letras)

# ------------------------------------------------------------------------------

# Escribe una función eliminar_letras(palabra, letra) que devuelva una versión de palabra que no contiene el carácter letra. 

def eliminar_letras(palabra, letra):
    return palabra.replace(letra, "")

# ------------------------------------------------------------------------------

# Escribe una función mayusculas_minusculas(palabra) que devuelva una cadena en la que las mayúsculas y las minúsculas estén al contrario. 

def mayusculas_minusculas(palabra):
    return palabra.swapcase()

# ------------------------------------------------------------------------------

# Escribe una función buscar(palabra, sub) que devuelva la posición en la que se puede encontrar sub dentro de palabra o -1 en caso de que no esté. 

def buscar(palabra, sub):
    return palabra.find(sub)

# ------------------------------------------------------------------------------

# Escribe una función num_vocales(palabra) que devuelva el número de vocales que aparece en la palabra. 

def num_vocales(palabra):
    return sum([palabra.lower().count(vocal) for vocal in ['a','e','i','o','u']], 0)

# ------------------------------------------------------------------------------

# Escribe una función vocales(palabra) que devuelva las vocales que aparecen en la palabra. 

def vocales(palabra):
    return [v for v in ['a','e','i','o','u'] if v in palabra.lower()]

# ------------------------------------------------------------------------------

# Escribe una función mayusculas(palabra) que devuelva la palabra pasada a mayúsculas. 

def mayusculas(palabra):
    return palabra.upper()

# ------------------------------------------------------------------------------

# Escribe una función inicio_fin_vocal(palabra) que determine si una palabra empieza y acaba con una vocal. 

def inicio_fin_vocal(palabra):
    return palabra[0].lower() in "aeiou" and palabra[-1].lower() in "aeiou"

# ------------------------------------------------------------------------------

# Escribe una función elimina_vocales(palabra) que elimine todas las vocales que aparecen en la palabra. 

def elimina_vocales(palabra):
    v = ['a','e','i','o','u', 'A','E','I','O','U']
    for vocal in v:
        palabra = palabra.replace(vocal, "")
    
    return palabra

# ------------------------------------------------------------------------------

# Escribe una función es_inversa(palabra1, palabra2) que determine si una palabra es la misma que la otra pero con los caracteres en orden inverso. Por ejemplo 'absd' y 'dsba' 

def es_inversa(palabra1, palabra2):
    return palabra1 == palabra2[::-1]

# ------------------------------------------------------------------------------

# Escribe una función comunes(palabra1, palabra2) que devuelva una cadena formada por los caracteres comunes a las dos palabras.

def comunes(palabra1, palabra2):
    return set(palabra1).intersection(palabra2)

# ------------------------------------------------------------------------------

# Escribe una función eco_palabra(palabra) que devuelva una cadena formada por palabra repetida tantas veces como sea su longitud. Por ejemplo 'hola' -> 'holaholaholahola' 

def eco_palabra(palabra):
    return palabra * len(palabra)

# ------------------------------------------------------------------------------

# Escribe una función palindromo(frase) que determine si frase es un palíndromo. Es decir, que se lea igual de izquierda a derecha que de derecha a izquierda (sin considerar espacios). 

def palindromo(frase):
    return es_inversa(frase.lower(), frase.lower())

# ------------------------------------------------------------------------------

# Escribe una función orden_alfabetico(palabra) que determine si las letras que forman palabra aparecen en orden alfabético. Por ejemplo: 'abejo' 

def orden_alfabetico(palabra):
    return list(palabra) == sorted(palabra)

# ------------------------------------------------------------------------------

# Escribe una función todas_las_letras(palabra, letras) que determine si se han usado todos los caracteres de letras en palabra. 

def todas_las_letras(palabra, letras):
    return len(set(letras).difference(palabra.lower())) == 0

# ------------------------------------------------------------------------------

# Nota: Merece más agrupar con itertools que recorrer la cadena a mano
from itertools import groupby

# Escribe una función es_triple_doble(palabra) que determine si palabra tiene tres pares de letras consecutivos. Por ejemplo: abgghhkkerf 

def es_triple_doble(palabra):
    # Contar ocurrencias
    groups = groupby(palabra)

    # Pasarlo a lista de pares letra-num
    counts = []
    for k, g in groups:
        counts.append([k,len(list(g))])
    
    # Comprobar consecutivos
    for i in range(len(counts)-2):
        if counts[i][1] == 2 and counts[i+1][1] == 2 and counts[i+2][1] == 2:
            return True
    
    return False

# ------------------------------------------------------------------------------

# Escribe una función trocear(palabra, num) que devuelva una lista con trozos de tamaño num de palabra. 

def trocear(palabra, num):
    return [palabra[i:i+num] for i in range(0, len(palabra), num)]

# ------------------------------------------------------------------------------

# Un anagrama de una palabra pall es una palabra formada con las mismas letras que pal1 pero en orden distinto. Escribe una función anagrama(palabra1, palabra2) que determine si es una anagrama. Ejemplo: marta - trama

def anagrama(palabra1, palabra2):
    return set(palabra1.lower()) == set(palabra2.lower())

# ------------------------------------------------------------------------------

# Un pangrama es una frase que contiene todas las letras del alfabeto. Escribe una función pangrama(frase) que determine si frase es o no un pangrama. Se usa para mostrar tipos de letra. Ejemplo "Benjamín pidió una bebida de kiwi y frasa. Noé, sin vergüenza, la más exquisita champaña del menú."

def pangrama(frase):
    return todas_las_letras(frase, 'áéíóúüabcdefghijklmnopqrstuvwxyz')

# ------------------------------------------------------------------------------

# En criptografía, un código César es una técnica de encriptado muy simple en la que cada letra del texto se reemplaza por otra letra que se encuentra un número fijo de posiciones más adelante en el alfabeto. Por ejemplo, si el desplazamiento es 3, la A se reemplazará con una D, la B con una E, la Z con una C, etc.
# Usando las funciones ord y chr construye una función a la que se le pase un string y devuelva su versión encriptada con desplazamiento arbitrario. encriptar(cad, desp)

def encriptar(cad, desp):
    # Aplica a cada elemento de la cadena una función lambda que aplica el desplazamiento
    # Al resultado lo recoge en una lista y lo convierta a un único string
    return ''.join(list(map(lambda x: chr(ord(x) + desp), cad)))

# ------------------------------------------------------------------------------

# Escribe una función suma_digitos(cad) que haga la suma de los dígitos de un número que está en cad. Modificar ahora la función para que también funcione si cad es un int.

def suma_digitos(cad):
    return sum(list(map(int, cad)), 0)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

palabra = "hOlA"
print("",
    contar_letras(palabra, "l"),
    eliminar_letras(palabra, "l"),
    mayusculas_minusculas(palabra),
    buscar(palabra, "hOa"),
    num_vocales(palabra),
    vocales(palabra),
    mayusculas(palabra),
    inicio_fin_vocal(palabra),
    elimina_vocales(palabra),
    es_inversa(palabra, "AlOh"),
    comunes(palabra, "asfol"),
    eco_palabra(palabra),
    palindromo("Abalaba"),
    orden_alfabetico("abejo"),
    todas_las_letras(palabra, "hola"),
    es_triple_doble("abgghhkk"),
    trocear(palabra, 1),
    anagrama("marta", "trama"),
    pangrama("Benjamín pidió una bebida de kiwi y frasa. Noé, sin vergüenza, la más exquisita champaña del menú"),
    encriptar(palabra, 3),
    suma_digitos("123"),
    sep = "\n--- "
)
# Este contenedor incluirá una instalación completa de Python 3.6 o superior
# además de los paquetes o bibliotecas habituales para análisis de datos con este lenguaje
# de programación: pandas, scikit-learn, scipy, numpy, matplotlib, …

# https://hub.docker.com/_/python/

FROM python:3.7
LABEL author="Ignacio Vellido Expósito"
ENV http_proxy http://stargate.ugr.es:3128

RUN apt-get update

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Copiar scripts de prueba 
# (se podría hacer separadamente, por automatizarlo se añade aquí)
COPY testDocker.py /home/testDocker.py

# Si hay que ejecutar algún script por defecto al acabar
# Mejor con pruebas ?
# CMD [ "python", "./cdpython.py" ]
CMD "python /home/testDocker.py > /home/testOutput.py"

# ------------------------------------------------------------------------------
# Versión base
# ------------------------------------------------------------------------------

# FROM ubuntu
# LABEL author="Ignacio Vellido Expósito"
# ENV http_proxy http://stargate.ugr.es:3128

# RUN apt-get update && \
#     apt-get install -y python3

# WORKDIR /usr/src/app

# COPY requirements.txt ./

# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# COPY . .

# # Copiar scripts de prueba 
# # (se podría hacer separadamente, por automatizarlo se añade aquí)
# COPY testDocker.py /home/testDocker.py

# # Si hay que ejecutar algún script por defecto al acabar
# # Mejor con pruebas ?
# # CMD [ "python", "./cdpython.py" ]
# CMD "python /home/testDocker.py > /home/testOutput.py"
# Este contenedor incluirá una instalación completa de Python 3.6 o superior
# además de los paquetes o bibliotecas habituales para análisis de datos con este lenguaje
# de programación: pandas, scikit-learn, scipy, numpy, matplotlib, …

# https://hub.docker.com/_/python/

FROM python:3
LABEL author="Ignacio Vellido Expósito"
ENV http_proxy http://stargate.ugr.es:3128

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Si hay que ejecutar algún script por defecto al acabar
# Mejor con pruebas ?
CMD [ "python", "./cdpython.py" ]
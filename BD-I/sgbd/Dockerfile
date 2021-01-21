# Este contenedor debe incluir un sistema gestor de bases de datos MySQL o de la
# familia (p.ej. MariaDB) configurado para permitir consultas desde el exterior del
# contenedor.

# https://hub.docker.com/_/mariadb/

FROM mariadb
LABEL author="Ignacio Vellido Expósito"
ENV http_proxy http://stargate.ugr.es:3128

RUN apt-get update
# RUN apt-get update && \
#     apt-get install -y openssh-server openssh-client

# Abrir puerto para consultas (mariadb abre el 3306 por defecto, redirigirlo al 16100)
# Port: 16100 
# Port: 16101

# Subir BD de prueba
COPY testDocker.sql /home/testDocker.sql

# ------------------------------------------------------------------------------
# Versión base
# ------------------------------------------------------------------------------

# FROM ubuntu
# LABEL author="Ignacio Vellido Expósito"
# ENV http_proxy http://stargate.ugr.es:3128

# RUN apt-get update && \
#     apt-get install -y mariadb-server \
#                         # openssh-server \
#                         # openssh-client

# # Abrir puerto para consultas (mariadb abre el 3306 por defecto, redirigirlo al 16100)
# EXPOSE 3306

# # Subir BD de prueba
# COPY testDocker.sql /home/testDocker.sql
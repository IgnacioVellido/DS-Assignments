# Este contenedor debe incluir un sistema gestor de bases de datos MySQL o de la
# familia (p.ej. MariaDB) configurado para permitir consultas desde el exterior del
# contenedor.

# https://hub.docker.com/_/mariadb/

# FROM mysql
FROM mariadb
LABEL author="Ignacio Vellido Expósito"
ENV http_proxy http://stargate.ugr.es:3128

RUN apt-get update && \
    apt-get install -y openssh-server openssh-client

# Abrir puerto para consultas
EXPOSE 8888
FROM ubuntu
LABEL author="Ignacio Vellido Expósito"
ENV http_proxy http://stargate.ugr.es:3128

RUN apt-get update && \
    apt-get install -y mariadb-server
                        # openssh-server \
                        # openssh-client

# Open port to listen to
EXPOSE 3306

# Upload test DB
COPY testDocker.sql /home/testDocker.sql
# Falta integrarla

# Launch MariaDB
CMD [ "mysqld" ]
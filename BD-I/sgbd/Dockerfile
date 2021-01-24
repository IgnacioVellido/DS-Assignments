FROM ubuntu
LABEL author="Ignacio Vellido Expósito"
# ENV http_proxy http://stargate.ugr.es:3128

RUN apt-get update && \
    apt-get install -y mysql-server

# Ensure we won't bind to localhost only
RUN grep -v bind-address /etc/mysql/my.cnf > temp.txt \
  && mv temp.txt /etc/mysql/my.cnf


# Upload test DB
COPY testDocker.sql /home/testDocker.sql
# Falta integrarla

VOLUME /var/lib/mysql
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
# Open port to listen to
EXPOSE 3306

# Launch MySQL
CMD [ "mysqld" ]
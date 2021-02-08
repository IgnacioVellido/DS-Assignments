FROM mysql
LABEL author="Ignacio Vellido Expósito"

# Upload test DB
ENV MYSQL_DATABASE testDB
ENV MYSQL_USER testUser
ENV MYSQL_PASSWORD testPass
ENV MYSQL_ROOT_PASSWORD root

# All scripts in docker-entrypoint-initdb.d/ are automatically
# executed during container startup
COPY testDocker.sql /docker-entrypoint-initdb.d/
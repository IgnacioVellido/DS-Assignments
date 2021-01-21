# Este contenedor incluirá una instalación completa de R junto con al menos los
# siguientes paquetes: tydiverse, caret, RSNNS, frbs, FSinR, forecast

# https://environments.rstudio.com/docker

# FROM rocker/r-base
# LABEL author="Ignacio Vellido Expósito"
# ENV http_proxy http://stargate.ugr.es:3128

# RUN apt-get update && \
#     apt-get install -y r-cran-tidyverse

# # Se podrían separar para evitar que si uno falla se corte la ejecución. 
# # Como está testeado se deja resumido

# RUN R -e "install.packages(c('tidyverse','caret','RSNNS','frbs','FSinR','forecast'),
#                             dependencies=TRUE, repos='http://cran.rstudio.com/')"

# # Copiar scripts de prueba 
# # (se podría hacer separadamente, por automatizarlo se añade aquí)
# COPY testDocker.R /home/testDocker.R

# # Lanzar script
# CMD R -e "source('/home/testDocker.R')"
# # CMD "R /home/testDocker.R > /home/testOutput.txt"

# ------------------------------------------------------------------------------
# Versión base
# ------------------------------------------------------------------------------

FROM ubuntu
LABEL author="Ignacio Vellido Expósito"
ENV http_proxy http://stargate.ugr.es:3128

RUN apt-get update && \
    apt-get install -y r-base \
                        r-cran-tidyverse

RUN R -e "install.packages(c('caret','RSNNS','frbs','FSinR','forecast'), \
                            dependencies=TRUE, repos='http://cran.rstudio.com/')"

# Copiar scripts de prueba 
# (se podría hacer separadamente, por automatizarlo se añade aquí)
COPY testDocker.R /home/testDocker.R

# Lanzar script
CMD R -e "source('/home/testDocker.R')"
# CMD "R /home/testDocker.R > /home/testOutput.txt"
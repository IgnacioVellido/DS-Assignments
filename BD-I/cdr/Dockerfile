# Este contenedor incluirá una instalación completa de R junto con al menos los
# siguientes paquetes: tydiverse, caret, RSNNS, frbs, FSinR, forecast

# https://environments.rstudio.com/docker

FROM rstudio/r-base
LABEL author="Ignacio Vellido Expósito"
ENV http_proxy http://stargate.ugr.es:3128

# Separados por si alguno de fallo
# tydiverse, caret, RSNNS, frbs, FSinR, forecast
RUN R -e "install.packages('tydiverse', dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('caret', dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('RSNNS', dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('frbs', dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('FSinR', dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('forecast', dependencies=TRUE, repos='http://cran.rstudio.com/')"

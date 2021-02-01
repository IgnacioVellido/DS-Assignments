FROM ubuntu
LABEL author="Ignacio Vellido Expósito"
# ENV http_proxy http://stargate.ugr.es:3128

# To don't get asked about geographic location (disable dpkg interactivity)
ENV DEBIAN_FRONTEND=noninteractive

# Install R and tidyverse
RUN apt-get update && \
    apt-get install -y r-base \
                        r-cran-tidyverse

RUN R -e "install.packages(c('caret','RSNNS','frbs','FSinR','forecast'), \
                            dependencies=TRUE, repos='http://cran.rstudio.com/')"

# Launch test script
COPY testDocker.R /home/testDocker.R
RUN cd /home && \
    Rscript /home/testDocker.R > /home/testOutput.txt

EXPOSE 22

# Launch R
CMD [ "R" ]
FROM ubuntu
LABEL author="Ignacio Vellido Expósito"
# ENV http_proxy http://stargate.ugr.es:3128

# To don't get asked about geographic location (disable dpkg interactivity)
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3 python3-pip

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Launch test script
COPY data /home/data
COPY testDocker.py /home/testDocker.py
RUN cd /home && \
    python3 /home/testDocker.py > /home/testOutput.txt


# For Azure (to keep container alive)
CMD [ "tail", "-f", "/dev/null" ]

# Launch Python
# CMD [ "python3" ]
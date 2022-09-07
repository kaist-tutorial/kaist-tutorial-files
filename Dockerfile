# Specify which base layers (default dependencies) to use
# You may find more base layers at https://hub.docker.com/
FROM python:3.7
#
# Creates directory within your Docker image
RUN mkdir -p /app/src/
# Don't place anything in below folders yet, just create them
RUN mkdir -p /app/data/archive/fruits-360_dataset/fruits-360/Training/Apple
RUN mkdir -p /app/data/archive/fruits-360_dataset/fruits-360/Training/Lemon
RUN mkdir -p /app/data/archive/fruits-360_dataset/fruits-360/Test/Apple
RUN mkdir -p /app/data/archive/fruits-360_dataset/fruits-360/Test/Lemon
RUN mkdir -p /app/model/
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
#
# Copies file from your Local system TO path in Docker image
COPY main.py /app/src/
COPY requirements.txt /app/src/  
#
# Installs dependencies within you Docker image
RUN pip3 install -r /app/src/requirements.txt
#
# Enable permission to execute anything inside the folder app
RUN chgrp -R 65534 /app && \
    chmod -R 777 /app

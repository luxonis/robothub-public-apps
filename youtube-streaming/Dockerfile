FROM ghcr.io/luxonis/robothub-app:2022.269.1517-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg

ARG FILE=app.py
ADD $FILE run.py

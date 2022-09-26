FROM ghcr.io/luxonis/robothub-app:2022.269.1517-ubuntu22.04

RUN pip3 install -U av

ARG FILE=app.py
ADD $FILE run.py

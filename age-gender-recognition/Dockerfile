FROM ghcr.io/luxonis/robothub-app:2022.269.1517-ubuntu22.04

RUN pip3 install -U numpy opencv-contrib-python-headless

ARG FILE=app.py

ADD script.py .

ADD det_model.blob .
ADD det_model.json .
ADD age_model.blob .
add age_model.json .

ADD $FILE run.py

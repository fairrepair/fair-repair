FROM ubuntu:16.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get autoremove -y

RUN apt-get install -y build-essential wget git python3 python3-pip unzip

RUN python3 -m pip install --upgrade pip && python3 -m pip install z3-solver

RUN python3 -m pip install pandas numpy scikit-learn PuLP matplotlib

COPY . .
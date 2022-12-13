FROM openfabric/openfabric-pyenv:0.1.9-3.8

RUN mkdir cognitive-assistant
WORKDIR /cognitive-assistant
COPY . .
RUN pip install torch torchtext pandas
RUN poetry install -vvv --no-dev
EXPOSE 5000
CMD ["sh","start.sh"]
FROM openfabric/openfabric-pyenv:0.1.9-3.8

RUN mkdir cognitive-assistant
WORKDIR /cognitive-assistant
COPY . .
RUN pip install torch torchtext pandas bs4 scikit-learn
RUN poetry install -vvv --no-dev
# degrading werkzeug and flask because newer version crash build with openfabric_pysdk
RUN pip install werkzeug==2.0.3
RUN pip install flask==2.1.3
EXPOSE 5000
CMD ["sh","start.sh"]
import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
import model.model as model
import model.queries as queries

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    # check if the required modules are installed
    try:
        import torch
        import torchtext
        import sklearn
        import pandas
    except:
        return "Please install ['torch', 'torchtext', 'sklearn', 'pandas'] python packages for the chatbot to work."
    
    return True


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        response = process(text)
        output.append(response)

    return SimpleText(dict(text=output))

############################################################
# Function to process the query              
############################################################
def process(text: str) -> str:
    """
    Process the question and generate a response.

    :param text: [String] The science question.

    :return: [String] The answer of the question.
    """
    intent = model.predict(text)
    if intent=="question_intent":
        try:
            response = queries.perform_query(text)
        except:
            response = "I'm afraid I do not know the answer for this question. Can you please rephrase or try something else?"
    else:
        response = queries.dialogs[intent]
    return response
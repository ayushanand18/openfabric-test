# Notes from Ayush Anand
This file contains the report made for the project containing information about the algorithm, method, results, and relevant discussion.

## Table of contents
+ [Introduction](#introduction)
+ [Necessary files](#files-included)
+ [Possible solution (and algorithm)](#possible-solution)
    + [Method](#method)
        + [Algorithm](#algorithm)
        + [Python implementation](#python-implementation)
+ [Discussion and concluding remarks](#discussion-and-conclusion)

## Introduction
The goal is to implement an NLP chatbot answering questions about science.

## Files included
The `model` directory contains all the files used for inferencing the trained model.
file|description
----|-----------
`intent_pair.csv`|dataset used to train the `TextClassificationModel`.
`notebook`|The Jupyter notebook used for training the PyTorch based model [can be found here](https://www.kaggle.com/code/theayushanand/text-intent-classification-using-pytorch).
`model.py`|script for inferencing the model.
`queries.py`|script for knowledge discovery when `question_intent` is trigerred and define responses to other intents.

## Possible solution
There are two stages in creating such a chatbot.
1. Define intents and responses that handles normal user queries.
2. Identify whether the query matches a normal intent or is a science question.
    + If it's a general text then pick the most correlated intent.
    + If it's a question based science then search for the answer to it.

We will use `PyTorch` to build classify input text into pre-defined `intents`. 

For the first part which is building a general chat flow we will define custom intents for general discussions. We built a `TextClassificationModel` using PyTorch and used `torchtext` for pre-processing pipeline. 
1. We created a custom dataset and defined some default intents. [Ref dataset](./model/intent_pair.csv)
2. Then we trained a PyTorch based Sequential model on the dataset. [Ref notebook](https://www.kaggle.com/code/theayushanand/text-intent-classification-using-pytorch)
3. We perform segmentation/tokenisation using `torchtext` on each query string and replace it with corresponding `int id` from the vocabulary we built over our training data.
4. We predict the `intent` from the query and perform action as detailed in the above algorithm.

For the second part we will need to discover knowledge from the internet and the method has been detailed below.

## Method
This method uses web scraping to extract information from a Google search of the question.
A Google is fast and convenient, and can extract knowledge very quickly from a vast number of resources including Wikipedia articles, science forums, etc,.

### Algorithm
This method works in the following manner:
1. Perform a Google search for the question.
2. Extract all query results.
3. If there is a snippet from Wikipedia article then return it as response because it is highly likely to answer the question.
4. If there is a featured snippet from Google then return it as response because it is highly likely to answer the question.
5. If both of the conditions fail, then extract the query excerpts alongside the query results and perform a `cosine similarity` test between question and the excerpt. 
    + Pick the most correlated answer and fetch that search result.
    + Pick the paragraph from the fetched page which was presented as answer.
    + Return this as the response.

### Python Implementation
```python
# import modules
import requests
import bs4
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
Tfidf_vect = TfidfVectorizer()

def return_response(text: str) -> str:
    # find the most correlated query
    def findbest(query, res_list):
        l = []
        for x in res_list:
            data = [query, x]
            vector_matrix = Tfidf_vect.fit_transform(data)
            cosine_similarity_matrix = cosine_similarity(vector_matrix)
            l.append(cosine_similarity_matrix[0][1])
        return l.index(max(l))

    # construct url
    url = f"https://google.com/search?q={text}"
    # also use some real browser headers because google might think we are bots
    headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"}
    # post request
    res = requests.get(url, headers=headers)
    # parse the html using bs4
    parsed = bs4.BeautifulSoup(res.text, "html.parser")
    # since search results are h3, lets pick the h3 tags
    headings = parsed.find_all("h3")
    # print(headings)
    headingsT = [info.getText() for info in headings]
    wiki = 1 if "Description" in headingsT else 0
    featured = 1 if "About featured snippets" in parsed.body.text else 0

    # to get wikipedia's excerpt
    if wiki:
        response = headings[headingsT.index("Description")].find_parent().find('span').text
    # check if an h3 with text: "Description" is present
    # move to its parent div, then go to its child span & get the text.
    # check if already a featured snippet is there then return it
    elif featured:
        response = parsed.body.find(text="About featured snippets").find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find('div').find_all('span')[0].text
    # or else perform a cosine similarity test and return the most correlated response
    else:
        responses = [x.find_parent().find_parent().find_parent().find_parent().find_parent().find_all('span')[-1].text for x in headings]
        result = findbest(text, responses)
        response = responses[result][:-3]
        address = headings[result].find_parent().find_parent().find_parent().find_parent().find_parent().find('a').attrs["href"]
        
        best_res = bs4.BeautifulSoup(requests.get(address, headers=headers).text, "html.parser")
        if (f"{response[:50].strip()}" in best_res.body.text):
            response = best_res.body.find(text=re.compile(f"^.*{response[:50].strip()}.*$")).text

    return response
```

## Discussion and Conclusion
+ The `TextClassificationModel` is pretty good at detecting the best matched `intent` to a user's query as can be seen from the model inference in the Jupyter notebook.
+ The `TextClassificationModel` we have built has limited pre-defined intents at the moment. For a general model we need to expand it by adding more `intents`. At this time, it has only 3 `intents` apart from detecting if a science-based question is asked. 
+ Also, the model is pre-trained right now. We can develop a feedback pipeline to it for continous learning after the initial training so that the classification continous to get more accurate.
+ Currently, we have only one set of response for each intent (except the `question_intent` wherein it does knowledge discovery from the web). We can expand it to support a varied response set for each intent to add a human touch.

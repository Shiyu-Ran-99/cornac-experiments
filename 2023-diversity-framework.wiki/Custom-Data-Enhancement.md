This page explains a data enhancement pipeline that can be used to add diversity-related features to existing datasets. The pipeline takes as input a CSV file and outputs a new JSON file with the added features.

## Features supported by the Data Enhancement Pipeline

Our data enhancement pipeline is designed to extract and analyze various features from textual data using NLP tools. Furthermore, the pipeline also allows for extracting existing external or metadata, if it is available. This pipeline can be easily extended with other features by adding your own custom functions. For more information on how to use the pipeline and implementation detail of each feature extraction, please refer to the sections provided below.

Here are the key features currently supported by the pipeline:

| features              | metrics           | related function | requirements                                                                                                                              | external library       |
|-----------------------|-------------------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| category              | calibration, binomial diversity, Gini coefficient, intra-list diversity, expected intra-list diversity     | [category.py](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/blob/DataEnhancement/DataEnhancement/function/category.py)      | item id and  category meta data (item id and category) file or item text and candidate category list                                      | [transformer](https://huggingface.co/tasks/zero-shot-classification)            |
| complexity            | calibration       | [complexity.py](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/blob/DataEnhancement/DataEnhancement/function/complexity.py)    | item text which has more than 100 words                                                                                                   | [py-readability-metrics](https://pypi.org/project/readability/) |
| sentiment             | activation   | [sentiment.py](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/blob/DataEnhancement/DataEnhancement/function/sentiment.py)     | item text                                                                                                                                 | [nltk](https://www.nltk.org/install.html)                   |
| named entity          |       -            | [ner.py](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/blob/DataEnhancement/DataEnhancement/function/ner.py)           | item text                                                                                                                                 | [spaCy](https://spacy.io/usage/linguistic-features#named-entities)                  |
| enriched named entity |      -             |      [enrich_ne.py](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/blob/DataEnhancement/DataEnhancement/function/enrich_ne.py)            | named entities                                                                                                                            | [wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page)               |
| region       | intra-list diversity, expected intra-list diversity    | [region.py](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/blob/DataEnhancement/DataEnhancement/function/region.py?ref_type=heads)         | enriched named entities (list of geopolitical entities such as countries, cities, states, and regions appeared in text)                                      |                        |
| political party       | representation    | [party.py](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/blob/DataEnhancement/DataEnhancement/function/party.py)         | enriched named entities (list of person entities appeared in text and their political parties )                                      |                        |
| minor-major ratio     | alternative voice | [min_maj.py](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/blob/DataEnhancement/DataEnhancement/function/min_maj.py)       | enriched named entities (list of person entities appeared in text and their gender, citizen, ethinicity, place of birt information ) |                        |
| story                 | fragmentation     | [story.py](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/blob/DataEnhancement/DataEnhancement/function/story.py)         | item date, item text and item category                                                                                                    | [python-louvain](https://python-louvain.readthedocs.io/en/latest/)         |

#### 1. Category 

This feature identifies the category of item, for instance, if the item is a news article, the category can represent its topic or subject matter. Category feature can be used for calibration, binomial diversity and Gini coefficient metrics. It can also easily transformed as feature vector and used for intra-list diversity and expected intra-list diversity metrics.
  
Users have the flexibility to expand the category feature using two distinct ways:   

**a) Using Metadata Information**   
If user has an external `metadata file` which has item id and its category information, the system seamlessly links item IDs to the metadata, effectively merging the data much like joining two tables.  
  
Let't say input data and metadata csv file looks like following: 

`input_file_path = './input/id_text.csv'`. 
| id | text     |
|----|----------|
| 1  | The global economic recovery is gathering momentum, offering hope for a brighter future...       |
| 2  | The game had it all—a dramatic comeback, jaw-dropping plays, and a thrilling overtime period that will be etched in sports history...      |
| 3  | The introduction of this vaccine sends a powerful message of hope to people worldwide...        |
  
`metadata_file_path = './input/news.csv'`
| id | category |
|----|----------|
| 1  | Economics        |
| 2  | Sports      |
| 3  | Public health       |

```python

# meta_data : metadata file dataframe
# id : id row (value) of input file dataframe

category = meta_data[meta_data['id']==row]['category'] # Retrieve category value from metadata if the input data's id exists.

if not category.empty : 
    category = category.values[0]
else:
    category = -1

```

**2) Using Zero-Shot Classification**   
If there is no available metadata, user can provide `a list of potential labels`, which serves as input for a pre-trained [zero-shot learning classifier](https://huggingface.co/tasks/zero-shot-classification) from hugging face. Then classifier gauges the textual content of each item and assigns the label that best matches it. To heighten accuracy, a threshold of 0.5 has been implemented, with scores below this value resulting in classification as "unknown."`

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

# text : text row (value) of input file dataframe
# candidate_labels : given by user, e.g. ["sports", "public health", "economics"]

classifier(text, candidate_labels) # it will return label and its probability

```
Both case, output would look like :

| id | text | category       |
|----|------|----------------|
| 1  | The global economic recovery is gathering momentum, offering hope for a brighter future... | economics     |
| 2  | The game had it all—a dramatic comeback, jaw-dropping plays, and a thrilling overtime period that will be etched in sports history... | sports |
| 3  | The introduction of this vaccine sends a powerful message of hope to people worldwide... | public health |



#### 2. Sentiment 

The sentiment feature is the emotional tone or polarity expressed within a piece of text, and this can be used for the representation metric. 
  
One of the most popular python library to compute the sentiment of text is [NLTK's sentiment analyzer](https://www.nltk.org/install.html). The NLTK sentiment analyzer provides a compound score that combines individual positivity, negativity, and neutrality scores. This compound score falls within a range, typically between -1 (extremely negative) and 1 (extremely positive). The score's magnitude indicates the intensity of the expressed sentiment. Sentiment feature can be used for activation metric. 

```python
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# text: text row (value) of input file dataframe

scores = analyzer.polarity_scores(text) # text: "I am happy"
sentiment = scores['compound']          # 1.0

``` 

With the same input as above, the output would look like : 

| id | text     | sentiment |
|----|----------|-----------|
| 1  | The global economic recovery is gathering momentum, offering hope for a brighter future... |    0.7  |
| 2  | The game had it all—a dramatic comeback, jaw-dropping plays, and a thrilling overtime period that will be etched in sports history...      |0.9|
| 3  | The introduction of this vaccine sends a powerful message of hope to people worldwide...        |1.0|

#### 3. Complexity 
The complexity score measures the reading ease of the text using the Flesch-Kincaid reading ease test. Many U.S. states also use Flesch-Kincaid Grade Level to score other legal documents such as business policies and financial forms to ensure their texts are no higher than a ninth grade level of reading difficulty. A lower score indicates a more complex text, while a higher score suggests greater readability. The complexity of text can be computed [Python Readability library] (https://pypi.org/project/readability/) and used for calibration metric.

```python
import nltk
nltk.download('punkt')
from readability import Readability

# text: text row (value) of input file dataframe

r = Readability(text)
fk = r.flesch_kincaid()
score = fk.score # it will show the Flesch-Kincaid score of given text
```

With the same input as above, the output would look like : 

| id | text     | complexity |
|----|----------|------------|
| 1  | The global economic recovery is gathering momentum, offering hope for a brighter future... |    10.0  |
| 2  | The game had it all—a dramatic comeback, jaw-dropping plays, and a thrilling overtime period that will be etched in sports history...      |5.0|
| 3  | The introduction of this vaccine sends a powerful message of hope to people worldwide...        |7.0|

#### 4. Named Entities 
This feature identifies and counts named entities in the text. This won't be directly able to be used to metric, but need to be post-process. 
  
User can first decide type of entities to extract such as person, event, location etc, then Python natural language processing library [spaCy](https://spacy.io/usage/linguistic-features#named-entities) will extract the named entities appeared in given text. Once entities are extracted, it clusters similar named entities with the Louvain community detection algorithm and compute frequency.

```python
import spacy

NER = spacy.load("en_core_web_sm")

# text: text row (value) of input file dataframe
# ents : given by user, e.g. ['PERSON', 'GPE', 'ORG']

text= NER(text)

for word in text.ents:

    if word.label_ in entities:
      ner.append({'text': word.text, 'alternative':[],  'label': word.label_, 'start_char': word.start_char, 'end_char': word.end_char})
    else:
      continue
``` 


With the same input as above, the output would look like :  

| id | text     | named_entities |
|----|----------|----------------|
| 1  | The global economic recovery is gathering momentum, offering hope for a brighter future... |    [{'text': 'L.A.', 'alternative': ['L.A.'], 'spans': [(87 ,91)], 'frequency': 1, 'label': 'GPE'},{'text': 'Tiffany', 'alternative': ['Tiffany & Co.', 'Tiffany'], 'spans': [(101,112)], [(200,206)], 'frequency': 3, 'label': 'ORG'}]  |
| 2  | The game had it all—a dramatic comeback, jaw-dropping plays, and a thrilling overtime period that will be etched in sports history...      |[{'text': 'Mossimo Giannulli', 'alternative': ['Mossimo Giannulli', 'Giannulli', 'Mossimo'], 'spans': [(52 ,68),(122,131),(215,222)], 'frequency': 3, 'label': 'PERSON'}, ]|
| 3  | The introduction of this vaccine sends a powerful message of hope to people worldwide...        |[]|


#### 5. Enriched Named Entities
Once named entities are collected, it can be further enriched by sending a query to Wikidata with extra information. Currently our pipeline can extend person entity with its given name, family name, occupation, political party, gender, citizenship, ethnicity and place of birth. These recognized named entities with extra information can serve as a valuable resource for calculating various features, such as political viewpoints baed on person's party if he/she is politician or determining the representation of majority/minority groups based on gender, citizenship, ethinicity of person appeared in text.

```python
import requests
import json
import pandas as pd

def get_enriched_ne(ne_list):
    enriched_ne = []
    for entity in ne_list:
        print(entity)

        if entity['label'] == 'PERSON':
            print("found person")
            name = entity['text']
            full_name, info = get_person_data(entity['alternative'])
            info['key'] = name
            info['frequency'] = entity['frequency']
            info['alternative'] = entity['alternative']
            enriched_ne.append(dict({full_name:info}))

        if entity['label'] == 'ORG':
            print("found organization")
            name = entity['text']
            name, info = get_org_data(name)
            info['frequency'] = entity['frequency']
            info['alternative'] = entity['alternative']
            enriched_ne.append(dict({name:info}))

    return enriched_ne

def get_person_data(alternatives):
    ...

def get_org_data(name):
    ...

def execute_query(query):
    """
    Sends a SPARQL query to Wikidata.
    """
    url = 'https://query.wikidata.org/sparql'

    try:
        r = requests.get(url, params={'format': 'json', 'query': query})
        return r

    except (ConnectionAbortedError, ConnectionError):
        return ConnectionError

def read_response(data, label):
    ...

def read_person_response_list(response):
    ...

def read_org_response_list(response):
    ...

def org_data_query(label):
        language_tag = 'en'
        try:
            query = """
            SELECT DISTINCT ?ideology WHERE { 
                ?s wdt:P31 wd:Q7210356. # instance of political organization
                ?s rdfs:label '""" + label + """'@"""+language_tag+""" .

                OPTIONAL {
                  ?s wdt:P1142 ?a .
                  ?a rdfs:label ?ideology .
                  FILTER(LANG(?ideology) = "" || LANGMATCHES(LANG(?ideology), \""""+language_tag+"""\")) }                
            }            
            """

            r = execute_query(query)
            return read_org_response_list(r)

        except ConnectionAbortedError:
            return []

def person_data_query(label):
        language_tag = 'en'
        try:
            query = """
                SELECT DISTINCT ?s ?givenname ?familyname ?occupations ?party ?position ?gender ?citizen ?ethnicity ?place_of_birth ?sexuality WHERE { 
                ?s ?label '""" + label + """'@"""+language_tag+""" .
              OPTIONAL {
                ?s wdt:P735 ?a . 
                ?a rdfs:label ?givenname .
                FILTER(LANG(?givenname) = "" || LANGMATCHES(LANG(?givenname), \""""+language_tag+"""\"))
              }
              OPTIONAL {
                ?s wdt:P734 ?b . 
                ?b rdfs:label ?familyname .
                FILTER(LANG(?familyname) = "" || LANGMATCHES(LANG(?familyname), \""""+language_tag+"""\"))
              }
              OPTIONAL {
                ?s wdt:P106 ?c .
                ?c rdfs:label ?occupations .
                FILTER(LANG(?occupations) = "" || LANGMATCHES(LANG(?occupations), \""""+language_tag+"""\"))
              }
              OPTIONAL {
                ?s wdt:P102 ?d .
                ?d rdfs:label ?party .
                FILTER(LANG(?party) = "" || LANGMATCHES(LANG(?party), \""""+language_tag+"""\"))
              }
              OPTIONAL {
                ?s wdt:P21 ?f .
                ?f rdfs:label ?gender .
                FILTER(LANG(?gender) = "" || LANGMATCHES(LANG(?gender), \""""+language_tag+"""\"))
              }
              OPTIONAL {
                   ?s wdt:P172 ?g . 
                   ?g rdfs:label ?ethnicity .
                   FILTER(LANG(?ethnicity) = "" || LANGMATCHES(LANG(?ethnicity), \""""+language_tag+"""\"))
                }
               OPTIONAL {
                   ?s wdt:P19 ?pb . 
                   ?pb wdt:P17 ?country .
                   ?country rdfs:label ?place_of_birth .
                   FILTER(LANG(?place_of_birth) = "" || LANGMATCHES(LANG(?place_of_birth), \""""+language_tag+"""\"))
                }
              OPTIONAL {
                ?s wdt:P27 ?h .
                ?h rdfs:label ?citizen
                FILTER(LANG(?citizen) = "" || LANGMATCHES(LANG(?citizen), \""""+language_tag+"""\"))
                }
              
            }"""
            r = execute_query(query)
            print("got person wiki response!")
            return read_person_response_list(r)

        except (ConnectionAbortedError, requests.exceptions.ChunkedEncodingError):  # in case the connection fails
            return []
``` 


If named_entity from input data is given as bellow: 
```
[{'text': 'Mossimo Giannulli', 
  'alternative': ['Mossimo Giannulli', 'Giannulli', 'Mossimo'], 
  'spans': [(52 ,68),(122,131),(215,222)],
  'frequency': 3, 
  'label': 'PERSON'}, ]
```

The enriched named entity would look like: 
```
[{'Mossimo Giannulli': {'givenname': ['Mossimo'], 
                        'familyname': ['Giannulli'], 
                        'gender': ['male'], 
                        'occupations': ['cricketer'], 
                        'party': [], 
                        'positions': [], 
                        'citizen': ['Australia'], 
                        'ethnicity': [], 
                        'sexuality': [], 
                        'place_of_birth': ['Australia'], 
                        'frequency': 3, 
                        'alternative': ['Mossimo Giannulli', 'Giannulli', 'Mossimo']}}
```
  
As you can see, attributes existing in wikidata can only be retreived.


#### 6. Region 
The region feature represents a multi-labeled category derived from geographic named entities, rendering it suitable for assessing both intra-list diversity and expected intra-list diversity. To enhance its accuracy, all named entities labeled as 'GPE' are queried in Wikidata, and when a response is received, it is saved as the region.

```python
import requests
import json

def get_region(ne_list):

    regions = []

    for ner in ne_list:
        if ner['label'] == 'GPE':
            region = get_region_data(ner['text'])

            if region =='true' : 
                regions.append(ner['text'].lower())

    return set(regions)


def get_region_data(region):
    ...
    

def execute_query(query):
    """
    Sends a SPARQL query to Wikidata.
    """
    url = 'https://query.wikidata.org/sparql'

    try:
        r = requests.get(url, params={'format': 'json', 'query': query})
        return r

    except (ConnectionAbortedError, ConnectionError):
        return ConnectionError


def region_data_query(label):
    """
    Create SPARQL query 
    """
    language_tag = 'en'
    try:
        query = """
                SELECT ?boolean WHERE { 
                BIND(EXISTS{?item ?label '""" + label + """'@"""+language_tag+""" .} AS ?boolean)
                }"""
        r = execute_query(query)
        return r
            

    except (ConnectionAbortedError, requests.exceptions.ChunkedEncodingError):  # in case the connection fails
        return []
```

If the named entity list in input is : 
```
[{'text': 'L.A.', 'alternative': ['L.A.'], 'frequency': 1, 'label': 'GPE'},
 {'text': 'Tiffany', 'alternative': ['Tiffany & Co.', 'Tiffany'], 'frequency': 3, 'label': 'ORG'},
 {'text': 'Paris', 'alternative': ['Paris'], 'frequency': 1, 'label': 'GPE'},]
```
The output would be : 
```
['L.A','Paris']
```

#### 7. Political party
The political party feature discerns the political viewpoint expressed in the text. It can be extracted based on counting political party of people or political ideology of organizations appeared in the text and their frequency. This feature provides insights into the text's political stance and used for representative metric.

```python
def get_party(ne_list):

  parties = {}

  for entity in ne_list: 
    for entity_name, entity_dict in entity.items(): 

      # if there is a party attribute in extended named entity
      if 'party' in entity_dict.keys():
        if entity_dict['party'] : 
          for party in entity_dict['party']: 
            if not parties.get(party):
              parties[party] = 0
            parties[party] += entity_dict['frequency']

      # if there is a ideology attribute in extended named entity
      if 'ideology' in entity_dict.keys():
        if entity_dict['ideology'] : 
          for party in entity_dict['ideology']: 
            if not parties.get(party):
              parties[party] = 0
            parties[party] += entity_dict['frequency']

  return parties
```


If the enriched named entity list in input is : 
```
[{'Demi Moore': {'givenname': ['Demetria', 'Gene', 'Demi'], 
                 'familyname': ['Moore', 'Guynes'], 
                 'gender': ['female'], 
                 'occupations': ['television actor', 'film producer'], 
                 'party': ['Democratic Party'],
                 'citizen': ['United States of America'], 
                 'ethnicity': [], 
                 'sexuality': [], 
                 'place_of_birth': ['United States of America'], 
                 'frequency': 1, 
                 'alternative': ['Demi Moore']}}]
```

The output would be : 
```
{'Democratic Party' : 1}
```

#### 7. Minority majority ratio
This feature assesses the ratio of minority voices versus majority voices in the text. It specifically focuses on identifying whether the viewpoint holders belong to a "protected group" or not. Examples of such groups include non-male/male, non-white/white, etc. Understanding this ratio offers valuable insights into the diversity of opinions presented in the text. This can be used for alternative voice metrics. 

Currently our pipeline compute three distinct ratios : gender ratio, ethinicity ratio and mainstream ratio based on various attributes extracted from extended named entities feature. 

**i) gender ratio** 
User can first decide which gender should be considered as majority (value can be single or multiple) in config.py file. Then it will count genders considered as majority as majority group and minority as minority group and compute ratio of each group.

**ii) ethinicity ratio** 
User can decide which citizenship, ethiniticy and place of birth should be considered as majority. Even though person's citizenship is belong to majority, one of his/her ethinicity or place of birth does not met majority, it will classified as minority group. 

**iii) mainstream ratio**  
it will simply computed by checking if person has wikidata information, which means this person is known to public. person who does not able to find information from wiki would considered as minority group. 

```python
def get_min_maj_ratio(ne_list, **kwargs): 
  
  major_genders = kwargs['major_gender']
  major_citizens = kwargs['major_citizen']
  major_ethinicities = kwargs['major_ethinicity']
  major_place_of_births = kwargs['major_place_of_birth']

  count = {'gender' : [0,0], 'ethnicity': [0,0], 'mainstream': [0,0]}
  ratio = {}

  for entity in ne_list:
    for entity_name, entity_dict in entity.items():

      # calculate gender score (male as majority, others are minority)
      if 'gender' in entity_dict.keys() and len(entity_dict['gender']) == 1 :
        if entity_dict['gender'][0] in major_genders:
          count['gender'][1] += entity_dict['frequency']
        else: 
          count['gender'][0] += entity_dict['frequency']

      # calculate Ethnicity score (people with a 'United States' ethnicity or place of birth are majority)
      if 'citizen' in entity_dict:
        loop_break = False
        etinity_match = False
        place_of_birth_match = False

        for major_citizen in major_citizens:
            loop_break = False
            if major_citizen in entity_dict['citizen']:
                loop_break = True

                for major_ethnicity in major_ethinicities:
                    if (major_ethnicity in entity_dict['ethnicity']) or (entity_dict['ethnicity'] == []):
                        etinity_match = True

                for major_place_of_birth in major_place_of_births:
                    if (major_place_of_birth in entity_dict['place_of_birth']) or (entity_dict['place_of_birth'] == []):
                        place_of_birth_match = True

                if (etinity_match and place_of_birth_match):
                    count['ethnicity'][1] += entity_dict['frequency']
                    break

                count['ethnicity'][0] += entity_dict['frequency']
                break

        if not loop_break:
          count['ethnicity'][0] += entity_dict['frequency']

      # calculate mainstream score by checking givenname 
      if 'givenname' in entity_dict.keys() :
        count['mainstream'][1] += entity_dict['frequency']
      else:
        count['mainstream'][0] += entity_dict['frequency']

  for k, v in count.items():
    try: 
      ratio[k] = [round(v[0] /(v[1]+v[0]),4), round(v[1] /(v[1]+v[0]),4)]
    except: 
      continue

  return ratio
```

If the enriched named entity list in input is : 
```
[{'Demi Moore': {'givenname': ['Demetria', 'Gene', 'Demi'], 
                 'familyname': ['Moore', 'Guynes'], 
                 'gender': ['female'], 
                 'occupations': ['television actor', 'film producer'], 
                 'party': ['Democratic Party'],
                 'citizen': ['United States of America'], 
                 'ethnicity': [], 
                 'place_of_birth': ['United States of America'], 
                 'frequency': 1, 
                 'alternative': ['Demi Moore']}},
 {'Krista Smith': {'givenname': ['Krista'], 
                   'familyname': ['Smith'], 
                   'gender': ['female'], 
                   'occupations': ['basketball player', 'researcher'], 
                   'party': [], 
                   'citizen': [], 
                   'ethnicity': [], 
                   'place_of_birth': [], 
                   'frequency': 1, 
                   'alternative': ['Krista Smith']}}]
```

The output would be : 
```
 {'gender': [1.0, 0.0], 'ethnicity': [0.5, 0.5], 'mainstream': [0.0, 1.0]}
```


#### 8. Story
The story chain is a cluster in a set of texts according to the principles noted in Nicholls et al. This can be used for fragmentation metric.

It first transform the text to Tf-Idf vector and compute cosine similarity of texts which are same topic within 3-day of time window. Then it is converted to the graph and compute the partition of the graph nodes which maximises the modularity using the Louvain heuristices algorithms with [python-louvain](https://python-louvain.readthedocs.io/en/latest/).

```python
from datetime import datetime, timedelta
import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
from collections import defaultdict
from statistics import mode, StatisticsError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

import itertools
from difflib import SequenceMatcher



def get_story(df):
    '''
    this code is from https://github.com/svrijenhoek/RADio/blob/f1ce0d8bb0d7235f0c48b1745a8a81060683846a/dart/preprocess/identify_stories.py
    '''

    categories = list(df['category'].unique())
    v = TfidfVectorizer(stop_words='english')
    threshold = 0.5
    cosines = []

    df['date'] = pd.to_datetime(df['date'])
    first_date = df.date.min() # datetime.strptime("2019-10-01", '%Y-%m-%d')
    last_date = df.date.max() # datetime.strptime("2019-12-07", '%Y-%m-%d')
    delta = last_date - first_date

    for i in range(delta.days+1):
        today = first_date + timedelta(days=i)
        yesterday = today - timedelta(days=1)
        past_three_days = today - timedelta(days=3)
            
        documents_3_days = df.loc[(df['date'] >= past_three_days) & (df['date'] <= today)] # self.handlers.articles.get_all_in_timerange(past_three_days, today)
        documents_1_day = df.loc[(df['date'] >= yesterday) & (df['date'] <= today)] # self.handlers.articles.get_all_in_timerange(yesterday, today)
            
        for category in categories:
            subset_3 = documents_3_days.loc[documents_3_days['category'] == category]
            subset_1 = documents_1_day.loc[documents_1_day['category'] == category]

            if not subset_1.empty and not subset_3.empty:

                subset_1_matrix = v.fit_transform(subset_1['text'].tolist())
                subset_3_matrix = v.fit_transform(subset_1['text'].tolist())

                cosine_similarities = cosine_similarity(subset_1_matrix, subset_3_matrix)

                for x, row in enumerate(cosine_similarities):
                    for y, cosine in enumerate(row):
                        if threshold <= cosine < 1:
                            x_id = list(subset_1.index.values)[x]
                            y_id = list(subset_3.index.values)[y]
                            cosines.append({'x': x_id, 'y': y_id, 'cosine': cosine})   
   

    cosine_df = pd.DataFrame(cosines)
    cosine_df = cosine_df.drop_duplicates()

    G = nx.from_pandas_edgelist(cosine_df, 'x', 'y', edge_attr=True)
    # create partitions, or stories
    partition = community_louvain.best_partition(G)
    stories = partition

    ## rewrite the code to return as series
    df['story'] = 0
    for k, v in stories.items():
        df.at[k, 'story'] = v
        
    return df
```

When input data with id, text, category and date is given, it will extended as follow

| id     | text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | date     | category | story |
|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|----------|-------|
| N55189 | "Wheel of Fortune," humorously introduced himself by stating he had been in a loveless marriage for 12 years to someone he jokingly referred to as an "old battle-ax named Kim" and …                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 10/15/19 | tv       | 1     |
| N46039 | In New Orleans, structural engineer Walter Zehner, upon hearing news of a devastating construction collapse at a site he previously worked on, …                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 10/15/19 | news     | 0     |
| N51741 | Actress Felicity Huffman reported to a federal prison in Northern California on Tuesday to start a 14-day sentence for her role in a massive college admissions scandal that rocked elite universities around the U.S. …                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 10/16/19 | tv       | 1     |
| N53234 | A recent storm on the Outer Banks unearthed an old shipwreck buried on the beach in Hatteras Island. A local bar shared photos of the old wooden ship in its final resting place in the area known as …                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 10/17/19 | news     | 0     |
| N11276 | Have a holly, jolly six-figure Christmas.As the countdown to the holiday season begins, Tiffany & Co. is offering branded advent calendars ..                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 10/24/19 | finance  | 4     |
| …      | …   | …        | …        | …     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

 
## How to Use


To use the data enhancement pipeline, follow these steps:

1. Clone the [repository](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/tree/test-workpackage/data_enhancement/customDataEnhancement) to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Prepare the input CSV file and metadata file (optional) in the input folder.
input csv should have following structure:  

| id     | text                                                                                                                                                                                                                     | date     |
|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| N51741 | Actress Felicity Huffman reported to a federal prison in Northern California on Tuesday to start a 14-day sentence for her role in a massive college admissions scandal that rocked elite universities around the U.S. … | 10/16/19 |
| N53234 | A recent storm on the Outer Banks unearthed an old shipwreck buried on the beach in Hatteras Island. A local bar shared photos of the old wooden ship in its final resting place in the area known as …                  | 10/17/19 |
| N11276 | Have a holly, jolly six-figure Christmas.As the countdown to the holiday season begins, Tiffany & Co. is offering branded advent calendars ..                                                                            | 10/24/19 |

The input CSV file should have the following columns:

* id : the unique identifier for each item (e.g. id of news article)
* text : the description of each item (e.g. original text of news article)
* date (optional) : the published date of item 

Note : 
- when enrich the category, and its metadata is available, id column is enough as input. 
- when enrich the story, id, text, date columns are required 

metadata file should have following structure: 
| id | category |
|----|----------|
| 1  | tv      |
| 2  | news       |
| 3  | finance     |

4. Open **config.py** and modify the settings according to your requirements.

attributes has all available extendable feature as key and each feature has nested dictionary with the 'enrich' key. user can mark as True if corresponding feature want to be extracted. 

```python
# Input and output file paths
input_file_path = './test/input/id_text_date.csv' # input file must be csv file 
output_file_path = './test/output/enhanced_test.json' # output file must be json file

# Define attributes for enriching features
attributes = {'category' : {'enrich' : False, # Example: True or False
			    'meta_file_path' : './test/input/news.csv', # If not exists, leave as empty string (e.g. '')
		            'candidate_labels': [] # Example: ["politics", "public health", "economics", "business", "sports", "life"]
		            },  
	      'sentiment' : {'enrich' : False},
	      'complexity' : {'enrich' : False},
	      'named_entities' : {'enrich' : True,
			  	  'entities' : ['PERSON', 'GPE', 'ORG'], # Example: ['PERSON', 'GPE', 'ORG', 'NORP', 'FAC', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME']
			  	  'extend_person_with_wiki': False,  # Example: True or False
			  	  'extend_organization_with_wiki': False # Example: True or False 
			  					   },
	      'region' : {'enrich' : True},
	      'political_party' : {'enrich' : False},
	      'min_maj_ratio': {'enrich' : False, 
				'major_gender' : ['male'], # It can be multiple values such as ['male', 'female']
				'major_citizen': ['United States of America'], # It can be multiple values such as ['France', 'Italy']
				'major_ethinicity': ['white people'], # It can be multiple values such as ['white people', 'hispanic', 'asian']
				'major_place_of_birth': ['United States of America']},
	      'story': {'enrich' : False}} 

```
* input_file_path: path to the input CSV file
* output_file_path: path to the output JSON file
* attributes : attributes has all available extendable feature as key and each feature has nested dictionary with the 'enrich' key. 'enrich' key with True value will be used to enhance the data and create new keys in the output JSON file that represent the corresponding attributes. Some feature has more attributes based on its implementation. For example, category takes extra variable either 'meta_file_path' or 'candidate_labels'. In case of named entities, user have flexibility to choose what entity to extract and option to extend data with wiki. For the min_maj_ratio, user can set the major gender, citizen, ethinicity, place of birth. 
  * category : since category can be extracted either using meta data or zero-shot classifier, if meta data exist, file path has to be given in 'meta_file_path'. if there is no available meta data, user should provide possible label list in 'candidate_labels'. 
  * named entity : Types of named entity to be extracted can be defined in 'entities'. If user mark 'extend_person_with_wiki' as True, person entity will be extended with wikidata information. Possible entity options are as following : 

    - PERSON: Represents names of people, including both first and last names.
    - ORG: Stands for organizations, companies, institutions, and other formal groups.
    - GPE: Denotes geopolitical entities such as countries, cities, states, and regions.
    - LOC: Represents non-geopolitical locations, landmarks, and other place names.
    - DATE: Indicates specific dates or date ranges.
    - TIME: Represents specific times or time intervals.
    - MONEY: Denotes monetary values, including currency symbols.
    - PERCENT: Represents percentage values.
    - FAC: Represents names of buildings, airports, highways, bridges, and similar facilities.
    - LANGUAGE: Denotes names of languages.
    - PRODUCT: Represents names of products, items, goods, and software.
    - EVENT: Represents names of events, conferences, festivals, and similar occurrences.
    - WORK_OF_ART: Denotes titles of books, songs, films, artworks, and other creative works.
    - LAW: Represents names of laws, regulations, and legal documents.
    - NORP: Denotes nationalities, religious or political groups, and similar entities.
    - QUANTITY: Represents measurements, quantities, or values with units.
    - CARDINAL: Represents cardinal numbers.
    - ORDINAL: Represents ordinal numbers (e.g., "first," "second").

  * min-maj_ratio : User need to define what gender, citizenship, ethinicity and place of birth will be considered as majority group. 

5. Run **main.py** using the command `python main.py`.
6. The output json file will be generated in the output folder. the output json file would nested dictionary with its id and extended features pair.
```
{
    "N51741": {
        "text": "Actress Felicity Huffman reported to a federal prison in Northern California on Tuesday to start a 14-day sentence for her role in a massive college admissions scandal that rocked elite universities around the U.S. …",
        "date": "10/16/19",
        "category": "tv",
        "sentiment": 0.6597,
        "complexity": 29.11679389,
        "named_entities": [
            {"text": "U.S.", "alternative": ["U.S."], "frequency": 1, "label": "GPE"},
            {"text": "Georgia", "alternative": ["Georgia"], "frequency": 1, "label": "GPE"},
            {"text": "Lori Loughlin", "alternative": ["Lori Loughlin", "Lori", "Loughlin"], "frequency": 4, "label": "PERSON"},
            {"text": "Mossimo Giannulli", "alternative": ["Mossimo Giannulli", "Giannulli", "Mossimo"], "frequency": 3, "label": "PERSON"},
            {"text": "Huffman", "alternative": ["Huffman"], "frequency": 2, "label": "PERSON"},
            {"text": "Us Weekly", "alternative": ["Us Weekly"], "frequency": 1, "label": "ORG"},
            {"text": "SAT", "alternative": ["SAT"], "frequency": 1, "label": "ORG"}
        ],
        "enriched_ne": {
            "Lori Loughlin": {
                "givenname": ["Lori"],
                "familyname": ["Loughlin"],
                "gender": ["female"],
                "occupations": ["television actor", "television producer", "actor", "film actor", "model", "voice actor"],
                "party": [],
                "positions": [],
                "citizen": ["United States of America"],
                "ethnicity": [],
                "sexuality": [],
                "place_of_birth": ["United States of America"],
                "key": "Lori Loughlin",
                "frequency": 4,
                "alternative": ["Lori Loughlin", "Lori", "Loughlin"]
            },
            "Mossimo Giannulli": {
                "givenname": ["Mossimo"],
                "familyname": ["Giannulli"],
                "gender": ["male"],
                "occupations": ["businessperson", "fashion designer"],
                "party": [],
                "positions": [],
                "citizen": ["United States of America"],
                "ethnicity": [],
                "sexuality": [],
                "place_of_birth": ["United States of America"],
                "key": "Mossimo Giannulli",
                "frequency": 3,
                "alternative": ["Mossimo Giannulli", "Giannulli", "Mossimo"]
            },
            "Bella": {
                "key": "Bella",
                "frequency": 2,
                "alternative": ["Bella"]
            },
            "William H. Macy": {
                "givenname": ["William"],
                "familyname": ["Macy"],
                "gender": ["male"],
                "occupations": ["screenwriter", "television actor", "character actor", "film producer", "theatrical director", "film director", "film actor", "stage actor", "actor", "theatre director", "merchant", "teacher", "writer"],
                "party": ["Democratic Party"],
                "positions": [],
                "citizen": ["United States of America"],
                "ethnicity": [],
                "sexuality": [],
                "place_of_birth": ["United States of America"],
                "key": "William H. Macy",
                "frequency": 1,
                "alternative": ["William H. Macy"]
            },
            "Huffman": {
                "key": "Huffman",
                "frequency": 2,
                "alternative": ["Huffman"]
            }
        },
        "region": ["U.A", "Georgia"],
        "party": {"Democratic Party": 1},
        "min_maj_ratio": {
            "gender": [0.5, 0.5],
            "ethnicity": [0.0, 1.0],
            "mainstream": [0.3333333333333333, 0.6666666666666666]
        },
        "story": 1
    },
    "N53234": {
        "text": "A recent storm on the Outer Banks unearthed an old shipwreck buried on the beach in Hatteras Island. A local bar shared photos of the old wooden ship in its final resting place in the area known as …",
        "date": "10/17/19",
        "category": "news",
        "sentiment": -0.9932,
        "complexity": 14.93154158,
        "named_entities": [
            {"text": "Demi Moore", "alternative": ["Demi Moore"], "frequency": 1, "label": "PERSON"},
            {"text": "Bruce Willis", "alternative": ["Bruce Willis", "Bruce"], "frequency": 3, "label": "PERSON"},
            {"text": "Jimmy Fallon", "alternative": ["Jimmy Fallon"], "frequency": 1, "label": "PERSON"},
            {"text": "Krista Smith", "alternative": ["Krista Smith"], "frequency": 1, "label": "PERSON"},
            {"text": "Netflix", "alternative": ["Netflix"], "frequency": 1, "label": "ORG"}
        ],
        "enriched_ne": {
            "Demi Moore": {
                "givenname": ["Demetria", "Gene", "Demi"],
                "familyname": ["Moore", "Guynes"],
                "gender": ["female"],
                "occupations": ["television actor", "film producer", "stage actor", "actor", "film director", "film actor", "model", "voice actor"],
                "party": ["Democratic Party"],
                "positions": [],
                "citizen": ["United States of America"],
                "ethnicity": [],
                "sexuality": [],
                "place_of_birth": ["United States of America"],
                "

```
 
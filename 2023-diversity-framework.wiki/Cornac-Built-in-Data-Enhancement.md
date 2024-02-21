<p> This wiki page provides an overview of how we enhanced the built-in dataset in Cornac. 
  
Cornac is a versatile library for building and evaluating recommender systems. It offers various built-in datasets that can be used for training and evaluating recommendation algorithms. However
Cornac's default recommender algorithms primarily focus on accuracy and rely solely on the user-item-rating tuple. In order to effectively measure and enhance diversity, we recognized the necessity of incorporating additional features that could represent the diversity of items . These extra features, such as category, genre, sentiment, and others, play a crucial role in computing diversity metrics.</p>

## Overview
<p> Our enhancement strategy involved two main steps:

1. Analyzing Existing Features:
- We thoroughly examined the available features in the existing Cornac dataset.
- For each feature, we assessed its suitability for utilization as a diversity algorithm.
- If a feature was found to be appropriate, we extracted it and incorporated it into the enhanced dataset.

2. Retrieving Metadata:
- In cases where the existing features were not suitable or insufficient, we turned to the metadata of the original data.
- We carefully analyzed the metadata to identify relevant information that could serve as valuable features.
- The useful metadata extracted from the original data was then incorporated into the enhanced dataset. </p>

By following this strategy, we aimed to expand the range of features available in the built-in dataset, enabling more effective diversity algorithms for recommendation systems.


## summary
| Cornac built-in Data | original/meta data link                                                                             | available features in Cornac            | enhanced features          |
|----------------------|------------------------------------------------------------|----------------------------------------------|----------------------------|
| Amazon clothing      | https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/                                     | item text description | category, brand            |
| Amazon digital music | https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/                                     | user-item-review                        | sentiment                  |
| Amazon office        | https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/                                     | user-item-rating                        |                            |
| Amazon toy and games | https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/                                     | user-item-rating                        | category, brand, sentiment |
| citeUlike            | http://www.wanghao.in/CDL.htm                                                             | user-item-rating                        |                            |
| Epinions             | http://www.trustlet.org/downloaded_epinions.html                                          | user-item-rating                        |                            |
| FilmTrust            | https://www.librec.net/datasets.html                                                      | user-item-rating                        |                            |
| Movielens            | https://grouplens.org/datasets/movielens/                                                 | user-item-rating                        | genre                      |
| Netflix              | https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data?select=combined_data_2.txt | user-item-rating                        |                            |
| Tradesy              | http://jmcauley.ucsd.edu/data/tradesy/                                                    | user-item-rating                        | item category              |


## [Amazon Clothing](https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_clothing)
In Cornac, Amazon clothing dataset provide user-item-rating, item-item interaction and item text description. In the future, item text description can be used to extract more features (e.g. category, named entity) however since external metadata is available [(link)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), which includes category and brand name, we extended this dataset by simple matching with these metadata. 

## [Amazon Digital Music]()
`currently this dataset is not loadable. there might be an bug in Cornac`
Amazon Digital Music dataset provides user-item-rating and user-item-review. We can extract sentiment from user-item-review feature. 

## [Amazon office]()
Amazon office dataset provides user-item-rating and item-item interaction. So the only way to extend this dataset is retreiving some useful feature from metadata. However item id in Cornac dataset and metadata does not match. So potentially there is no way to extend this dataset.

## [Amazon toy and games]()
Amazon toy and games dataset provides user-item-rating and user-item-sentiment. we can directly use this sentiment value and additionally extract the category and brand feature from metadata. 

## [citeUlike]()

## [Epinions]()

## [FilmTrust]()

## [Movielens]()

## [Netflix]()

## [Tradesy]()



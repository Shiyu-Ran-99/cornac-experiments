# 1. Cornac Built-in Datasets Overview

| Built-in Datasets     | available features                                                                                                  | link to original/meta data                                                                | useful feature from meta data | metrics                                                                                                                                     | notes                                                                                                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Amazon clothing](https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_clothing)       | .load_text()                                                                                   | [link](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)                                     | category, brand               | calibration, binomial diversity, Gini coefficient, intra-list diversity, expected intra-list diversity                                      |                                                                                                                                                                       |
| [Amazon digital music](https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_digital_music)  |                                                                                                                     | [link](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)                                                                                          | category                      | calibration, binomial diversity, Gini coefficient, intra-list diversity, expected intra-list diversity                                      |                                                                                                                                                                       |
| [Amazon office](https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_office)         |                                                                                                                     | [link](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)                                                                                          | category                      | calibration, binomial diversity, Gini coefficient, intra-list diversity, expected intra-list diversity                                      | item id is not matching                                                                                                                                               |
| [Amazon toy and games](https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_toy)  | .load_sentiment() |                                                                                           | category, brand               | calibration, binomial diversity, Gini coefficient, intra-list diversity, expected intra-list diversity, activation                          |                                                                                                                                                                       |
| [citeUlike](https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.citeulike)             | .load_text()                  | [link](http://www.wanghao.in/data/ctrsr_datasets.rar)                                                             |  | calibration (category, complexity), binomial diversity, Gini coefficient, intra-list diversity, expected intra-list diversity, activation,  | since original text is available from cornac, we can extract pretty much everything like news articles : complexity, tone, category  |
| [Epinions](https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.epinions)              |                                                                                                                     | [link](http://www.trustlet.org/downloaded_epinions.html)                                         |                               |                                                                                                                                             |                                                                                                                                                                       |
| [FilmTrust](https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.filmtrust)             |                                                                                                                     | [link](https://www.librec.net/datasets.html)                                                      |                               |                                                                                                                                             | link does not work                                                                                                                                                    |
| [Movielens](https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.movielens)             |       .load_plot()                                                                                                              | [link](https://grouplens.org/datasets/movielens/)                                                 | movie title, genre            | calibration, binomial diversity, Gini coefficient, intra-list diversity, expected intra-list diversity                                      | movie title can be queried to wikidata and retreve useful feature                                                                                                     |
| [Netflix](https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.netflix)               |                                                                                                                     | [link](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data?select=combined_data_2.txt) | movie title                   | calibration, binomial diversity, Gini coefficient, intra-list diversity, expected intra-list diversity                                      | movie title can be queried to wikidata and retreve useful feature                                                                                                     |
| [Tradesy](https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.tradesy)               |                                                                                                                     | [link](http://jmcauley.ucsd.edu/data/tradesy/)                                                    | item category                 | calibration, binomial diversity, Gini coefficient, intra-list diversity, expected intra-list diversity                                      |                                                                                                                                                                       |


# 2. Potential Usage 
### 1. Amazon Review

This dataset contains product reviews and metadata from Amazon. It includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs) as separate files.

Users have the freedom to manipulate available attributes, and one of the easiest use cases is retrieving category or brand as a categorical feature as shown in the following Python code. This json file can be loaded in our dataloader function and can be used as diversity metrics such as  calibration metrics, : 

```python
import cornac
import pandas as pd
import json

feedback = cornac.datasets.amazon_clothing.load_feedback()
feedback_df = pd.DataFrame(feedback)
feedback_df.rename(columns = {'0':'user', '1':'item', '2':'rating'}, inplace = True)

# unique item id
items = feedback_df['item'].unique()

# load the Amazon data 
path = '../dataset/Amazonm_Clothing_metadata.json'

chunks = pd.read_json(path, lines=True, chunksize = 10000)

dic = {}

for chunk in chunks:  
    for item in items:
        if item in dic.keys():
            continue
        else:
            cat = chunk[chunk['asin']==item]['brand']
            if not cat.empty :
                dic[item] = cat.values[:1][0]

with open('brand.json', 'w') as fp:
    json.dump(dic, fp)

```



#### 1.1. Amazon clothing 

**Sample metadata**
```
{"category": ["Clothing, Shoes & Jewelry", "Costumes & Accessories", "Kids & Baby", "Girls", "Accessories", "3 layers of tulle", "6\" long, stretched waist measures 11 1/2\" across. Fits up to 7 years.", "Sequins line the edge of the tulle on the top layer.", "Great for babys up to about age 7", "Makes a Great gift for any princess"], 
"description": ["6\" long, stretched waist measures 11 1/2\" across. Fits up to 7 years."], "title": "Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory", 
"brand": "Big Dreams", 
"feature": ["3 layers of tulle", "6\" long, stretched waist measures 11 1/2\" across. Fits up to 7 years.", "Sequins line the edge of the tulle on the top layer.", "Great for babys up to about age 7", "Makes a Great gift for any princess"], 
"rank": "19,963,069inClothing,ShoesJewelry(", 
"date": "5 star5 star (0%)", 
"asin": "0000037214"}
```

brand.json, category.json is available in teams 

#### 1.2. Amazon digital music 

**Sample metadata**
```
{"category": [], 
 "tech1": "", 
 "description": [], 
 "fit": "", 
 "title": "Master Collection Volume One", 
 "also_buy": ["B000002UEN", "B000008LD5", "B01J804JKE", "7474034352", "B004ZLBTXW", "B000008LDH",], 
"tech2": "", 
"brand": "John Michael Talbot", 
"feature": [], 
"rank": "58,291 in CDs & Vinyl (", 
"also_view": ["B000002UEN", "B000008LD5", "7474034352", ], 
"main_cat": "<img src=\"https://images-na.ssl-images-amazon.com/images/G/01/digital/music/logos/amzn_music_logo_subnav._CB471835632_.png\" class=\"nav-categ-image\" alt=\"Digital Music\"/>", 
"similar_item": "", 
"date": "", 
"price": "$18.99", 
"asin": "0001377647", 
"imageURL": [], 
"imageURLHighRes": []}
```


#### 1.3. Amazon office 
**Sample metadata**
```
{"category": ["Office Products", "Office &amp; School Supplies", "Calendars, Planners &amp; Personal Organizers"], 
"tech1": "", 
"description": ["Paulo Coelho, regarded by millions as an alchemist of words, is one of this centurys most influential writers. His books not only make it to the top of the bestseller lists, they also provoke social and cultural debate. He deals with subjects, ideas and philosophies that touch the aspirations of those many readers who are in search of their own path and of new ways of understanding the world.", "", ""], 
"fit": "", 
"title": "Paulo Coelho Moments 2012 Day Planner", 
"also_buy": ["0525435077"], 
"tech2": "", 
"brand": "Visit Amazon's Paulo Coelho Page", 
"feature": [], 
"rank": "3,872,855 in Books (", 
"also_view": ["1101972645"], 
"main_cat": "Books", 
"similar_item": "", 
"date": "", 
"price": "", 
"asin": "030794655X", 
"imageURL": [], 
"imageURLHighRes": []}

```

#### 1.4. Amazon toy and games 
**Sample metadata**
```
{"category": ["Toys & Games", "Puzzles", "Jigsaw Puzzles"], 
"tech1": "", 
"description": ["Three Dr. Suess' Puzzles: Green Eggs and Ham, Favorite Friends, and One Fish Two Fish Red Fish Blue Fish"], 
"fit": "", 
"title": "Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle", 
"also_buy": [], "tech2": "", 
"brand": "Dr. Seuss", 
"feature": ["Three giant floor puzzles", "Includes: Dr. Suess Green Eggs and Ham, Favorite Friends, and One Fish Two Fish Blue Fish", "Each puzzle has 48 pieces", "Ages 3 and up"], 
"rank": [">#2,230,717 in Toys & Games (See Top 100 in Toys & Games)", ">#57,419 in Toys & Games > Puzzles > Jigsaw Puzzles"], 
"also_view": [], 
"main_cat": "Toys & Games", 
"similar_item": "", 
"date": "", 
"price": "", 
"asin": "0000191639", 
"imageURL": ["https://images-na.ssl-images-amazon.com/images/I/51rn8TxbcoL._SS40_.jpg", "https://images-na.ssl-images-amazon.com/images/I/51j1Fep1niL._SS40_.jpg"], 
"imageURLHighRes": ["https://images-na.ssl-images-amazon.com/images/I/51rn8TxbcoL.jpg", "https://images-na.ssl-images-amazon.com/images/I/51j1Fep1niL.jpg"]}
```
brand.json, category.json is available in teams

### 2. CiteUlike 
CiteULike-A was utilized in the IJCAI paper 'Collaborative Topic Regression with Social Regularization' by Wang, Chen, and Li. This dataset was sourced from CiteULike and Google Scholar, and it comprises abstracts, titles, and tags for individual articles and enables users to curate their collections of articles.  

Since this dataset includes textual data in the 'raw.abstract' field, one potential use case would be to run our data enhancement pipeline with the appropriate input data format (e.g., id-text or id-text-date CSV file). This would make most features available, such as category, complexity, sentiment, and story (note: publication date is required for the story).


**sample data** 
| doc.id | title                                       | citeulike.id | raw.title                                 | raw.abstract                                                                                                                                                                                                                   |
|--------|---------------------------------------------|--------------|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1      | The metabolic world of Escherichia coli is not small | 42           | The metabolic world of Escherichia coli is not small | To elucidate the organizational and evolutionary principles of the metabolism of living organisms, recent studies have addressed the graph-theoretic analysis of large biochemical networks responsible for the synthesis and degradation of cellular building blocks [Jeong, H., Tombor, B., Albert, R., Oltvai, Z. N. & Barabási, A. L. (2000) Nature 407, 651-654; Wagner, A. & Fell, D. A. (2001) Proc. R. Soc. London Ser. B 268, 1803-1810; and Ma, H.-W. & Zeng, A.-P. (2003) Bioinformatics 19, 270-277]. ... |


### 3. Epinions 

The Epinions dataset is built form a who-trust-whom online social network of a general consumer review site Epinions.com. Members of the site can decide whether to ''trust'' each other. All the trust relationships interact and form the Web of Trust which is then combined with review ratings to determine which reviews are shown to the user. It contains 75,879 nodes and 50,8837 edges.

The dataset is a directed graph with 'FromNodeId' and 'ToNodeId,' and users can come up with new metrics that utilize user-user trust information.


**sample data**   
| FromNodeId | ToNodeId |
|------------|----------|
| 0          | 4        |
| 0          | 5        |
| ...        | ...      |



### 4. FilmTrust 

FilmTrust is a website that integrates social networks with movie ratings and reviews. Using FOAF-based social networks augmented with trust ratings, the site computes predictive movie ratings based on the ratings of trusted people in the network. 
  
user-user trust dataset is downloadable from [here](https://guoguibing.github.io/librec/datasets/filmtrust.zip). 

**sample data**  
| user-id (trustor) | user-id (trustee) | trust-value |
|-------------------|-------------------|-------------|
| 2                 | 104               |      1      |
| 5                 | 1509              |      1      |


### 5. Movielens 

GroupLens Research has collected and made available rating data sets from the MovieLens web site (https://movielens.org). The data sets were collected over various periods of time, depending on the size of the set. Cornac has 4 different variation of Movielens dataset : [‘100K’, ‘1M’, ‘10M’, ‘20M’]. 

Original Movielens dataset consists of several files. 

`tags.csv`  
 Each line of this file after the header row represents one tag applied to one movie by one user, and has the following format:

| userId | movieId | tag               | timestamp   |
|--------|---------|-------------------|-------------|
| 3      | 260     | classic           | 1439472355  |
| 4      | 1732    | dark comedy       | 1573943598  |
| 4      | 1732    | great dialogue    | 1573943604  |
| 4      | 7569    | so bad it's good  | 1573943455  |


Tags are user-generated metadata about movies. Each tag is typically a single word or short phrase. The meaning, value, and purpose of a particular tag is determined by each user.

Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

`movies.csv`    
It includes genre information in a pipe-separated list, and all genres are as follows: Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western, and (no genres listed).

| movieId | title                      | genres                                  |
|---------|----------------------------|----------------------------------------|
| 1       | Toy Story (1995)           | Adventure|Animation|Children|Comedy|Fantasy |
| 2       | Jumanji (1995)             | Adventure|Children|Fantasy            |
| 3       | Grumpier Old Men (1995)    | Comedy|Romance                         |


Users can use this file as metadata to extract categories using our pipeline. Furthermore, users may send a title as a SPARQL query to Wikidata to retrieve interesting information and use it as features.

`links.csv`  
It contains identifiers that can be used to link to other sources of movie data.

| movieId | imdbId | tmdbId |
|---------|--------|--------|
| 1       | 114709 | 862    |
| 2       | 113497 | 8844   |
| 3       | 113228 | 15602  |
| 4       | 114885 | 31357  |

- movieId is an identifier for movies used by https://movielens.org. E.g., the movie Toy Story has the link https://movielens.org/movies/1.
- imdbId is an identifier for movies used by http://www.imdb.com. E.g., the movie Toy Story has the link http://www.imdb.com/title/tt0114709/.
- tmdbId is an identifier for movies used by https://www.themoviedb.org. E.g., the movie Toy Story has the link https://www.themoviedb.org/movie/862.

User can scrap useful information from each website and use as new features. 



### 6. Netflix 
Netflix dataset has metadata called `movie_titles.txt` and it contains year of release range from 1890 to 2005 and movie title.

**sample data**  
| movieID | YearOfRelease   | Title                     |
|---------|-----------------|---------------------------|
| 3167    | 1987            | Evil Dead 2: Dead by Dawn |
| ...     | ...             | ...                       |

Users can send a title as a SPARQL query to Wikidata to retrieve interesting information and use it as features.


### 7. Tradesy

These datasets contain peer-to-peer trades from various recommendation platforms.

**sample data**  
```
{
  'lists':
  {
    'bought': ['466', '459', '457', '449'],
    'selling': [],
    'want': [],
    'sold': ['104', '103', '102']
  },
  'uid': '2'
}
```

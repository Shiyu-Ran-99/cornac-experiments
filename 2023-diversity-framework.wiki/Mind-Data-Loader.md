## Description
The functions detailed in the subsequent sections are available within the **'cornac/datasets/mind.py'** module. The aim is to transform externally enhanced data into the requisite format required by the diversity framework.
Prior to employing functions, it is strongly advised to review the structure and content of the provided data file. 

## Functions
**load_feedback**  
This function is designed to handle rating data loading. The necessary format involves a CSV file that includes three essential columns: user, item, and rating. These columns must follow a specific order: user first, followed by item, and then rating. However, if the CSV file contains an index and consequently has four columns, the load_feedback function will exclude the first column. The output is a list of tuples containing all user-item-rating pairs.  

**load_sentiment**  
This function loads sentiment data associated with items. The requisite data structure can take the form of a JSON file (recommended) or a CSV file.  

<!-- TODO:
- [ ] maybe say in words that the expected format is item_id: sentiment_value; this is mentioned for subsequent enrichment values, but it would look nice to have a consistent way of describing them;   -->

For JSON files, the expected format is {item id: item sentiment_value}. For example,     
```
{
    "N55189": 0.6597,
    "N46039": -0.9932,
    "N51741": -0.4344
}
```
In this structure, the keys represent the raw IDs of the items, while the corresponding values denote the sentiment values attributed to the articles.   

Alternatively, if opting for a CSV file, the first column should contain the item IDs, and the second column contains the corresponding sentiment values.  

The output of this function is a dictionary containing item and item sentiment.   


**load_category**  
The purpose of this function is to load item categories into a dictionary. It's important to emphasize that this function pertains to scenarios where each item is linked to a single category. The input can be provided in either a JSON format (recommended) or a CSV format.

When using a JSON file as input, the expected format is {item id: item category}. For example,    
```
{
  "N11276": "finance",
    "N264": "autos",
    "N40716": "tv",
    "N28088": "movies",
    "N43955": "entertainment"
}
```
In this format, the keys correspond to raw item ids, while the values denote the respective category each item belongs to.  
Alternatively, if opting for a CSV file as input, the first column should contain the item raw IDs, and the second column contains the corresponding category for each item.  

The output of this function is a dictionary containing item and item category infomation.  

**load_category_multi**  
This function is different with the **load_category** function.  Firstly, it can accommodate either a single category or multiple categories assigned to an item. Secondly, the resulting output from this function is a dictionary that pairs items with an encoding signifying the categories associated with them. 
The input for this function can be provided in either JSON format (recommended) or CSV format.  
When employing a JSON file as input, the expected format is {item id: item categories}. For example,  
```
{
    "N55528": ["lifestyle", "health"], 
    "N18955": ["health", "sports"],
    "N61837": ["news","weather"], 
    "N53526": "health",
    "N38324": ["health","food"], 
    "N2073": "sports"
}
```
In this case, there are six categories—namely, "lifestyle", "health", "sports", "news","weather" and "food"—then the item "N2073" belonging solely to the "sports" category would be represented as [0, 0, 1, 0, 0, 0]. Similarly, the item "N55528" associated with both the "lifestyle" and "health" categories would yield an output of [1, 1, 0, 0, 0, 0].  

For items designated with a single category, the JSON dictionary should adhere to the following format: `{"N2073":"sports"}`. Conversely, items associated with multiple categories should follow a structure as a list: `{"N55528": ["lifestyle","health"]}`.

If users opt to use CSV as the input format, the first column should list the item raw IDs, while the second column can house either a single category or a comma-separated list of multiple categories.


**load_complexity**  
This function is designed to load item complexity. It allows for either JSON (recommended) or CSV files.  
When using a JSON file as input, the expected format is {item id: item complexity}. For example,  
```
{
    "N55189": 29.1167938931,
    "N46039": 14.9315415822,
    "N51741": 33.1415942029,
    "N53234": 12.4489795918
}
```
When using json file as input, this function will exclude items with NaN (Not a Number) complexity values, delivering a dictionary containing only items with valid complexity values.
For users choosing to use a csv file as input, ensure that the first column holds the item raw ID, while the second column corresponds to the respective item complexity. Prior to use, please review the content to verify that the complexity values are valid and can be successfully converted to numeric values in Python.


**load_story**  
This function aims to load item story data into a dictionary, with item raw IDs serving as keys and their respective stories as values. It accommodates input in either JSON format (the recommended choice) or CSV format.  
When utilizing a JSON file as input, the expected format is {item id: item story}. For example,  
```
{
    "N55189": 458.0,
    "N46039": 0.0,
    "N51741": 458.0,
    "N53234": 397.0
}
```
For users opting for CSV input, please ensure that the first column contains the item's raw ID, while the second column corresponds to the respective item story. Before proceeding, it's advisable to thoroughly examine the content to ensure that the story values are valid and can be effectively converted into integer values using Python.


**load_entities**  
This function is designed to compile item parties information into a dictionary, and it offers the flexibility of receiving input in either JSON format (recommended) or as a CSV file.  

In the case of JSON input, the expected format is {item id: {party: frequency}}. For example,  
```
{
        "N38895": {
        "Democratic Party": 4
    },
    "N30924": {},
    "N58251": {
        "Republican Party": 2,
        "Federalist Party": 2,
        "Democratic Party": 2
    },
}
```
Alternatively, for users opting to provide input via a CSV file, the first column should contain the item raw IDs, and the second column should list all mentions of parties separated by commas. For example:  

| Item    | Entities |
| -------- | ------- |
| N38895  | "Democratic Party,Democratic Party,Democratic Party,Democratic Party"   |
| N58251  | "Republican Party,Republican Party,Federalist Party,Federalist Party,Democratic Party,Democratic Party" |     


In the case of JSON input, the function will filter out items that have empty parties information. when using a CSV file as input, it is imperative to thoroughly review and confirm that the file exclusively contains items with valid party information.
The output from this function is a dictionary with item raw IDs as keys and a list of mentioned parties as values. In the examples, the output would appear as follows:
```
{
    "N38895": ["Democratic Party", "Democratic Party", "Democratic Party", "Democratic Party"],
    "N58251": ["Republican Party", "Republican Party", "Federalist Party", "Federalist Party", "Democratic Party", "Democratic Party"]
}
```

<!-- TODO
- [ ] for the minority / majority scores it not clear what values are considered; for instance, does 0.0 refer to male or female? also, ethnicity can have multiple values, why binary? and if binary, which are they? it is ok if not all possible values are taken into account, but the description here should be a bit more clear so that users understand how the input is supposed to look like;   -->


**load_min_maj**  
This function manages the enriched minority score and majority score features. The minority score and majority score are specifically designed to determine whether viewpoint holders belong to a "protected group" or not. In our [data enhancement pipeline](Custom-Data-Enhancement.md), users have the flexibility to define how gender, ethnicity, and mainstream are categorized as the majority.  

Within this data loading function, users can define the "data_type" parameter from their JSON file. For example, if the file encompasses minority and majority scores pertaining to gender, ethnicity, and mainstream, users can specify "data_type" as either "gender," "ethnicity," or "mainstream." The default selection is "mainstream".    
 
The expected json format is {item id: {data_type:[minority score, majority score]}}. For example,  
```
{
    "N55189": {
        "gender": [
            0.0,
            1.0
        ],
        "ethnicity": [
            0.0,
            1.0
        ],
        "mainstream": [
            0.9412,
            0.0588
        ]
    },
    "N46039": {
        "gender": [
            0.0,
            1.0
        ],
        "ethnicity": [
            1.0,
            0.0
        ],
        "mainstream": [
            0.9333,
            0.06670000000000001
        ]
    }
}
```
For instance, in the provided example, when evaluating the gender of the news item "N55189", it is considered that males represent the majority voice, while others are categorized as the minority voice. Since males are frequently mentioned in the news, the gender minority score is 0, and the majority score is 1. In the same example, the majority voice from an ethnicity perspective is defined as individuals with a 'United States' ethnicity or place of birth, with all others being part of the minority. If all individuals mentioned in the news belong to this group, the majority score is set to 1. The determination of the mainstream category in the same example is based on whether a person has Wikidata information, signifying their public recognition. Individuals who cannot be found in Wikidata are considered part of the minority group.    

If users opt to use CSV as input, it's essential to ensure that the first column contains the item ID, the second column contains the item's minority score, and the third column contains the majority score.   
The output of this function is a dictionary with item raw IDs serving as keys. The associated values are numpy arrays, where the minority score is at position 0 and the majority score is at position 1.  


**load_text**  
This function is responsible for retrieving text data associated with each item. Users have the option to utilize either JSON (the recommended choice) or CSV files as input. 
When choosing JSON input, the expected format is {item id: text}. For example,  
```
{
    "N55189": "text",
     "N46039": "text"
}
```
For CSV input, ensure that the first column contains the item's raw ID, while the second column holds the corresponding text.  
The result is a dictionary where item raw IDs serve as keys, and the corresponding item text is the associated value. It is strongly advised to review the data file prior to utilizing this function.

**build**  
This function is responsible for transforming external Item IDs into Cornac's internal IDs.
To utilize it, you provide the dictionary obtained after executing the earlier data loading functions, along with the "id_map" that you acquire after inputting user-item-rating data into the Cornac system. The "id_map" serves as a bridge, containing mapping from item original(raw)ids to mapped Cornac's integer indices.
After executing this "build" function, an output dictionary is generated. In this dictionary, Cornac's internal Item IDs serve as keys, and the corresponding features act as values. This output can be used when you initialize diversity metrics, facilitating further processing within the diversity framework.  


## Data loading pipeline

```
feedback = mind.load_feedback(
    fpath="./tests/enriched_data/mind_uir.csv")
sentiment = mind.load_sentiment(
    fpath="./tests/enriched_data/sentiment.json")
category = mind.load_category(
    fpath="./tests/enriched_data/category.json")
complexity = mind.load_complexity(
    fpath="./tests/enriched_data/complexity.json")
story = mind.load_story(fpath="./tests/enriched_data/story.json")
entities = mind.load_entities(fpath="./tests/enriched_data/party.json")
genre = mind.load_category_multi(
    fpath="./tests/enriched_data/category.json")
min_maj = mind.load_min_maj(fpath="./tests/enriched_data/min_maj.json")

rs = RatioSplit(
    data=feedback,
    test_size=0.2,
    rating_threshold=0.5,
    seed=123,
    exclude_unknowns=True,
    verbose=True,
)

Item_sentiment = mind.build(
    data=sentiment, id_map=rs.train_set.iid_map)
Item_category = mind.build(data=category, id_map=rs.train_set.iid_map)
Item_complexity = mind.build(
    data=complexity, id_map=rs.train_set.iid_map)
Item_stories = mind.build(data=story, id_map=rs.train_set.iid_map)
Item_genre = mind.build(data=genre, id_map=rs.train_set.iid_map)
Item_entities = mind.build(data=entities, id_map=rs.train_set.iid_map)
Item_min_major = mind.build(data=min_maj, id_map=rs.train_set.iid_map)
Item_feature = Item_genre
act = Activation(item_sentiment=Item_sentiment, k=200)
cal = Calibration(item_feature=Item_category,
                    data_type="category", k=200)
cal_complexity = Calibration(
    item_feature=Item_complexity, data_type="complexity", k=200)
bino = Binomial(item_genre=Item_genre, k=200)
alt = AlternativeVoices(item_minor_major=Item_min_major,k=200)
repre = Representation(item_entities=Item_entities,k=200)
frag = Fragmentation(item_story=Item_stories, n_samples=1, k=200)
ild = ILD(item_feature=Item_feature, k=200)
ndcg = NDCG_score(k=200)
eild = EILD(item_feature=Item_feature, k=200)
gini = GiniCoeff(item_genre=Item_genre, k=200)
Experiment(eval_method=rs,
            models=[UserKNN(k=3, similarity="pearson",
                            name="UserKNN")],
            metrics=[act, cal, cal_complexity, bino,
                    ndcg, gini, frag, ild, eild, repre, alt],
            verbose=True,
            ).run()
```
Users are required to enhance their data and subsequently utilize the generated feature file. This feature file should be loaded and mapped to Cornac's internal item IDs. Then, users can initialize diversity metrics using the resulting dictionary.  
By following this pipeline, enriched data can be loaded, and the diversity framework can be executed.

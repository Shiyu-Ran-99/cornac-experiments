import argparse
import cornac
from cornac.eval_methods import RatioSplit
from cornac.metrics import MAE, RMSE, MSE, FMeasure, Precision, Recall, NDCG, NCRR, MRR, AUC, MAP
from cornac.datasets import mind as mind
from cornac.metrics import NDCG_score
from cornac.metrics import GiniCoeff
from cornac.metrics import ILD
from cornac.metrics import EILD
from cornac.metrics import Activation
from cornac.metrics import Calibration
from cornac.metrics import Fragmentation
from cornac.metrics import Representation
from cornac.metrics import AlternativeVoices
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer

feedback = mind.load_feedback(fpath="./tests/enriched_data/mind_uir_20k.csv")
sentiment = mind.load_sentiment(fpath="./tests/enriched_data/sentiment.json")
category = mind.load_category(fpath="./tests/enriched_data/category.json")
complexity = mind.load_complexity(
    fpath="./tests/enriched_data/complexity.json")
story = mind.load_story(fpath="./tests/enriched_data/story.json")
genre = mind.load_category_multi(fpath="./tests/enriched_data/category.json")
entities = mind.load_entities(fpath="./tests/enriched_data/party.json")
min_maj = mind.load_min_maj(fpath="./tests/enriched_data/min_maj.json")

text_dict = mind.load_text(fpath="./tests/enriched_data/text.json")
text = list(text_dict.values())
item_ids = list(text_dict.keys())
item_text_modality = TextModality(
    corpus=text,
    ids=item_ids,
    tokenizer=BaseTokenizer(sep=" ", stop_words="english"),
    max_vocab=8000,
    max_doc_freq=0.5,
)
mind_ratio_split = RatioSplit(
    data=feedback,
    test_size=0.2,
    exclude_unknowns=True,
    item_text=item_text_modality,
    verbose=True,
    seed=123,
    rating_threshold=0.5,
)
Item_sentiment = mind.build(
    data=sentiment, id_map=mind_ratio_split.train_set.iid_map)
Item_category = mind.build(
    data=category, id_map=mind_ratio_split.train_set.iid_map)
Item_complexity = mind.build(
    data=complexity, id_map=mind_ratio_split.train_set.iid_map)
Item_stories = mind.build(
    data=story, id_map=mind_ratio_split.train_set.iid_map)
Item_entities = mind.build(
    data=entities, id_map=mind_ratio_split.train_set.iid_map)
Item_min_major = mind.build(
    data=min_maj, id_map=mind_ratio_split.train_set.iid_map)
Item_genre = mind.build(data=genre, id_map=mind_ratio_split.train_set.iid_map)
Item_feature = Item_genre

diversity_objective_dict = {}
diversity_objective_dict['activation'] = Activation(item_sentiment=Item_sentiment, divergence_type='JS')
diversity_objective_dict['calibration_category'] = Calibration(item_feature=Item_category,
                       data_type="category", divergence_type='JS')
diversity_objective_dict['calibration_complexity'] = Calibration(item_feature=Item_category,
                       data_type="complexity", divergence_type='JS')
diversity_objective_dict['fragmentation'] = Fragmentation(item_story=Item_stories, n_samples=1,
                         divergence_type='JS')
diversity_objective_dict['ild'] = ILD(item_feature=Item_feature)
diversity_objective_dict['ndcg'] = NDCG_score()
diversity_objective_dict['eild'] = EILD(item_feature=Item_feature)
diversity_objective_dict['alternative_voices'] = AlternativeVoices(item_minor_major=Item_min_major, divergence_type='JS')
diversity_objective_dict['representation'] = Representation(item_entities=Item_entities, divergence_type='JS')
diversity_objective_dict['ginicoeff'] = GiniCoeff(item_genre=Item_genre)

parser = argparse.ArgumentParser(
    description='Benchmarking for the Cornac Algorithms')
parser.add_argument('--model', type=str, default='CTR',
                    help=' the name of the recommender model')
parser.add_argument('--topk', type=int, default=100,
                    help='the number of items in the top@k list')
parser.add_argument('--rerank', type=int, default=10,
                    help='the top@k items recommended to be reranked')
parser.add_argument('--k', type=int, default=50,
                    help='the candidate item list where the reranking items will be chosen from')
parser.add_argument('--lambda_constant', type=float, default=0,
                    help='weight factor of the diversity metrics')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--CTR_k', type=int, default=400,
                    help='The dimension of the latent factors in CTR model')
parser.add_argument('--CTR_lambda_v', type=float, default=0.01,
                    help='The regularization parameter for items in CTR model')
parser.add_argument('--CTR_lambda_u', type=float, default=0.01,
                    help='The regularization parameter for users in CTR model')
parser.add_argument('--HFT_k', type=int, default=10,
                    help='The dimension of the latent factors in HFT model')
parser.add_argument('--ConvMF_lambda_v', type=float, default=0.01,
                    help='The regularization hyper-parameter for item latent factor in ConvMF model')
parser.add_argument('--ConvMF_lambda_u', type=float, default=0.01,
                    help='The regularization hyper-parameter for user latent factor in ConvMF model')
parser.add_argument('--DAE_dims', type=int, default=200,
                    help='The dimension of autoencoder layer in DAE model')
parser.add_argument('--ENMF_neg_weight', type=float, default=0.1,
                    help='Negative weight in ENMF model')
parser.add_argument('--diversity_objective', type=str, default='ild',
                    help='metric name for diversity objective')
args = parser.parse_args()

metrics = [MAE(), RMSE(), MSE(), MRR(), AUC(), MAP(),
           FMeasure(k=args.topk), Precision(k=args.topk),
           Recall(k=args.topk), NDCG(k=args.topk), NCRR(k=args.topk),

           FMeasure(k=args.rerank), Precision(k=args.rerank),
           Recall(k=args.rerank), NDCG(k=args.rerank), NCRR(k=args.rerank),

           Activation(item_sentiment=Item_sentiment,
                      divergence_type='JS', k=args.topk),
           Calibration(item_feature=Item_category,
                       data_type="category", divergence_type='JS', k=args.topk),
           Calibration(item_feature=Item_complexity,
                       data_type="complexity", divergence_type='JS', k=args.topk),
           Fragmentation(item_story=Item_stories, n_samples=1,
                         divergence_type='JS', k=args.topk),
           ILD(item_feature=Item_feature, k=args.topk),
           NDCG_score(k=args.topk),
           EILD(item_feature=Item_feature, k=args.topk),
           GiniCoeff(item_genre=Item_genre, k=args.topk),
           AlternativeVoices(item_minor_major=Item_min_major, divergence_type='JS', k=args.topk),
           Representation(item_entities=Item_entities, divergence_type='JS', k=args.topk),

           Activation(item_sentiment=Item_sentiment,
                      divergence_type='JS', k=args.rerank),
           Calibration(item_feature=Item_category, data_type="category",
                       divergence_type='JS', k=args.rerank),
           Calibration(item_feature=Item_complexity,
                       data_type="complexity", divergence_type='JS', k=args.rerank),
           Fragmentation(item_story=Item_stories, n_samples=1,
                         divergence_type='JS', k=args.rerank),
           ILD(item_feature=Item_feature, k=args.rerank),
           NDCG_score(k=args.rerank),
           EILD(item_feature=Item_feature, k=args.rerank),
           GiniCoeff(item_genre=Item_genre, k=args.rerank),
           AlternativeVoices(item_minor_major=Item_min_major, divergence_type='JS', k=args.rerank),
           Representation(item_entities=Item_entities, divergence_type='JS', k=args.rerank),
           ]

reranking_metrics = [
           FMeasure(k=args.rerank), Precision(k=args.rerank),
           Recall(k=args.rerank), NDCG(k=args.rerank), NCRR(k=args.rerank),

           Activation(item_sentiment=Item_sentiment,
                      divergence_type='JS', k=args.rerank),
           Calibration(item_feature=Item_category, data_type="category",
                       divergence_type='JS', k=args.rerank),
           Calibration(item_feature=Item_complexity,
                       data_type="complexity", divergence_type='JS', k=args.rerank),
           Fragmentation(item_story=Item_stories, n_samples=1,
                         divergence_type='JS', k=args.rerank),
           ILD(item_feature=Item_feature, k=args.rerank),
           NDCG_score(k=args.rerank),
           EILD(item_feature=Item_feature, k=args.rerank),
           GiniCoeff(item_genre=Item_genre, k=args.rerank),
           AlternativeVoices(item_minor_major=Item_min_major, divergence_type='JS', k=args.rerank),
           Representation(item_entities=Item_entities, divergence_type='JS', k=args.rerank),
           ]

if args.model == 'CTR':
    ctr = cornac.models.CTR(
        k=args.CTR_k,
        lambda_u=args.CTR_lambda_u,
        lambda_v=args.CTR_lambda_v,
        eta=0.01,
        a=1,
        b=0.01,
        max_iter=50,
        trainable=True,
        verbose=True,
        seed=123,
    )
    cornac.Experiment(
        eval_method=mind_ratio_split,
        models=[ctr],
        metrics=metrics,
        k=args.k,  # the number of candidate set
        rerank=args.rerank,  # the number of re-ranking items
        lambda_constant=args.lambda_constant,
        diversity_objective=[diversity_objective_dict[args.diversity_objective]],
        reranking_metrics=reranking_metrics
    ).run()

elif args.model == 'HFT':
    hft = cornac.models.HFT(
        k=args.HFT_k,
        max_iter=40,
        grad_iter=5,
        l2_reg=0.001,
        lambda_text=0.01,
        vocab_size=8000,
        seed=123,
    )
    cornac.Experiment(
        eval_method=mind_ratio_split,
        models=[hft],
        metrics=metrics,
        k=args.k,  # the number of candidate set
        rerank=args.rerank,  # the number of re-ranking items
        lambda_constant=args.lambda_constant,
        diversity_objective=[diversity_objective_dict[args.diversity_objective]],
        reranking_metrics=reranking_metrics
    ).run()

elif args.model == 'ConvMF':
    conv_mf = cornac.models.ConvMF(
        k=300,
        n_epochs=50,
        cnn_epochs=5,
        lambda_u=args.ConvMF_lambda_u,
        lambda_v=args.ConvMF_lambda_v,
        verbose=True,
        seed=123,
    )
    cornac.Experiment(
        eval_method=mind_ratio_split,
        models=[conv_mf],
        metrics=metrics,
        k=args.k,  # the number of candidate set
        rerank=args.rerank,  # the number of re-ranking items
        lambda_constant=args.lambda_constant,
        diversity_objective=[diversity_objective_dict[args.diversity_objective]],
        reranking_metrics=reranking_metrics
    ).run()

elif args.model == 'DAE':
    dae = cornac.models.DAE(
        qk_dims=[args.DAE_dims],
        pk_dims=[args.DAE_dims],
        n_epochs=100,
        batch_size=100,
        learning_rate=0.001,
        weight_decay=0.0,
        dropout_p=0.5,
        seed=123,
        verbose=True,
    )
    cornac.Experiment(
        eval_method=mind_ratio_split,
        models=[dae],
        metrics=metrics,
        k=args.k,  # the number of candidate set
        rerank=args.rerank,  # the number of re-ranking items
        lambda_constant=args.lambda_constant,
        diversity_objective=[diversity_objective_dict[args.diversity_objective]],
        reranking_metrics=reranking_metrics
    ).run()

elif args.model == 'ENMF':
    enmf = cornac.models.ENMF(
        embedding_size=64,
        num_epochs=160,
        batch_size=256,
        neg_weight=args.ENMF_neg_weight,
        lambda_bilinear=[0.0, 0.0],
        lr=0.05,
        dropout_p=0.7,
        early_stopping=None,
        trainable=True,
        verbose=True,
        seed=123,
    )
    cornac.Experiment(
        eval_method=mind_ratio_split,
        models=[enmf],
        metrics=metrics,
        k=args.k,  # the number of candidate set
        rerank=args.rerank,  # the number of re-ranking items
        lambda_constant=args.lambda_constant,
        diversity_objective=[diversity_objective_dict[args.diversity_objective]],
        reranking_metrics=reranking_metrics
    ).run()

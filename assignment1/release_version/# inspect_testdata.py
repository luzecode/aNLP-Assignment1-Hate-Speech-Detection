# inspect_testdata.py
from utils import read_hate_tweets

TWEETS_ANNO = './data/NAACL_SRW_2016.csv'
TWEETS_TEXT = './data/NAACL_SRW_2016_tweets.json'

# example usage / shape
train_dt, test_dt = read_hate_tweets(TWEETS_ANNO, TWEETS_TEXT)
print(len(train_dt), len(test_dt))
tokens, label = test_dt[0]
# tokens is a list of strings, label is "offensive" or "nonoffensive"
print(type(tokens), tokens[:10], label)
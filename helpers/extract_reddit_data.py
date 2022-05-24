import pandas as pd
from psaw import PushshiftAPI
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns


def data_prep_posts(subreddit, start_time, end_time, filters, limit):
    if len(filters) == 0:
        filters = ['id', 'authors', 'created_utc', 'domain',
                   'url', 'title', 'num_comments']  # defaults if filters are not passed
    posts = list(api.search_submissions(subreddit=subreddit, after=start_time, before=end_time,
                                        filter=filters, limit=limit))

    return pd.DataFrame(posts)


def data_prep_comments(term, start_time, end_time, filters, limit):
    if (len(filters) == 0):
        filters = ['id', 'author', 'created_utc', 'body',
                   'permalink', 'subreddit']
    comments = list(api.search_comments(q=term, after=start_time, before=end_time,
                                        filter=filters, limit=limit))
    return pd.DataFrame(comments)


if __name__ == '__main__':
    api = PushshiftAPI()

    subreddit = 'wallstreetbets'
    start_time = int(dt.datetime(2020, 11, 1).timestamp())
    end_time = int(dt.datetime(2021, 2, 1).timestamp())
    filters = []
    limit = 100
    aggs = 'created_utc'

    df_posts = data_prep_posts(subreddit, start_time, end_time, filters, limit)

    df_posts['datetime'] = df_posts['created_utc'].map(lambda t: dt.datetime.fromtimestamp(t))
    df_posts = df_posts.drop('created_utc', axis=1)
    #  Sort by date
    df_posts = df_posts.sort_values(by='datetime')
    # convert time stamp to date time for easier plotting
    df_posts['datetime'] = pd.to_datetime(df_posts['datetime'])
    stop = 0

import pandas as pd
import datetime as dt
from plotting_helpers import barplot_percentages_on_top, line_plot, log_log_plot
import matplotlib.pyplot as plt
from calculations_helpers import extract_values_counts_as_lists
import seaborn as sns


def data_prep_posts(subreddit, start_time, end_time, filters, limit, api):
    if len(filters) == 0:
        filters = ['id', 'authors', 'created_utc', 'domain',
                   'url', 'title', 'num_comments']  # defaults if filters are not passed
    posts = list(api.search_submissions(subreddit=subreddit, after=start_time, before=end_time,
                                        filter=filters, limit=limit))

    return pd.DataFrame(posts)


def data_prep_comments(term, start_time, end_time, filters, limit, api):
    if (len(filters) == 0):
        filters = ['id', 'author', 'created_utc', 'body',
                   'permalink', 'subreddit']
    comments = list(api.search_comments(q=term, after=start_time, before=end_time,
                                        filter=filters, limit=limit))
    return pd.DataFrame(comments)


def prepare_posts_df(df):
    relevant_columns_df = df[['author', 'author_premium', 'created_utc', 'link_flair_text', 'num_comments', 'subreddit_subscribers',
                              'title', ]]
    relevant_columns_df.to_csv("../kaggleData/cleaned_posts_data.csv", index=False)
    return relevant_columns_df

def convert_utc_todatetime(df):
    df['datetime'] = df['created_utc'].map(lambda t: dt.datetime.fromtimestamp(t))
    df = df.drop('created_utc', axis=1)
    df = df.sort_values(by='datetime')
    # convert time stamps to date time for easier plotting
    df['datetime'] = pd.to_datetime(df['datetime']).dt.date
    df = df.set_index(pd.DatetimeIndex(df['datetime']))
    df.to_csv("../kaggleData/cleaned_posts_data_dt_index", index=False)
    return df


if __name__ == '__main__':
    # api = PushshiftAPI()

    # subreddit = 'wallstreetbets'
    # start_time = int(dt.datetime(2020, 11, 1).timestamp())
    # end_time = int(dt.datetime(2021, 2, 1).timestamp())
    # filters = []
    # limit = 100
    # aggs = 'created_utc'
    #
    # df_posts = data_prep_posts(subreddit, start_time, end_time, filters, limit)
    #
    # df_posts['datetime'] = df_posts['created_utc'].map(lambda t: dt.datetime.fromtimestamp(t))
    # df_posts = df_posts.drop('created_utc', axis=1)
    # #  Sort by date
    # df_posts = df_posts.sort_values(by='datetime')
    # # convert time stamp to date time for easier plotting
    # df_posts['datetime'] = pd.to_datetime(df_posts['datetime'])
    # stop = 0
    sns.set_style('darkgrid')
    comments_filepath = "../kaggleData/archive/wallstreetbets_comments.csv"
    posts_filepath = "../kaggleData/archive/wallstreetbets_posts.csv"
    cleaned_posts_data_path = "../kaggleData/cleaned_posts_data.csv"
    # wsb_comment_data = pd.read_csv(comments_filepath)
    wsb_posts_data = pd.read_csv(cleaned_posts_data_path)
    wsb_posts_data = convert_utc_todatetime(wsb_posts_data)

    one_post_per_author_df = wsb_posts_data.groupby('author').first()
    value_counts_author_premium = one_post_per_author_df['author_premium'].value_counts()

    author_values, author_counts = extract_values_counts_as_lists(wsb_posts_data, 'author')
    author_values = author_values[:10000]
    author_counts_sliced = author_counts[:10000]

    date_value, date_counts = extract_values_counts_as_lists(wsb_posts_data, 'datetime', False)
    author_value_counts_dict = dict(zip(author_values, author_counts_sliced))

    # barplot_percentages_on_top(one_post_per_author_df, "Number of post authors with premium accounts during analysed "
    #                                                    "period",
    #                            'author_premium', 'Has premium account')
    #
    # line_plot(author_values, author_counts_sliced, "Posts per author", "Usernames", "Count", 400, [0, 1000])
    # line_plot(date_value, date_counts, "Posts per day", "Date", "Count", 5)

    all_author_entries = wsb_posts_data["author"].to_numpy(dtype="str")

    log_log_plot(author_counts, xlabel="Post Count", ylabel="Number of Authors", title="Author Post Count Distribution")
    stop = 0

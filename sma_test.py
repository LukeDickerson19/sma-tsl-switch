import time
import subprocess
import sys
sys.path.insert(0, './')
from poloniex import poloniex
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
import numpy as np



''' NOTES

    STRATEGY DESCRIPTION:

        Switch back and forth between a long and short TSL (just tracking, no investment).
        When SMA is going up, invest in the long TSL, and when SMA is going down invest in the short TSL.

    EXPERIMENT DESCRIPTION:
        For each SMA window lengths w in SMA_WINDOWS:
            Create a SMA from t0 to t0-w
            Create a 2nd SMA from t0+w/2 to t0-w/2
                This 2nd SMA will have no lag (but impossible to calculate in live trading), it will be called the "trend"

            Determine the percentage of time the SMA has a similar slope with the trend (both positive or both negative),
            and also determine the percentage of the time they have opposite slopes (one positive, one negative).

        And do this for each coin's price data in COINS

        Output the results in a table at the end of the experiment.

        Is the SMA with the trend more than 50% of the time?

    '''

# constants
QUERI_POLONIEX = False
BACKTEST_DATA_FILE = './price_data_multiple_coins-BTC_ETH_XRP_LTC_ZEC_XMR_STR_DASH_ETC-2hr_intervals-08_01_2018_7am_to_08_01_2019_4am.csv'
LONG_TSLS_FILES = './data/long/'
SHORT_TSLS_FILES = './data/short/'
TETHER = 'USDT'
COINS = [
    'BTC',
    'ETH',
    'XRP',
    'LTC',
    'ZEC',
    'XMR',
    'STR',
    'DASH',
    'ETC',
]
PAIRS = [TETHER + '_' + coin for coin in COINS]
TF = 0.0025 # TF = trading fee
# SMA_WINDOWS = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99] # most accurate if they're all odd integers
SMA_WINDOWS = [100]#[10, 20, 30, 40, 50, 100, 200, 300, 400, 500] # most accurate if they're all odd integers
TSL_VALUES = [0.05]#[0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20]

# pprint constants
DEBUG_WITH_CONSOLE = True
DEBUG_WITH_LOGFILE = True
DEBUG_LOGFILE_PATH = './log.txt'
DEFAULT_INDENT = '|  '
DEFAULT_DRAW_LINE = False


# pretty print the string
# arguments:
#   string = what will be printed
#   indent = what an indent looks like
#   num_indents = number of indents to put in front of the string
#   new_line_start = print a new line in before the string
#   new_line_end = print a new line in after the string
#   draw_line = draw a line on the blank line before or after the string
def pprint(string='',
    indent=DEFAULT_INDENT,
    num_indents=0,
    new_line_start=False,
    new_line_end=False,
    draw_line=DEFAULT_DRAW_LINE):

    if DEBUG_WITH_CONSOLE:

        total_indent0 = ''.join([indent] * num_indents)
        total_indent1 = ''.join([indent] * (num_indents + 1))

        if new_line_start:
            print(total_indent1 if draw_line else total_indent0)

        print(total_indent0 + string)

        if new_line_end:
            print(total_indent1 if draw_line else total_indent0)

    if DEBUG_WITH_LOGFILE:

        f = open(DEBUG_LOGFILE_PATH, 'a')

        new_indent = '\t'

        total_indent0 = ''.join([new_indent] * num_indents)
        total_indent1 = ''.join([new_indent] * (num_indents + 1))

        if new_line_start:
            f.write((total_indent1 if draw_line else total_indent0) + '\n')

        # all these regex's are to make tabs in the string properly
        # asdfasdf is to make sure there's no false positives
        # when replacing the indent
        indent2 = re.sub('\|', 'asdfasdf', indent)
        string = re.sub(indent2, new_indent, re.sub('\|', 'asdfasdf', string))
        f.write(total_indent0 + string + '\n')

        if new_line_end:
            f.write((total_indent1 if draw_line else total_indent0) + '\n')

        f.close()


# setup connection to servers
def poloniex_server():

    API_KEY = '...'
    SECRET_KEY = '...'

    return poloniex(API_KEY, SECRET_KEY)

# get backtesting data
def get_past_prices_from_poloniex(
    startTime, endTime, period, num_periods, conn, save=True):

    # get history data from startTime to endTime
    startTime_unix = time.mktime(startTime.timetuple())
    endTime_unix = time.mktime(endTime.timetuple())

    # get price history data for each pair into a dictionary
    dct = { pair :
        conn.api_query("returnChartData", {
            'currencyPair': pair,
            'start': startTime_unix,
            'end': endTime_unix,
            'period': period
        }) for pair in PAIRS}

    # create 'unix_date' and 'datetime' columns
    df = pd.DataFrame()
    dates = [dct[PAIRS[0]][t]['date'] for t in num_periods]
    df['unix_date'] = pd.Series(dates)
    df['datetime'] = df['unix_date'].apply(
        lambda unix_timestamp : \
        datetime.fromtimestamp(unix_timestamp))

    # remove unneeded data
    for pair, data in dct.items():
        coin = pair[len(TETHER + '_'):]
        data2 = [data[t]['close'] for t in num_periods]
        df[coin] = pd.Series(data2)

    # save df to file
    if save:
        df.to_csv(BACKTEST_DATA_FILE)

    return df
def get_past_prices_from_csv_file():

    # get data from csv file
    df = pd.read_csv(BACKTEST_DATA_FILE, index_col=[0])

    # convert datetime column from string to datetime object
    df['datetime'] = df['datetime'].apply(
        lambda s : datetime.strptime(s, '%Y-%m-%d %H:%M:%S'))

    return df

def init_dct(df):

    # create dct and put time in its own df
    dct = {
        'time_df' : df[['unix_date', 'datetime']],
        'asset_dct' : {}
    }

    # put price data its own df
    for i, coin in enumerate(COINS):
        dct['asset_dct'][coin] = {
            'price_df' : pd.DataFrame({
                'price'     : df[coin],
                'pct_chng'  : df[coin].pct_change()
            })
        }

    # later stuff (sma and tsl) will go in their own dfs as well
    # no indeces will be dropped and reset
    return dct

def get_sma_dct(dct, output_sma_data=False):

    pct_with_trend_df = pd.DataFrame(columns=COINS, index=SMA_WINDOWS)
    pct_with_trend_df.index.name = 'sma_window   '
    total_ave_pct_with_trend = 0

    for i, coin in enumerate(COINS):
        
        sma_dct = {}

        for j, w in enumerate(SMA_WINDOWS):

            # create a SMA from t0 to t0-w
            price_series = dct['asset_dct'][coin]['price_df']['price']
            sma_label = '%sp_SMA' % w
            sma_series = price_series.rolling(window=w).mean()

            # calculate the trend (another SMA from t0+w/2 to t0-w/2)
            # (by shifting the SMA up w/2 indeces and clipping the ends)
            # ... the trend is an SMA with no lag (impossible to calculate in live trading)
            trend_label = '%sp_trend' % w
            trend_series = sma_series.shift(periods=-int(w/2))

            # determine when the SMA has a similar slope to the trend (both positive or both negative)
            # and when they have opposite slope (one positive one negative)
            sma_with_trend_bool_series = \
                ((sma_series.diff() > 0) & (trend_series.diff() > 0)) | \
                ((sma_series.diff() < 0) & (trend_series.diff() < 0))

            # calculate PERCENT of the time the SMA is with the trend
            pct_with_trend = sma_with_trend_bool_series.value_counts(normalize=True).loc[True] * 100
            pct_with_trend_df.at[w, coin] = pct_with_trend
            total_ave_pct_with_trend += pct_with_trend

            # calculate when the sma has a positive slope
            sma_positive_slope_bool_series = sma_series.diff() > 0

            sma_dct[w] = {
                'sma_label'      : sma_label,
                'trend_label'    : trend_label,
                'pct_with_trend' : pct_with_trend,
                'df'             : pd.DataFrame({
                    'trend'              : trend_series,
                    'sma'                : sma_series,
                    'sma_with_trend'     : sma_with_trend_bool_series,
                    'sma_positive_slope' : sma_positive_slope_bool_series
                })
            }

        # print(coin)
        # for k, v in sma_dct[9].items():
        #     print(k)
        #     print(v)
        #     print()
        # input()

        dct['asset_dct'][coin]['sma_dct'] = sma_dct

    if output_sma_data:

        plot_sma_data(dct)

        print('\nPercent of the time the SMA is with the trend:\n')
        print(pct_with_trend_df)

        total_ave_pct_with_trend /= (len(COINS) * len(SMA_WINDOWS))
        print('\nTotal average percent = %.2f %%\n' % total_ave_pct_with_trend)


    return dct
def plot_sma_data(dct):
    for i, coin in enumerate(COINS):
        for j, w in enumerate(SMA_WINDOWS):
            price_series = dct['asset_dct'][coin]['price_df']['price']
            sma_dct_w = dct['asset_dct'][coin]['sma_dct'][w]
            skip_plotting = plot_sma_data_helper(coin, price_series, sma_dct_w)
            if skip_plotting:
                return
def plot_sma_data_helper(coin, price_series, sma_dct_w):

    fig, axes = plt.subplots(figsize=(11, 6))
    
    axes.plot(price_series,  label='price')
    axes.plot(sma_dct_w['df']['sma'],   label=sma_dct_w['sma_label'])
    axes.plot(sma_dct_w['df']['trend'], label=sma_dct_w['trend_label'])

    # highlight x ranges where the SMA is with the trend
    ranges_sma_with_trend = []
    range_start, range_end = None, None
    for index, value in sma_dct_w['df']['sma_with_trend'].items():
        if value: # True
            if range_start != None:
                pass # continue on ...
            else: # just starting
                range_start = index # started new range
        else: # False
            if range_start != None: # found the end
                range_end = index
                ranges_sma_with_trend.append((range_start, range_end))
                range_start, range_end = None, None
            else:
                pass # continue on ... 
    for range_start, range_end in ranges_sma_with_trend:
        # we have to subtract one (as seen below)
        # b/c something to do with the plotting
        # but the df is correct
        axes.axvspan(range_start -1, range_end -1, facecolor='gray', alpha=0.5)

    # set title
    axes.title.set_text(
        '%s/%s %s is w/ the %s %.2f %% of the time' % (
            coin, TETHER,
            sma_dct_w['sma_label'],
            sma_dct_w['trend_label'],
            sma_dct_w['pct_with_trend']))

    # create legend
    plt.legend(loc=(1.02, 0.40))

    # write text explaining when SMA is with trend
    axes.text(
        1.02, 0.05,
        'time when SMA is\nwith the trend',
        transform=axes.transAxes,
        fontsize=10)

    # place grey rectangle box above text
    axes.text(
        1.02, 0.15, '            ',
        transform=axes.transAxes,
        bbox=dict(
            boxstyle='square',
            facecolor='lightgray',
            edgecolor='none'
        ))

    # adjust subplots and display it
    ''' https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
    subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        left  = 0.125  # the left side of the subplots of the figure                      percentage
        right = 0.9    # the right side of the subplots of the figure                     percentage
        bottom = 0.1   # the bottom of the subplots of the figure                         percentage
        top = 0.9      # the top of the subplots of the figure                            percentage
        wspace = 0.2   # the amount of width reserved for blank space between subplots    number
        hspace = 0.2   # the amount of height reserved for white space between subplots   number
        '''
    plt.subplots_adjust(
        left   = 0.10,
        right  = 0.85,
        bottom = 0.10,
        wspace = 0.25, hspace=0.5)
    plt.show()

    # determine if we want to continue to plot SMAs or not
    user_input = input('Press s to skip to end of test, or any other key to continue: ')
    return (user_input == 's' or user_input == 'S')



if __name__ == '__main__':

    conn = poloniex_server()

    # variables
    startTime = datetime(2018, 8, 1, 0, 0, 0)  # year, month, day, hour, minute, second
    endTime   = datetime(2019, 8, 1, 0, 0, 0)
    # period = duration of time steps between rebalances
    #   300 s   900 s    1800 s   7200 s   14400 s   86400 s
    #   5 min   15 min   30 min   2 hrs    4 hrs     1 day
    period = 2 * 60 * 60  # duration of intervals between updates

    # determines the proper number of time steps from startTime to endTime for the given period
    num_periods = range(int((endTime - startTime).total_seconds() / period))

    # import backtest data of COIN1 and COIN2 pair
    # columns=[unix_date, datetime, BTC, ETH, XRP, LTC, ZEC, XMR, STR, DASH, ETC]
    df = get_past_prices_from_poloniex(startTime, endTime, period, num_periods, conn) \
        if QUERI_POLONIEX else get_past_prices_from_csv_file()

    # initialize dct object with price data
    dct = init_dct(df)

    # calculate SMA data
    dct = get_sma_dct(dct, output_sma_data=True)


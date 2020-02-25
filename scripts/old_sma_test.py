import time
import sys
sys.path.insert(0, './')
from poloniex import poloniex
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 10)
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
TRADING_FEE = 0.0025
# SMA_WINDOWS = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99] # most accurate if they're all odd integers
SMA_WINDOWS = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500] # most accurate if they're all odd integers

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
    startTime, endTime, period, num_periods, conn):

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
    df.to_csv(BACKTEST_DATA_FILE)

    return df

def get_past_prices_from_csv_file():

    return pd.read_csv(BACKTEST_DATA_FILE, index_col=[0])



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
    df = get_past_prices_from_poloniex(startTime, endTime, period, num_periods, conn) \
        if QUERI_POLONIEX else get_past_prices_from_csv_file()
    # columns=[unix_date, datetime, BTC, ETH, XRP, LTC, ZEC, XMR, STR, DASH, ETC]

    # get percent change of price each time step
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pct_change.html   
    df.rename(columns={coin : coin + '_price' for coin in COINS}, inplace=True)
    for coin in COINS:
        df['%s_pct_chng' % coin] = df[coin + '_price'].pct_change()
    df.drop([0], inplace=True) # remove first row (b/c it has a NaN value)
    df.reset_index(drop=True, inplace=True) # reset index accordingly
    # columns=[unix_date, datetime, BTC_price, BTC_pct_chng, ETH_price, ET_pct_chng, ... ]

    # print(df)

    # # iterate over data
    # for i, row in df.iterrows():
    #     print(i)
    #     print(row)
    #     print()

    #     input()

    # can also put it all in a dct ... might be easier this way ...
    dct = {}
    for coin in COINS:
        dct[coin] = pd.DataFrame({
            'datetime' : df['datetime'],
            'price'    : df[coin + '_price'],
            'pct_chng' : df[coin + '_pct_chng']
        })

    pct_with_trend_df = pd.DataFrame(columns=COINS, index=SMA_WINDOWS)
    pct_with_trend_df.index.name = 'sma_window   '
    total_ave_pct_with_trend = 0
    p = True # if p: # plot, else don't plot
    for i, (coin, df) in enumerate(dct.items()):
        for j, w in enumerate(SMA_WINDOWS):

            # Create a SMA from t0 to t0-w
            col1_lbl = '%sp_SMA' % w
            df[col1_lbl] = df['price'].rolling(window=w).mean()

            # Create another SMA from t0+w/2 to t0-w/2
            # (by shifting the real SMA up w/2 indeces and clipping the ends)
            # ... this 2nd SMA will have no lag (but impossible to calculate in live trading)
            col2_lbl = '%sp_trend' % w
            df[col2_lbl] = df[col1_lbl].shift(periods=-int(w/2))
            nan_at_beginning = list(range(int(w-1))) # OR do int(w/2) to clip the col2_lbl
            nan_at_end = list(map(lambda i : (df.shape[0]-1) - i, list(range(int(w/2)))[::-1]))
            indeces_to_clip = nan_at_beginning + nan_at_end
            df.drop(indeces_to_clip, inplace=True) # remove first row (b/c it has a NaN value)
            df.reset_index(drop=True, inplace=True) # reset index accordingly

            # Determine the percentage of time both SMAs
            # have similar slope (both positive or both negative) and
            # when they have opposite slope (one positive one negative).
            #     when they have similar slope, the 1st SMA is with the trend,
            #     when they have opposite slope, the 1st SMA hasn't realized the trend shift yet.
            df['sma_with_trend'] = \
                ((df[col1_lbl].diff() > 0) & (df[col2_lbl].diff() > 0)) | \
                ((df[col1_lbl].diff() < 0) & (df[col2_lbl].diff() < 0))
            df.drop([0], inplace=True) # remove first row b/c the diff has an NaN here
            df.reset_index(drop=True, inplace=True) # reset index accordingly

            # plot the price, SMA and trend,
            # and highlight the regions where the sma is with the trend
            # and create a legend labeling everything and outputting the
            # percentage of the time the SMA is with the trend            
            pct_with_trend = df['sma_with_trend'].value_counts(normalize=True).loc[True] * 100
            pct_with_trend_df.at[w, coin] = pct_with_trend
            total_ave_pct_with_trend += pct_with_trend
            if p:
                fig, axes = plt.subplots(figsize=(11, 6))
                axes.plot(df['price'],  label='price')
                axes.plot(df[col1_lbl], label=col1_lbl)
                axes.plot(df[col2_lbl], label=col2_lbl)
                ranges_sma_with_trend = []
                range_start, range_end = None, None
                for index, value in df['sma_with_trend'].items():
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
                    axes.axvspan(range_start, range_end, color='gray', alpha=0.5)
                axes.title.set_text(
                    '%s/%s %s is w/ the %s %.2f %% of the time' % (
                        coin, TETHER, col1_lbl, col2_lbl, pct_with_trend))                
                plt.legend(loc=(1.02, 0.40))
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
                user_input = input('Press s to skip to end of test, or any other key to continue: ')
                p = not (user_input == 's' or user_input == 'S')


    print('\nPercent of the time the SMA is with the trend:\n')
    print(pct_with_trend_df)

    total_ave_pct_with_trend /= (len(COINS) * len(SMA_WINDOWS))
    print('\nTotal average percent = %.2f %%\n' % total_ave_pct_with_trend)


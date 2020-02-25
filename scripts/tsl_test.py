import time
import subprocess
import sys
sys.path.insert(0, './')
from poloniex import poloniex
from block_printer import BlockPrinter
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
MAX_ROWS = 200
pd.set_option('display.max_rows', MAX_ROWS)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None # source: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
import numpy as np



# constants
QUERI_POLONIEX = False
CALCULATE_TSL_DATA = True
BACKTEST_DATA_FILE = '../data/price_data/price_data_multiple_coins-BTC_ETH_XRP_LTC_ZEC_XMR_STR_DASH_ETC-2hr_intervals-08_01_2018_7am_to_08_01_2019_4am.csv'
# BACKTEST_DATA_FILE = '../data/price_data/price_data_one_coin-BTC_USD-5min_intervals-ONE_QUARTER-11-21-2019-12am_to_02-21-2020-12am.csv'
BACKTEST_DATA_FILE2 = '../../tsls/sma_switch1/data/price_data.csv'
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
COINS = ['BTC']
PAIRS = [TETHER + '_' + coin for coin in COINS]
TF = 0.0025 # TF = trading fee
INCLUDE_TF = True  # flag if we want to include the TF in our calculations

# SMA_WINDOWS = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99] # most accurate if they're all odd integers
SMA_WINDOWS = [100]#[50]#[10, 20, 30, 40, 50, 100, 200, 300, 400, 500] # most accurate if they're all odd integers
TSL_VALUES = [0.01]#[0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20]

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
def get_past_prices_from_csv_file2():

    # get data from csv file
    df = pd.read_csv(BACKTEST_DATA_FILE2, index_col=[0])

    df = df.rename(columns={'date' : 'unix_date', 'price' : 'BTC'})

    # convert 'date' column from unix timestamp to datetime
    df['datetime'] = df['unix_date'].apply(
        lambda unix_timestamp : datetime.fromtimestamp(unix_timestamp))

    # # convert datetime column from string to datetime object
    # df['datetime'] = df['datetime'].apply(
    #     lambda s : datetime.strptime(s, '%Y-%m-%d %H:%M:%S'))

    return df

def init_dct(df):

    # create dct and put time in its own df
    dct = {
        'time_df'   : df[['unix_date', 'datetime']],
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

# get SMA data
def get_sma_dct(dct, output_sma_data=False):

    print('\nCalculating SMA data ...')
    start_time = datetime.now()

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

            # also ... create the boolinger bands for this SMA
            std_dev_series = price_series.rolling(window=w).std()

            sma_w_dct = {
                'sma_label'      : sma_label,
                'trend_label'    : trend_label,
                'pct_with_trend' : pct_with_trend,
                'df'             : pd.DataFrame({
                    'trend'                 : trend_series,
                    'sma'                   : sma_series,
                    'sma_with_trend'        : sma_with_trend_bool_series,
                    'sma_positive_slope'    : sma_positive_slope_bool_series,
                    'std_dev'               : std_dev_series,
                    'bollinger_upper_bound' : sma_series + 2 * std_dev_series,
                    'bollinger_lower_bound' : sma_series - 2 * std_dev_series
                })
            }
            sma_dct[w] = sma_w_dct

        # print(coin)
        # for k, v in sma_dct[9].items():
        #     print(k)
        #     print(v)
        #     print()
        # input()

        dct['asset_dct'][coin]['sma_dct'] = sma_dct

    end_time = datetime.now()
    print('SMA data aquired. Duration: %.2f seconds\n' % (end_time - start_time).total_seconds())

    if output_sma_data:

        plot_sma_data(dct)

        print('\nPercent of the time the SMA is with the trend:\n')
        print(pct_with_trend_df)

        total_ave_pct_with_trend /= (len(COINS) * len(SMA_WINDOWS))
        print('\nTotal average percent = %.2f %%\n' % total_ave_pct_with_trend)


    return dct
def plot_sma_data(dct):
    for i, coin in enumerate(COINS):
        price_series = dct['asset_dct'][coin]['price_df']['price']
        for j, w in enumerate(SMA_WINDOWS):
            sma_w_dct = dct['asset_dct'][coin]['sma_dct'][w]
            skip_plotting = plot_sma_data_helper(coin, price_series, sma_w_dct)
            if skip_plotting:
                return
def plot_sma_data_helper(coin, price_series, sma_w_dct):

    fig, axes = plt.subplots(figsize=(11, 6))

    axes.plot(price_series,  label='price')
    axes.plot(sma_w_dct['df']['sma'],   label=sma_w_dct['sma_label'])
    axes.plot(sma_w_dct['df']['trend'], label=sma_w_dct['trend_label'])

    # highlight x ranges where the SMA is with the trend
    ranges_sma_with_trend = []
    range_start, range_end = None, None
    for index, value in sma_w_dct['df']['sma_with_trend'].items():
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
            sma_w_dct['sma_label'],
            sma_w_dct['trend_label'],
            sma_w_dct['pct_with_trend']))

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

# get TSL data
def get_tsl_dct(dct, output_tsl_data=False, save_tsl_data=True):

    print('\nCalculating TSL data ...')
    start_time = datetime.now()

    for i, (coin, coin_data) in enumerate(dct['asset_dct'].items()):
        # print('coin = %s' % coin)
        price_df = coin_data['price_df']
        # print('price_df')
        # print(price_df)
        for j, (w, sma_w_dct) in enumerate(coin_data['sma_dct'].items()):
            # print('sma_window = %d' % int(w))
            sma_w_df = sma_w_dct['df']
            # print('sma_w_df')
            # print(sma_w_df)

            tsl_dct = {}
            for x in TSL_VALUES:
                # print('x = %.2f' % x)
                tsl_x_dct = get_tsl_dct_helper1(coin, w, x, price_df['price'])
                tsl_dct[x] = tsl_x_dct

            dct['asset_dct'][coin]['sma_dct'][w]['tsl_dct'] = tsl_dct

    end_time = datetime.now()
    print('TSL data aquired. Duration: %.2f seconds\n' % (end_time - start_time).total_seconds())

    if save_tsl_data:
        base_path = '../data/tsl_data/'
        subprocess.run(['mkdir', base_path])
        for coin in COINS:
            subprocess.run(['mkdir', base_path + 'asset_%s/' % (coin)])
            for w in SMA_WINDOWS:
                subprocess.run(['mkdir', base_path + 'asset_%s/sma_window_%s/' % (coin, w)])
                for x in TSL_VALUES:
                    base_path2 = base_path + 'asset_%s/sma_window_%s/tsl_x_%s/' % (coin, w, x)
                    subprocess.run(['mkdir', base_path2])
                    long_csv_path  = base_path2 + 'long_df.csv'
                    short_csv_path = base_path2 + 'short_df.csv'
                    long_df  = dct['asset_dct'][coin]['sma_dct'][w]['tsl_dct'][x]['long_df']
                    short_df = dct['asset_dct'][coin]['sma_dct'][w]['tsl_dct'][x]['short_df']
                    long_df.to_csv(long_csv_path)
                    short_df.to_csv(short_csv_path)

    if output_tsl_data:

        plot_tsl_data(dct)

    return dct
def get_tsl_dct_helper1(coin, w, x, price_series, verbose=False, bp=BlockPrinter()):

    columns = {
        'enter_price',   # the price that the trade was entered at
        'stop_loss',     # the price that it will take to exit the trade
        'dxv',           # dxv = dx VALUE (not the percentage), aka difference between enter_price and stop_loss
        'cur_price_pl',  # profit/loss (pl) of the current actual price
        'cur_sl_pl',     # profit/loss (pl) of the current stop loss
        'tot_price_pl',  # profit/loss (pl) of the total (from beginning until now) of the current price
        'tot_sl_pl',     # profit/loss (pl) of the total (from beginning until now) of the current stop loss
        'active',        # boolean flag if the TSL exists or not
        'triggered',     # boolean flag if the TSL has been triggered or not
        'invested'       # boolean flag if the algorithm is invested long/short
    }
    long_df = pd.DataFrame(columns=columns)
    short_df = pd.DataFrame(columns=columns)

    # init boolean flags to False (except 'invested' b/c its more readable to update incrementally)
    long_df['active'],    short_df['active']    = False, False
    long_df['triggered'], short_df['triggered'] = False, False
    # long_df['invested'],  short_df['invested']  = False, False

    tracking_long = True
    for i, price in enumerate(price_series):

        if verbose:
            print('-' * 100)
            print('i = %d   price = %.4f' % (i, price))
            print('TRACKING LONG' if tracking_long else 'TRACKING SHORT')
        else:
            bp.print('%.1f %%' % (100 * i / price_series.shape[0]))

        # if your tracking long, update the long function 1st and the short tsl 2nd, and vice versa
        if tracking_long:

            # update long tsl 1st
            triggered = update_long_tsl(x, long_df, price, i, init_tsl=i==0, verbose=verbose)

            # then update the short tsl 2nd
            if not triggered:
                short_df.at[i, 'dxv']         = np.nan
                short_df.at[i, 'stop_loss']   = np.nan
                short_df.at[i, 'enter_price'] = np.nan
                short_df.at[i, 'triggered']   = False
                short_df.at[i, 'active']      = False

            else: # if the long tsl was triggered, start tracking short
                tracking_long = False
                _ = update_short_tsl(x, short_df, price, i, init_tsl=True, verbose=verbose)

        else: # tracking short

            triggered = update_short_tsl(x, short_df, price, i, verbose=verbose)

            if not triggered:
                long_df.at[i, 'dxv']         = np.nan
                long_df.at[i, 'stop_loss']   = np.nan
                long_df.at[i, 'enter_price'] = np.nan
                long_df.at[i, 'triggered']   = False
                long_df.at[i, 'active']      = False

            else: # if the short tsl was triggered, start tracking long
                tracking_long = True
                _ = update_long_tsl(x, long_df, price, i, init_tsl=True, verbose=verbose)

        if verbose:
            print('\nlong_df')
            print(long_df)

            print('\nshort_df')
            print(short_df)

        # input()

    return {
        'long_df'  : long_df,
        'short_df' : short_df
    }
def get_tsl_dct_helper2(coin, w, x, price_series, verbose=False, bp=BlockPrinter()):

    columns = {
        'enter_price',   # the price that the trade was entered at
        'stop_loss',     # the price that it will take to exit the trade
        'dxv',           # dxv = dx VALUE (not the percentage), aka difference between enter_price and stop_loss
        'cur_price_pl',  # profit/loss (pl) of the current actual price
        'cur_sl_pl',     # profit/loss (pl) of the current stop loss
        'tot_price_pl',  # profit/loss (pl) of the total (from beginning until now) of the current price
        'tot_sl_pl',     # profit/loss (pl) of the total (from beginning until now) of the current stop loss
        'active',        # boolean flag if the TSL exists or not
        'triggered',     # boolean flag if the TSL has been triggered or not
        'invested'       # boolean flag if the algorithm is invested long/short
    }
    long_df = pd.DataFrame(columns=columns)
    short_df = pd.DataFrame(columns=columns)

    # init boolean flags to False (except 'invested' b/c its more readable to update incrementally)
    long_df['active'],    short_df['active']    = False, False
    long_df['triggered'], short_df['triggered'] = False, False
    # long_df['invested'],  short_df['invested']  = False, False

    tracking_long = True
    for i, price in enumerate(price_series):

        if verbose:
            print('-' * 100)
            print('i = %d   price = %.4f' % (i, price))
            print('TRACKING LONG' if tracking_long else 'TRACKING SHORT')

        # if your tracking long, update the long function 1st and the short tsl 2nd, and vice versa
        if tracking_long:

            # update long tsl 1st
            triggered = update_long_tsl(x, long_df, price, i, init_tsl=i==0, verbose=verbose)

            # then update the short tsl 2nd
            if not triggered:
                short_df.at[i, 'dxv']         = np.nan
                short_df.at[i, 'stop_loss']   = np.nan
                short_df.at[i, 'enter_price'] = np.nan
                short_df.at[i, 'triggered']   = False
                short_df.at[i, 'active']      = False

            else: # if the long tsl was triggered, start tracking short
                tracking_long = False
                _ = update_short_tsl(x, short_df, price, i, init_tsl=True, verbose=verbose)

        else: # tracking short

            triggered = update_short_tsl(x, short_df, price, i, verbose=verbose)

            if not triggered:
                long_df.at[i, 'dxv']         = np.nan
                long_df.at[i, 'stop_loss']   = np.nan
                long_df.at[i, 'enter_price'] = np.nan
                long_df.at[i, 'triggered']   = False
                long_df.at[i, 'active']      = False

            else: # if the short tsl was triggered, start tracking long
                tracking_long = True
                _ = update_long_tsl(x, long_df, price, i, init_tsl=True, verbose=verbose)

        if verbose:
            print('\nlong_df')
            print(long_df)

            print('\nshort_df')
            print(short_df)

        # input()

    return {
        'long_df'  : long_df,
        'short_df' : short_df
    }
def update_long_tsl(x, df, price, i, init_tsl=False, verbose=False):

    # update active status
    df.at[i, 'active'] = True

    # update TSL variables
    sl   = df.at[i, 'stop_loss']   = df.loc[i-1, 'stop_loss']   if i != 0 else np.nan  # sl = (previous) stop loss
    dxv  = df.at[i, 'dxv']         = df.loc[i-1, 'dxv']         if i != 0 else np.nan
    entp = df.at[i, 'enter_price'] = df.loc[i-1, 'enter_price'] if i != 0 else np.nan
    triggered = False

    if init_tsl:
        if verbose: print('initializing long tsl')
        dxv = df.at[i, 'dxv']   = price * x  # update dxv
        df.at[i, 'stop_loss']   = price - dxv # set new tsl
        df.at[i, 'enter_price'] = price # update new entrance price
    elif price <= sl:  # stop loss triggered!
        if verbose: print('long tsl: stop loss triggered!')
        triggered = True
    elif price <= sl + dxv and price > sl: # else it stayed within its stop loss
        if verbose: print('long tsl: price stayed within its stop loss')
        # pass
    elif price > sl + dxv: # else it went up, thus dragging the stop loss up also
        if verbose: print('long tsl: price went up')
        df.at[i, 'stop_loss'] = price - dxv
    else:
        if verbose: print('LONG TSL FUCKED UP!')

    # update TSL triggered status
    df.at[i, 'triggered'] = triggered

    return triggered
def update_short_tsl(x, df, price, i, init_tsl=False, verbose=False):

    # update active status
    df.at[i, 'active'] = True

    # update TSL variables
    sl   = df.at[i, 'stop_loss']   = df.loc[i-1, 'stop_loss']   if i != 0 else np.nan  # sl = (previous) stop loss
    dxv  = df.at[i, 'dxv']         = df.loc[i-1, 'dxv']         if i != 0 else np.nan
    entp = df.at[i, 'enter_price'] = df.loc[i-1, 'enter_price'] if i != 0 else np.nan
    triggered = False

    if init_tsl:
        if verbose: print('initializing short tsl')
        dxv = df.at[i, 'dxv']   = price * x  # update dxv
        df.at[i, 'stop_loss']   = price + dxv # set new tsl
        df.at[i, 'enter_price'] = price # update new entrance price
    elif price >= sl:  # stop loss triggered!
        if verbose: print('short tsl: stop loss triggered!')
        triggered = True
    elif price >= sl - dxv and price < sl: # else it stayed within its stop loss
        if verbose: print('short tsl: price stayed within its stop loss')
        # pass
    elif price < sl - dxv: # else it went down, thus dragging the stop loss down also
        if verbose: print('short tsl: price went down')
        df.at[i, 'stop_loss'] = price + dxv
    else:
        if verbose: print('SHORT TSL FUCKED UP')

    # update TSL triggered status
    df.at[i, 'triggered'] = triggered

    return triggered
def plot_tsl_data(dct):

    # create date_labels and x_tick_indeces
    first_date = df['datetime'].iloc[0]
    date_fmt = '%m-%d-%Y'
    date_labels = [first_date.strftime(date_fmt)]
    x_tick_indeces = [0]
    previous_month = first_date.strftime('%m')
    for i, row in df.iterrows():
        current_month = row['datetime'].strftime('%m')
        if current_month != previous_month:
            date_labels.append(row['datetime'].strftime(date_fmt))
            x_tick_indeces.append(i)
        previous_month = current_month
    last_date = df['datetime'].iloc[-1]
    if last_date != date_labels[-1]:
        date_labels.append(last_date.strftime(date_fmt))
        x_tick_indeces.append(df['datetime'].tolist().index(last_date))
    # for i, l in zip(x_tick_indeces, date_labels):
    #     print(i, l)

    # plot each TSL
    for i, coin in enumerate(COINS):
        price_series = dct['asset_dct'][coin]['price_df']['price']
        for j, w in enumerate(SMA_WINDOWS):
            sma_w_dct = dct['asset_dct'][coin]['sma_dct'][w]
            for k, x in enumerate(TSL_VALUES):
                tsl_x_dct = sma_w_dct['tsl_dct'][x]
                skip_plotting = plot_tsl_data_helper(date_labels, x_tick_indeces, coin, price_series, sma_w_dct, x, tsl_x_dct)
                if skip_plotting:
                    return
def plot_tsl_data_helper(date_labels, x_tick_indeces, coin, price_series, sma_w_dct, x, tsl_x_dct):

    long_x_str,  long_df  = '%.1f' % (100*x), tsl_x_dct['long_df']
    short_x_str, short_df = '%.1f' % (100*x), tsl_x_dct['short_df']

    sma_data = sma_w_dct['df']['sma']
    sma_lbl  = '%s' % sma_w_dct['sma_label']

    bollinger_lbl = 'bollinger bands (2 std devs)'
    bollinger_upper_bound = sma_w_dct['df']['bollinger_upper_bound']
    bollinger_lower_bound = sma_w_dct['df']['bollinger_lower_bound']

    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        num='stop loss = %s' % long_x_str,
        figsize=(10.5, 6.5),
        sharex=True, sharey=False)

    _legend_loc, _b2a = 'center left', (1, 0.5) # puts legend ouside plot

    ax.plot(price_series,          color='black', label='price')
    # ax.plot(sma_data,              color='blue',  label=sma_lbl)
    # ax.plot(bollinger_upper_bound, color='blue',  label=bollinger_lbl, linestyle='--')
    # ax.plot(bollinger_lower_bound, color='blue',  label=None, linestyle='--')
    ax.plot(long_df['stop_loss'],  color='green', label='%s%% long TSL ' % long_x_str)
    ax.plot(short_df['stop_loss'], color='red',   label='%s%% short TSL ' % short_x_str)
    ax.legend(loc=_legend_loc, bbox_to_anchor=_b2a)
    ax.grid()
    # ax.yaxis.grid()  # draw horizontal lines
    ax.yaxis.set_zorder(-1.0)  # draw horizontal lines behind histogram bars
    ax.set_title('TSL Chart')
    ax.set_xticks(x_tick_indeces)
    ax.set_xticklabels('')

    # highlight SMA slope + green
    # highlight SMA slope - red
    def highlight_sma_slope(up=True, color='green'):
        ranges_sma_direction = []
        range_start, range_end = None, None
        for index, value in sma_w_dct['df']['sma_positive_slope'].items():
            if value == up: # True
                if range_start != None:
                    pass # continue on ...
                else: # just starting
                    range_start = index # started new range
            else: # False
                if range_start != None: # found the end
                    range_end = index
                    ranges_sma_direction.append((range_start, range_end))
                    range_start, range_end = None, None
                else:
                    pass # continue on ...
        for range_start, range_end in ranges_sma_direction:
            ax.axvspan(range_start, range_end, color=color, alpha=0.5)
    highlight_sma_slope(up=True,  color='green')
    highlight_sma_slope(up=False, color='red')

    ax.set_xticks(x_tick_indeces)
    ax.set_xticklabels(date_labels, ha='right', rotation=45)  # x axis should show date_labeles

    # plt.tight_layout()
    fig.subplots_adjust(
        right=0.80,
        left=0.075,
        bottom=0.15,
        top=0.95)

    plt.show()

    # determine if we want to continue to plot SMAs or not
    user_input = input('Press s to skip to end of test, or any other key to continue: ')
    return (user_input == 's' or user_input == 'S')
def get_tsl_dct_from_csv_files(dct):

    print('\nGetting TSL data from CSV files ...')
    start_time = datetime.now()

    for coin in COINS:
        for w in SMA_WINDOWS:
            dct['asset_dct'][coin]['sma_dct'][w]['tsl_dct'] = {}
            for x in TSL_VALUES:
                filepath = '../data/tsl_data/asset_%s/sma_window_%s/tsl_x_%s/' % (coin, w, x)
                try:
                    long_df  = pd.read_csv(filepath + 'long_df.csv',  index_col=[0])
                    short_df = pd.read_csv(filepath + 'short_df.csv', index_col=[0])
                except:
                    print('Failed to find data at: %s\n' % (filepath))
                    sys.exit()
                
                dct['asset_dct'][coin]['sma_dct'][w]['tsl_dct'][x] = {
                    'long_df'  : long_df,
                    'short_df' : short_df
                }

    end_time = datetime.now()
    print('TSL data aquired. Duration: %.2f seconds\n' % (end_time - start_time).total_seconds())

    return dct


# get investment data
def get_investments(dct, output_investment_data=False):

    print('\nCalculating investment returns ...')
    start_time = datetime.now()

    for i, (coin, coin_data) in enumerate(dct['asset_dct'].items()):
        # print('coin = %s' % coin)
        price_df = coin_data['price_df']
        # print('price_df')
        # print(price_df)
        for j, (w, sma_w_dct) in enumerate(coin_data['sma_dct'].items()):
            # print('sma_window = %d' % int(w))
            sma_w_df = sma_w_dct['df']
            # print('sma_w_df')
            # print(sma_w_df)

            tsl_dct = {}
            for k, (x, tsl_x_dct) in enumerate(sma_w_dct['tsl_dct'].items()):
                tsl_x_dct = get_investments_helper3(
                    coin, w, x, price_df['price'], sma_w_df, tsl_x_dct, verbose=False)
                tsl_dct[x] = tsl_x_dct

            dct['asset_dct'][coin]['sma_dct'][w]['tsl_dct'] = tsl_dct

    end_time = datetime.now()
    print('Investment returns aquired. Duration: %.2f seconds\n' % (end_time - start_time).total_seconds())

    if output_investment_data:

        plot_investment_data(dct)

    return dct
def get_investments_helper2(coin, w, x, price_series, sma_w_df, tsl_x_dct, verbose=False, bp=BlockPrinter()):

    ''' NOTE:
        This investment helper switches back and forth between long and short TSLs
        regardless of SMA. All tracking gets an investment.

        This requires the use of: get_tsl_dct_helper1().
        '''

    long_df, short_df = tsl_x_dct['long_df'], tsl_x_dct['short_df']
    invested_long, invested_short = True, False

    # init the 1st w 'invested' flags to false, and the 1st w 'tot_sl_pl' and 'tot_price_pl' to 1.0
    long_df['invested'][:w]      = False
    long_df['tot_sl_pl'][:w]     = 1.0
    long_df['tot_price_pl'][:w]  = 1.0
    short_df['invested'][:w]     = False
    short_df['tot_sl_pl'][:w]    = 1.0
    short_df['tot_price_pl'][:w] = 1.0

    tot_returns = [1.00] * w

    # print('BEFORE')

    # print('\nlong df')
    # print(long_df)
    # print('\nshort_df')
    # print(short_df)
    # print('tot_returns')
    # print(pd.Series(tot_returns))
    # input()

    # print(price_series)

    # iterate over the price data but start at w-1+1
    # -1 b/c 0 indexing, +1 b/c we want to start when we have an SMA slope to work w/
    for i, price in price_series.iloc[w-1+1:].items():

        # set investment status at i equal to what it was at i-1
        long_df.at[i,  'invested'] = invested_long
        short_df.at[i, 'invested'] = invested_short

        if verbose:
            print('-' * 100)
            print('i = %d   price = %.4f\n' % (i, price))
            print(sma_w_df.iloc[i])
            print('\nlong_df.iloc[%d]' % i)
            print(long_df.iloc[i])
            print('\nshort_df.iloc[%d]' % i)
            print(short_df.iloc[i])
        else:
            bp.print('%.1f %%' % (100 * i / price_series.shape[0]))

        # if invested long
        if long_df.at[i, 'invested']:

            if verbose: print('invested long')

            # if long TSL triggered
            if long_df.at[i, 'triggered']:

                if verbose: print('long TSL triggered')

                # exit long investment
                long_df = update_investment_returns(price, long_df, i, 'long')
                invested_long = False

                # enter short investment
                invested_short = True
                short_df.at[i, 'invested'] = invested_short
                short_df = update_investment_returns(price, short_df, i, 'short')

                if verbose: print('exitted long investment & entered short investment')

            else: # else long TSL wasn't triggered

                if verbose: print('long TSL not triggered')

                # update long investment
                long_df = update_investment_returns(price, long_df, i, 'long')

                if verbose: print('long investment updated')

        # elif invested short
        elif short_df.at[i, 'invested']:

            if verbose: print('invested short')

            # if short TSL triggered:
            if short_df.at[i, 'triggered']:

                if verbose: print('short TSL triggered')

                # exit short investment
                short_df = update_investment_returns(price, short_df, i, 'short')
                invested_short = False

                # enter long investment
                invested_long = True
                long_df.at[i, 'invested'] = invested_long
                long_df = update_investment_returns(price, long_df, i, 'long')

                if verbose: print('exitted short investment & entered long investment')

            else: # else short TSL not triggered

                if verbose: print('short TSL not triggered')

                # update short investment
                short_df = update_investment_returns(price, short_df, i, 'short')

                if verbose: print('updated short investment')


        long_df  = update_portfolio_returns(long_df,  i)
        short_df = update_portfolio_returns(short_df, i)
        tot_returns_i = tot_returns[-1]
        if long_df.at[i, 'invested'] and long_df.at[i, 'triggered']:
            tot_returns_i *= 1.0 + long_df.loc[i, 'cur_sl_pl']
        elif short_df.at[i, 'invested'] and short_df.at[i, 'triggered']:
            tot_returns_i *= 1.0 + short_df.loc[i, 'cur_sl_pl']
        tot_returns.append(tot_returns_i)

        if verbose:
            print()
            input()

    # print('AFTER')
    # print('MAX_ROWS = %d' % MAX_ROWS)

    # print('\nlong df')
    # print(long_df)
    # print('\nshort_df')
    # print(short_df)
    # print('tot_returns')
    # print(pd.Series(tot_returns))
    # input()


    return {
        'long_df'     : long_df,
        'short_df'    : short_df,
        'tot_returns' : tot_returns
    }
def get_investments_helper3(coin, w, x, price_series, sma_w_df, tsl_x_dct, verbose=False, bp=BlockPrinter()):

    ''' NOTE:
        This investment helper switches back and forth between long and short TSLs.
        If SMA slope is positive it can only go long.
        If SMA slope is negative it can only go short.
        When the SMA slope switches, for example, postive to negative, if the
        algo is already tracking a short TSL, it will not invest short.

        This requires the use of: get_tsl_dct_helper1().
        '''

    long_df, short_df = tsl_x_dct['long_df'], tsl_x_dct['short_df']
    invested_long, invested_short = False, False

    # init the 1st w 'invested' flags to false, and the 1st w 'tot_sl_pl' and 'tot_price_pl' to 1.0
    long_df['invested'][:w]      = False
    long_df['tot_sl_pl'][:w]     = 1.0
    long_df['tot_price_pl'][:w]  = 1.0
    short_df['invested'][:w]     = False
    short_df['tot_sl_pl'][:w]    = 1.0
    short_df['tot_price_pl'][:w] = 1.0

    tot_returns = [1.00] * w

    # print('BEFORE')

    # print('\nlong df')
    # print(long_df)
    # print('\nshort_df')
    # print(short_df)
    # print('tot_returns')
    # print(pd.Series(tot_returns))
    # input()

    # print(price_series)

    # iterate over the price data but start at w-1+1
    # -1 b/c 0 indexing, +1 b/c we want to start when we have an SMA slope to work w/
    for i, price in price_series.iloc[w-1+1:].items():

        # set investment status at i equal to what it was at i-1
        long_df.at[i,  'invested'] = invested_long
        short_df.at[i, 'invested'] = invested_short
        # long_df.at[i,  'invested'] = long_df.at[i-1,  'invested']
        # short_df.at[i, 'invested'] = short_df.at[i-1, 'invested']

        if verbose:
            print('-' * 100)
            print('i = %d   price = %.4f\n' % (i, price))
            print(sma_w_df.iloc[i])
            print('\nlong_df.iloc[%d]' % i)
            print(long_df.iloc[i])
            print('\nshort_df.iloc[%d]' % i)
            print(short_df.iloc[i])
        else:
            bp.print('%.1f %%' % (100 * i / price_series.shape[0]))


        if sma_w_df.iloc[i]['sma_positive_slope']: # SMA has positive slope

            if verbose: print('\nSMA slope is positive')

            # if invested long
            if long_df.at[i, 'invested']:

                if verbose: print('invested long')

                # if long TSL triggered
                if long_df.at[i, 'triggered']:

                    if verbose: print('long TSL triggered')

                    # exit long investment
                    long_df = update_investment_returns(price, long_df, i, 'long')
                    invested_long = False

                    if verbose: print('exitted long investment')

                else: # else long TSL wasn't triggered

                    if verbose: print('long TSL not triggered')

                    # update long investment
                    long_df = update_investment_returns(price, long_df, i, 'long')

                    if verbose: print('long investment updated')

            # elif invested short (for corner case: SMA is still catching up to trend shift):
            elif short_df.at[i, 'invested']:

                if verbose: print('invested short: SMA is still catching up to trend shift')

                # if short TSL triggered:
                if short_df.at[i, 'triggered']:

                    if verbose: print('short TSL triggered')

                    # exit short investment
                    short_df = update_investment_returns(price, short_df, i, 'short')
                    invested_short = False

                    # enter long investment
                    invested_long = True
                    long_df.at[i, 'invested'] = invested_long
                    long_df = update_investment_returns(price, long_df, i, 'long')

                    if verbose: print('exitted short investment & entered long investment')

                else: # else short TSL not triggered

                    if verbose: print('short TSL not triggered')

                    short_df = update_investment_returns(price, short_df, i, 'short')

                    if verbose: print('updated short investment')

            # else not invested long or short
            else:

                if verbose: print('not invested long')

                # if short TSL is triggered
                if short_df.at[i, 'triggered']:

                    if verbose: print('short TSL triggered')

                    # enter a long investment
                    invested_long = True
                    long_df.at[i, 'invested'] = invested_long
                    long_df = update_investment_returns(price, long_df, i, 'long')

                    if verbose: print('entered long investment')

                else:

                    if verbose: print('short TSL not triggered')

        else: # SMA has negative slope

            if verbose: print('\nSMA slope is negative')

            # if invested short
            if short_df.at[i, 'invested']:

                if verbose: print('invested short')

                # if short TSL triggered
                if short_df.at[i, 'triggered']:

                    if verbose: print('short TSL triggered')

                    # exit short investment
                    short_df = update_investment_returns(price, short_df, i, 'short')
                    invested_short = False

                    if verbose: print('exitted short investment')

                else: # else short TSL wasn't triggered

                    if verbose: print('short TSL not triggered')

                    # update short investment
                    short_df = update_investment_returns(price, short_df, i, 'short')

                    if verbose: print('short investment updated')

            # elif invested long (for corner case: SMA is still catching up to trend shift):
            elif long_df.at[i, 'invested']:

                if verbose: print('invested long: SMA is still catching up to trend shift')

                # if long TSL triggered:
                if long_df.at[i, 'triggered']:

                    if verbose: print('long TSL triggered')

                    # exit long investment
                    long_df = update_investment_returns(price, long_df, i, 'long')
                    invested_long = False

                    # enter short investment
                    invested_short = True
                    short_df.at[i, 'invested'] = invested_short
                    short_df = update_investment_returns(price, short_df, i, 'short')

                    if verbose: print('exitted long investment & entered short investment')

                else: # else long TSL not triggered

                    if verbose: print('long TSL not triggered')

                    long_df = update_investment_returns(price, long_df, i, 'long')

                    if verbose: print('updated long investment')

            # else not invested short or long
            else:

                if verbose: print('not invested short')

                # if long TSL is triggered
                if long_df.at[i, 'triggered']:

                    if verbose: print('long TSL triggered')

                    # enter a short investment
                    invested_short = True
                    short_df.at[i, 'invested'] = invested_short
                    short_df = update_investment_returns(price, short_df, i, 'short')

                    if verbose: print('entered short investment')

                else:

                    if verbose: print('long TSL not triggered')

        long_df  = update_portfolio_returns(long_df,  i)
        short_df = update_portfolio_returns(short_df, i)
        tot_returns_i = tot_returns[-1]
        if long_df.at[i, 'invested'] and long_df.at[i, 'triggered']:
            tot_returns_i *= 1.0 + long_df.loc[i, 'cur_sl_pl']
        elif short_df.at[i, 'invested'] and short_df.at[i, 'triggered']:
            tot_returns_i *= 1.0 + short_df.loc[i, 'cur_sl_pl']
        tot_returns.append(tot_returns_i)

        if verbose:
            print()
            input()

    # print('AFTER')
    # print('MAX_ROWS = %d' % MAX_ROWS)

    # print('\nlong df')
    # print(long_df)
    # print('\nshort_df')
    # print(short_df)
    # print('tot_returns')
    # print(pd.Series(tot_returns))
    # input()


    return {
        'long_df'     : long_df,
        'short_df'    : short_df,
        'tot_returns' : tot_returns
    }
def get_investments_helper4(coin, w, x, price_series, sma_w_df, tsl_x_dct, verbose=False, bp=BlockPrinter()):

    ''' NOTE:
        This investment helper switches back and forth between long and short TSLs.
        If SMA slope is positive it can only go long.
        If SMA slope is negative it can only go short.
        When the SMA slope switches, for example, postive to negative, if the
        algo is already tracking a short TSL, it will switch to tracking long
        and if the long TSL is triggered, a short TSL will start tracking
        WITH and investment as well.

        This requires the use of: get_tsl_dct_helper2().
        '''

    long_df, short_df = tsl_x_dct['long_df'], tsl_x_dct['short_df']
    invested_long, invested_short = False, False

    # init the 1st w 'invested' flags to false, and the 1st w 'tot_sl_pl' and 'tot_price_pl' to 0
    long_df['invested'][:w]      = False
    long_df['tot_sl_pl'][:w]     = 0.0
    long_df['tot_price_pl'][:w]  = 0.0
    short_df['invested'][:w]     = False
    short_df['tot_sl_pl'][:w]    = 0.0
    short_df['tot_price_pl'][:w] = 0.0

    # print('BEFORE')

    # print('\nlong df')
    # print(long_df)
    # print('\nshort_df')
    # print(short_df)
    # input()

    # print(price_series)

    # iterate over the price data but start at w-1+1
    # -1 b/c 0 indexing, +1 b/c we want to start when we have an SMA slope to work w/
    for i, price in price_series.iloc[w-1+1:].items():

        # set investment status at i equal to what it was at i-1
        long_df.at[i,  'invested'] = invested_long
        short_df.at[i, 'invested'] = invested_short
        # long_df.at[i,  'invested'] = long_df.at[i-1,  'invested']
        # short_df.at[i, 'invested'] = short_df.at[i-1, 'invested']

        if verbose:
            print('-' * 100)
            print('i = %d   price = %.4f\n' % (i, price))
            print(sma_w_df.iloc[i])
            print('\nlong_df.iloc[%d]' % i)
            print(long_df.iloc[i])
            print('\nshort_df.iloc[%d]' % i)
            print(short_df.iloc[i])
        else:
            bp.print('%.1f %%' % (100 * i / price_series.shape[0]))

        if sma_w_df.iloc[i]['sma_positive_slope']: # SMA has positive slope

            if verbose: print('\nSMA slope is positive')

            # if invested long
            if long_df.at[i, 'invested']:

                if verbose: print('invested long')

                # if long TSL triggered
                if long_df.at[i, 'triggered']:

                    if verbose: print('long TSL triggered')

                    # exit long investment
                    long_df = update_investment_returns(price, long_df, i, 'long')
                    invested_long = False

                    if verbose: print('exitted long investment')

                else: # else long TSL wasn't triggered

                    if verbose: print('long TSL not triggered')

                    # update long investment
                    long_df = update_investment_returns(price, long_df, i, 'long')

                    if verbose: print('long investment updated')

            # elif invested short (for corner case: SMA is still catching up to trend shift):
            elif short_df.at[i, 'invested']:

                if verbose: print('invested short: SMA is still catching up to trend shift')

                # if short TSL triggered:
                if short_df.at[i, 'triggered']:

                    if verbose: print('short TSL triggered')

                    # exit short investment
                    short_df = update_investment_returns(price, short_df, i, 'short')
                    invested_short = False

                    # enter long investment
                    invested_long = True
                    long_df.at[i, 'invested'] = invested_long
                    long_df = update_investment_returns(price, long_df, i, 'long')

                    if verbose: print('exitted short investment & entered long investment')

                else: # else short TSL not triggered

                    if verbose: print('short TSL not triggered')

                    short_df = update_investment_returns(price, short_df, i, 'short')

                    if verbose: print('updated short investment')

            # else not invested long or short
            else:

                if verbose: print('not invested long')

                # if short TSL is triggered
                if short_df.at[i, 'triggered']:

                    if verbose: print('short TSL triggered')

                    # enter a long investment
                    invested_long = True
                    long_df.at[i, 'invested'] = invested_long
                    long_df = update_investment_returns(price, long_df, i, 'long')

                    if verbose: print('entered long investment')

                else:

                    if verbose: print('short TSL not triggered')

        else: # SMA has negative slope

            if verbose: print('\nSMA slope is negative')

            # if invested short
            if short_df.at[i, 'invested']:

                if verbose: print('invested short')

                # if short TSL triggered
                if short_df.at[i, 'triggered']:

                    if verbose: print('short TSL triggered')

                    # exit short investment
                    short_df = update_investment_returns(price, short_df, i, 'short')
                    invested_short = False

                    if verbose: print('exitted short investment')

                else: # else short TSL wasn't triggered

                    if verbose: print('short TSL not triggered')

                    # update short investment
                    short_df = update_investment_returns(price, short_df, i, 'short')

                    if verbose: print('short investment updated')

            # elif invested long (for corner case: SMA is still catching up to trend shift):
            elif long_df.at[i, 'invested']:

                if verbose: print('invested long: SMA is still catching up to trend shift')

                # if long TSL triggered:
                if long_df.at[i, 'triggered']:

                    if verbose: print('long TSL triggered')

                    # exit long investment
                    long_df = update_investment_returns(price, long_df, i, 'long')
                    invested_long = False

                    # enter short investment
                    invested_short = True
                    short_df.at[i, 'invested'] = invested_short
                    short_df = update_investment_returns(price, short_df, i, 'short')

                    if verbose: print('exitted long investment & entered short investment')

                else: # else long TSL not triggered

                    if verbose: print('long TSL not triggered')

                    long_df = update_investment_returns(price, long_df, i, 'long')

                    if verbose: print('updated long investment')

            # else not invested short or long
            else:

                if verbose: print('not invested short')

                # if long TSL is triggered
                if long_df.at[i, 'triggered']:

                    if verbose: print('long TSL triggered')

                    # enter a short investment
                    invested_short = True
                    short_df.at[i, 'invested'] = invested_short
                    short_df = update_investment_returns(price, short_df, i, 'short')

                    if verbose: print('entered short investment')

                else:

                    if verbose: print('long TSL not triggered')

        long_df  = update_portfolio_returns(long_df,  i)
        short_df = update_portfolio_returns(short_df, i)

        if verbose:
            print()
            input()

    # print('AFTER')
    # print('MAX_ROWS = %d' % MAX_ROWS)

    # print('\nlong df')
    # print(long_df)
    # print('\nshort_df')
    # print(short_df)
    # # input()


    return {
        'long_df'  : long_df,
        'short_df' : short_df
    }
def update_investment_returns(price, df, i, investment_type):

    # update p/l variables

    tf = TF if INCLUDE_TF else 0
    l_or_s = 1 if investment_type == 'long' else -1 # l_or_s = long or short

    enter      = df.loc[i, 'enter_price'] * (1 + l_or_s * tf)
    sl_exit    = df.loc[i, 'stop_loss']   * (1 - l_or_s * tf)
    price_exit = price                    * (1 - l_or_s * tf)

    df.at[i, 'cur_sl_pl']    = l_or_s * (sl_exit    - enter) / enter
    df.at[i, 'cur_price_pl'] = l_or_s * (price_exit - enter) / enter

    return df
def update_portfolio_returns(df, i):

    df.at[i, 'tot_sl_pl'] = \
        df.loc[i-1, 'tot_sl_pl'] * \
        (1.0 + (df.loc[i, 'cur_sl_pl'] if \
            df.at[i, 'invested'] and df.at[i, 'triggered'] else 0.0))

    cur_price_pl = df.loc[i, 'cur_price_pl']
    df.at[i, 'tot_price_pl'] = \
        df.loc[i, 'tot_sl_pl'] + \
        (cur_price_pl if not np.isnan(cur_price_pl) else 0.0)

    return df
def plot_investment_data(dct):

    print('\nPlotting Results ...')

    # create date_labels and x_tick_indeces
    first_date = df['datetime'].iloc[0]
    date_fmt = '%m-%d-%Y'
    date_labels = [first_date.strftime(date_fmt)]
    x_tick_indeces = [0]
    previous_month = first_date.strftime('%m')
    for i, row in df.iterrows():
        current_month = row['datetime'].strftime('%m')
        if current_month != previous_month:
            date_labels.append(row['datetime'].strftime(date_fmt))
            x_tick_indeces.append(i)
        previous_month = current_month
    last_date = df['datetime'].iloc[-1]
    if last_date != date_labels[-1]:
        date_labels.append(last_date.strftime(date_fmt))
        x_tick_indeces.append(df['datetime'].tolist().index(last_date))
    # for i, l in zip(x_tick_indeces, date_labels):
    #     print(i, l)

    # plot each TSL
    for i, coin in enumerate(COINS):
        price_series = dct['asset_dct'][coin]['price_df']['price']
        for j, w in enumerate(SMA_WINDOWS):
            sma_w_dct = dct['asset_dct'][coin]['sma_dct'][w]
            for k, x in enumerate(TSL_VALUES):
                tsl_x_dct = sma_w_dct['tsl_dct'][x]
                skip_plotting = plot_investment_data_helper(
                    date_labels, x_tick_indeces, coin, price_series, sma_w_dct, x, tsl_x_dct)
                if skip_plotting:
                    return
def plot_investment_data_helper(date_labels, x_tick_indeces, coin, price_series, sma_w_dct, x, tsl_x_dct):

    long_x_str,  long_df  = '%.1f' % (100*x), tsl_x_dct['long_df']
    short_x_str, short_df = '%.1f' % (100*x), tsl_x_dct['short_df']

    sma_data = sma_w_dct['df']['sma']
    sma_lbl  = '%s' % sma_w_dct['sma_label']

    fig, ax = plt.subplots(
        nrows=3, ncols=1,
        num='asset = %s\tstop_loss = %s%%\tSMA window = %s' % (coin, long_x_str, sma_w_dct['sma_label'][:-5]),
        figsize=(10.75, 6.5),
        sharex=True, sharey=False)

    _legend_loc, _b2a = 'center left', (1, 0.5) # puts legend ouside plot

    ax[0].plot(price_series,          color='black', label='price')
    ax[0].plot(sma_data,              color='blue',  label=sma_lbl)
    ax[0].plot(long_df['stop_loss'],  color='green', label='%s%% long TSL ' % long_x_str)
    ax[0].plot(short_df['stop_loss'], color='red',   label='%s%% short TSL ' % short_x_str)
    ax[0].legend(loc=_legend_loc, bbox_to_anchor=_b2a)
    ax[0].grid()
    # ax[0].yaxis.grid()  # draw horizontal lines
    ax[0].yaxis.set_zorder(-1.0)  # draw horizontal lines behind histogram bars
    ax[0].set_title('Price, SMA, and TSL Chart')
    ax[0].set_xticks(x_tick_indeces)
    ax[0].set_xticklabels('')

    # highlight SMA slope + green
    # highlight SMA slope - red
    def highlight_sma_slope(up=True, color='green'):
        ranges_sma_direction = []
        range_start, range_end = None, None
        for index, value in sma_w_dct['df']['sma_positive_slope'].items():
            if value == up: # True
                if range_start != None:
                    pass # continue on ...
                else: # just starting
                    range_start = index # started new range
            else: # False
                if range_start != None: # found the end
                    range_end = index
                    ranges_sma_direction.append((range_start, range_end))
                    range_start, range_end = None, None
                else:
                    pass # continue on ...
        for range_start, range_end in ranges_sma_direction:
            ax[0].axvspan(range_start, range_end, color=color, alpha=0.5)
    highlight_sma_slope(up=True,  color='green')
    highlight_sma_slope(up=False, color='red')

    # write text explaining when SMA slope is positive and negative
    ax[0].text(
        1.10, 0.00,
        '+ SMA slope\n -  SMA slope',
        transform=ax[0].transAxes,
        fontsize=10)

    # place red and green rectangles next to that text
    ax[0].text(
        1.02, 0.12, '            ', # number of spaces controls rect width
        transform=ax[0].transAxes,
        bbox=dict(
            boxstyle='square',
            facecolor='green',
            alpha=0.5,
            edgecolor='none'
        ),
        fontsize=7) # fontsize controls rect height
    ax[0].text(
        1.02, 0.00, '            ',
        transform=ax[0].transAxes,
        bbox=dict(
            boxstyle='square',
            facecolor='red',
            alpha=0.5,
            edgecolor='none'
        ),
        fontsize=7)

    ''' Notes:
        cur_sl_pl was used instead of cur_price_pl because the sl will be triggered
        before the price reaches a value lower/higher than the stop loss
        '''
    def plot_vertical_lines(ax, df, color, label):
        for i, row in df.iterrows():
            if row['triggered']:
                ax.plot(
                    [i,   i],
                    [0.0, row['cur_sl_pl']],
                    color=color,
                    label=label)
    plot_vertical_lines(ax[1], long_df,  'green', 'Long Current Stop Loss P/L')
    plot_vertical_lines(ax[1], short_df, 'red',   'Short Current Stop Loss P/L')
    # ax[1].plot(long_df['cur_sl_pl'],  color='green', label='Long Current Stop Loss P/L')
    # ax[1].plot(short_df['cur_sl_pl'], color='red',   label='Short Current Stop Loss P/L')
    ax[1].plot(sma_w_dct['df']['sma'].diff().abs() / 100, color='blue', label='abs(SMA slope)')
    ax[1].legend(loc=_legend_loc, bbox_to_anchor=_b2a)
    ax[1].grid()
    # ax[1].yaxis.grid()  # draw horizontal lines
    # ax[1].yaxis.set_zorder(-1.0)  # draw horizontal lines behind histogram bars
    ax[1].set_title('Current TSL Profit/Loss')
    ax[1].set_xticks(x_tick_indeces)
    ax[1].set_xticklabels('')
    ax[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    ax[2].plot(long_df['tot_sl_pl'],     color='green', label='Total Long Stop Loss P/L')
    ax[2].plot(short_df['tot_sl_pl'],    color='red',   label='Total Short Stop Loss P/L')
    ax[2].plot(tsl_x_dct['tot_returns'], color='blue',  label='Total Combined Stop Loss P/L')
    ax[2].plot()
    ax[2].legend(loc=_legend_loc, bbox_to_anchor=_b2a)
    ax[2].grid()
    # ax[2].yaxis.grid()  # draw horizontal lines
    ax[2].yaxis.set_zorder(-1.0)  # draw horizontal lines behind histogram bars
    ax[2].set_title('Total TSL Profit/Loss')
    ax[2].set_xticks(x_tick_indeces)
    ax[2].set_xticklabels(date_labels, ha='right', rotation=45)  # x axis should show date_labeles
    ax[2].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    # plt.tight_layout()
    fig.subplots_adjust(
        right=0.75,
        left=0.075,
        bottom=0.15,
        top=0.95) # <-- Change the 0.02 to work for your plot.

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
    dct = get_sma_dct(dct, output_sma_data=False)
    # sma_w_dct = dct['asset_dct'][COINS[0]]['sma_dct'][SMA_WINDOWS[0]]
    # print(sma_w_dct['df']['sma'].diff().abs())
    # sys.exit()


    # calculate TSL data
    dct = get_tsl_dct(dct, output_tsl_data=False) \
            if CALCULATE_TSL_DATA else get_tsl_dct_from_csv_files(dct)

    # calculate investment p/l
    dct = get_investments(dct, output_investment_data=True)


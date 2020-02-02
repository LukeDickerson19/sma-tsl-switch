import time
import subprocess
import sys
sys.path.insert(0, './')
from poloniex import poloniex
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    
        for each coin in COINS:
            for each sma window length w in SMA_WINDOWS:
                for each TSL value (percent offset from current price) x in TSL_VALUES:
                    determine the p/l for each section of the SMA trend (when it is with it, and when it is against it)
                    plot 1: price, sma, trend, highlight when sign_of_sma_slope == sign_of_trend_slope, tsl_long, tsl_short
                    plot 2: current p/l of tsl_long and tsl_short
                    plot 3: total p/l of tsl_long, total p/l of tsl_short, net p/l

        Hopefully the profits made when the SMA is correct outway the losses when the SMA is wrong.
        I expect an x value significantly smaller than w will perform the best
        because we're trying to catch the volitility of a trend.

    IDEA:

        what if the distance the next timestep's value is from the previous timesteps moving average the more exponential the weights will become.
        A difference of 0 from the previous moving average will weight them equally for a simple moving average, but a large distance will make the
        weights very exponential. The equation for could be:

                w[i] = t[i] ^ (1 + d)

                d = absolute value of percentage distance of current timestep's price from previous timestep's moving average 
                        this value will vary from 0.00 to 1.00

            ... this could yeild an SMA that doesn't lag !!!

            ... or not, the lag is what makes it useful in the first place because it smooths out the smaller scale volitility

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
SMA_WINDOWS = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99] # most accurate if they're all odd integers
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

# get TSL's data
def get_tsl_dct_from_price_iteration(coin, df, w, tsl_vals):

    columns={
        'enter_price',   # the price that the trade was entered at
        'stop_loss',     # the price that it will take to exit the trade
        'dxv',           # dxv = dx VALUE (not the percentage), aka difference between enter_price and stop_loss
        'cur_price_pl',  # profit/loss (pl) of the current actual price
        'cur_sl_pl',     # profit/loss (pl) of the current stop loss
        'tot_price_pl',  # profit/loss (pl) of the total (from beginning until now) of the current price
        'tot_sl_pl'      # profit/loss (pl) of the total (from beginning until now) of the current stop loss
    }
    dct = {
        'price': df['price'],
        'long' : {
            '%.2f%%' % (100*x) : pd.DataFrame(columns=columns)
            for x in tsl_vals},
        'short' : {
            '%.2f%%' % (100*x) : pd.DataFrame(columns=columns)
            for x in tsl_vals}
    }

    N = len(dct['price'])
    for i, price in dct['price'].iteritems():

        print('---------------- i = %d / %d ------- price = %f -----------------' % (i, N, price))

        # long tsl
        for x, (x_str, df) in zip(tsl_vals, dct['long'].items()):

            # update TSL variables
            sl = df.loc[i-1, 'stop_loss'] if i != 0 else None  # sl = (previous) stop loss
            dxv = df.loc[i-1, 'dxv'] if i != 0 else None
            triggered = False
            if i == 0 or price <= sl:  # stop loss triggered !!!!!!!!! D-: !!!! er mer gerd!!! merney!!!
                # print('stop loss triggered!')
                triggered = True
                dxv = df.at[i, 'dxv'] = price * x  # update dxv
                df.at[i, 'stop_loss'] = price - dxv # set new tsl
                df.at[i, 'enter_price'] = price # update new entrance price
            elif price <= sl + dxv and price > sl: # else it stayed within its stop loss
                # print('price stayed within its stop loss')
                df.at[i, 'dxv'] = dxv
                df.at[i, 'stop_loss'] = sl  # keep the stop loss the same
                df.at[i, 'enter_price'] = df.loc[i-1, 'enter_price']
            elif price > sl + dxv: # else it went up, thus dragging the stop loss up also
                # print('price went up')
                df.at[i, 'dxv'] = dxv
                df.at[i, 'stop_loss'] = price - dxv
                df.at[i, 'enter_price'] = df.loc[i-1, 'enter_price']

            # update pl variables
            enter_price = df.loc[i, 'enter_price']
            cpt = (enter_price + df.loc[i, 'stop_loss']) * TF # cpt = cost per trade (both enter and exit)
            enter, exit = enter_price * (1 + TF), df.loc[i, 'stop_loss'] * (1 - TF)
            df.at[i, 'cur_sl_pl'] = (exit - enter) / enter
            enter, exit = enter_price * (1 + TF), price * (1 - TF)
            df.at[i, 'cur_price_pl'] = (exit - enter) / enter
            df.at[i, 'tot_sl_pl'] = \
                (df.loc[i-1, 'tot_sl_pl'] if i != 0 else 0.0) + \
                (df.loc[i-1, 'cur_sl_pl'] if triggered and i != 0 else 0.0)
                # don't forget, when i=0, triggered=True
            df.at[i, 'tot_price_pl'] = df.loc[i, 'tot_sl_pl'] + df.loc[i, 'cur_price_pl']

            # if x == 0.05:
            #     print('x = %s' % x)
            #     print('x_str = %s' % x_str)
            #     print('sl = %s' % sl)
            #     print('dxv = %s' % dxv)
            #     print('df:')
            #     print(df)
            #     input()

        # short tsl
        for x, (x_str, df) in zip(tsl_vals, dct['short'].items()):

            # update TSL variables
            sl = df.loc[i-1, 'stop_loss'] if i != 0 else None  # sl = (previous) stop loss
            dxv = df.loc[i-1, 'dxv'] if i != 0 else None
            triggered = False
            if i == 0 or price >= sl:  # stop loss triggered !!!!!!!!! D-: !!!! er mer gerd!!! merney!!!
                # print('stop loss triggered!')
                triggered = True
                dxv = df.at[i, 'dxv'] = price * x  # update dxv
                df.at[i, 'stop_loss'] = price + dxv # set new tsl
                df.at[i, 'enter_price'] = price # update new entrance price
            elif price >= sl - dxv and price < sl: # else it stayed within its stop loss
                # print('price stayed within its stop loss')
                df.at[i, 'dxv'] = dxv
                df.at[i, 'stop_loss'] = sl  # keep the stop loss the same
                df.at[i, 'enter_price'] = df.loc[i-1, 'enter_price']
            elif price < sl - dxv: # else it went down, thus dragging the stop loss down also
                # print('price went down')
                df.at[i, 'dxv'] = dxv
                df.at[i, 'stop_loss'] = price + dxv
                df.at[i, 'enter_price'] = df.loc[i-1, 'enter_price']

            # update pl variables
            enter_price = df.loc[i, 'enter_price']
            # df.at[i, 'cur_sl_pl'] = (enter_price - df.loc[i, 'stop_loss']) / enter_price
            # df.at[i, 'cur_price_pl'] = (enter_price - price) / enter_price
            exit, enter = enter_price * (1 - TF), df.loc[i, 'stop_loss'] * (1 + TF)
            df.at[i, 'cur_sl_pl'] = (exit - enter) / exit
            exit, enter = enter_price * (1 - TF), price * (1 + TF)
            df.at[i, 'cur_price_pl'] = (exit - enter) / exit


            df.at[i, 'tot_sl_pl'] = \
                (df.loc[i-1, 'tot_sl_pl'] if i != 0 else 0.0) + \
                (df.loc[i-1, 'cur_sl_pl'] if triggered and i != 0 else 0.0)
                # don't forget, when i=0, triggered=True
            df.at[i, 'tot_price_pl'] = df.loc[i, 'tot_sl_pl'] + df.loc[i, 'cur_price_pl']



            # if x == 0.05:
            #     print('x = %s' % x)
            #     print('x_str = %s' % x_str)
            #     print('sl = %s' % sl)
            #     print('dxv = %s' % dxv)
            #     print('df:')
            #     print(df)
            #     input()

    # save dct to csv files
    # ./data/
    #     price_data.csv
    #     long/coin/
    #            long<x0>.csv
    #            long<x1>.csv
    #            long<x2>.csv
    #         ...
    #     short/coin/
    #            short<x0>.csv
    #            short<x1>.csv
    #            short<x2>.csv
    #         ...
    subprocess.run(['rm', '-rf', '%s*' % LONG_TSLS_FILES + coin + '/'])
    subprocess.run(['mkdir', '-p', './%s/%s/' % (LONG_TSLS_FILES, coin)])
    for x, df in zip(tsl_vals, dct['long'].values()):
        df.to_csv(LONG_TSLS_FILES + coin + '/long%.2f.csv' % (100*x))
    subprocess.run(['rm', '-rf', '%s*' % SHORT_TSLS_FILES + coin + '/'])
    subprocess.run(['mkdir', '-p', './%s/%s/' % (SHORT_TSLS_FILES, coin)])
    for x, df in zip(tsl_vals, dct['short'].values()):
        df.to_csv(SHORT_TSLS_FILES + coin + '/short%.2f.csv' % (100*x))

    return dct
def get_tsl_dct_from_csv_files():
    dct = {
        'price' : get_past_prices_from_csv_file()['price'],
        'long'  : {
            '%.2f%%' % (100*x) :
                pd.read_csv(
                    LONG_TSLS_FILES + 'long%.2f.csv' % (100*x),
                    index_col=[0])
            for x in tsl_vals},
        'short' : {
            '%.2f%%' % (100*x) :
                pd.read_csv(
                    SHORT_TSLS_FILES + 'short%.2f.csv' % (100*x),
                    index_col=[0])
            for x in tsl_vals}
    }
    return dct

# display diagrams of price, TSL, and profit/loss data
def plot_tsl_dct(tsl_vals, dct, df):

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

    # for position_type in ['long', 'short']:
    #     for x, (x_str, df) in zip(tsl_vals, dct[position_type].items()):
    #         plot_df1(dct['price'], position_type, x_str, df, date_labels, x_tick_indeces)
    #         break
    for x, (long_x_str, long_df), (short_x_str, short_df) in \
    zip(tsl_vals, dct['long'].items(), dct['short'].items()):
        plot_df(
            dct['price'],
            long_x_str,  long_df,
            short_x_str, short_df,
            date_labels, x_tick_indeces)
        # break
def plot_df(
    prices,
    long_x_str, long_df,
    short_x_str, short_df,
    date_labels, x_tick_indeces):

    fig, ax = plt.subplots(
        nrows=3, ncols=1,
        num='stop loss = %s' % (long_x_str),
        figsize=(10.5, 6.5),
        sharex=True, sharey=False)

    _legend_loc, _b2a = 'center left', (1, 0.5) # puts legend ouside plot
    
    ax[0].plot(prices,                color='black', label='price')
    ax[0].plot(long_df['stop_loss'],  color='green', label='long stop loss')
    ax[0].plot(short_df['stop_loss'], color='red',   label='short stop loss')
    ax[0].legend(loc=_legend_loc, bbox_to_anchor=_b2a)
    ax[0].grid()
    # ax[0].yaxis.grid()  # draw horizontal lines
    ax[0].yaxis.set_zorder(-1.0)  # draw horizontal lines behind histogram bars
    ax[0].set_title('Price and TSLs')
    ax[0].set_xticks(x_tick_indeces)
    ax[0].set_xticklabels('')

    ax[1].plot(long_df['cur_price_pl'],  color='green', label='Long Current Stop Loss P/L')
    ax[1].plot(short_df['cur_price_pl'], color='red',   label='Short Current Stop Loss P/L')
    ax[1].legend(loc=_legend_loc, bbox_to_anchor=_b2a)
    ax[1].grid()
    # ax[1].yaxis.grid()  # draw horizontal lines
    # ax[1].yaxis.set_zorder(-1.0)  # draw horizontal lines behind histogram bars
    ax[1].set_title('Current TSL Profit/Loss')
    ax[1].set_xticks(x_tick_indeces)
    ax[1].set_xticklabels('')
    ax[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    ax[2].plot(long_df['tot_sl_pl'],  color='green', label='Total Long Stop Loss P/L')
    ax[2].plot(short_df['tot_sl_pl'], color='red',   label='Total Short Stop Loss P/L')
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

    # convert datetime column from string to datetime object
    df['datetime'] = df['datetime'].apply(
        lambda s : datetime.strptime(s, '%Y-%m-%d %H:%M:%S'))

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

            # calculate percent of the time the SMA is witrh the trend
            pct_with_trend = df['sma_with_trend'].value_counts(normalize=True).loc[True] * 100
            pct_with_trend_df.at[w, coin] = pct_with_trend
            total_ave_pct_with_trend += pct_with_trend


            # do tsl strategy
            tsl_dct = get_tsl_dct_from_price_iteration(coin, df, w, TSL_VALUES)
            plot_tsl_dct(TSL_VALUES, tsl_dct, df)

            print('endddd')
            sys.exit()

            # # plot the price, SMA and trend,
            # # and highlight the regions where the sma is with the trend
            # # and create a legend labeling everything and outputting the
            # # percentage of the time the SMA is with the trend            
            # if p:
            #     fig, axes = plt.subplots(figsize=(11, 6))
            #     axes.plot(df['price'],  c='black', label='price')
            #     axes.plot(df[col1_lbl], c='blue',  label=col1_lbl)
            #     axes.plot(df[col2_lbl], c='cyan', label=col2_lbl)
            #     ranges_sma_with_trend = []
            #     range_start, range_end = None, None
            #     for index, value in df['sma_with_trend'].items():
            #         if value: # True
            #             if range_start != None:
            #                 pass # continue on ...
            #             else: # just starting
            #                 range_start = index # started new range
            #         else: # False
            #             if range_start != None: # found the end
            #                 range_end = index
            #                 ranges_sma_with_trend.append((range_start, range_end))
            #                 range_start, range_end = None, None
            #             else:
            #                 pass # continue on ... 
            #     for range_start, range_end in ranges_sma_with_trend:
            #         plt.axvspan(range_start, range_end, color='gray', alpha=0.5)
            #     axes.title.set_text(
            #         '%s/%s %s is w/ the %s %.2f %% of the time' % (
            #             coin, TETHER, col1_lbl, col2_lbl, pct_with_trend))                
            #     plt.legend(loc=(1.02, 0.40))
            #     # adjust subplots and display it
            #     ''' https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
            #     subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
            #         left  = 0.125  # the left side of the subplots of the figure                      percentage
            #         right = 0.9    # the right side of the subplots of the figure                     percentage
            #         bottom = 0.1   # the bottom of the subplots of the figure                         percentage
            #         top = 0.9      # the top of the subplots of the figure                            percentage
            #         wspace = 0.2   # the amount of width reserved for blank space between subplots    number
            #         hspace = 0.2   # the amount of height reserved for white space between subplots   number
            #         '''
            #     plt.subplots_adjust(
            #         left   = 0.10,
            #         right  = 0.85,
            #         bottom = 0.10,
            #         wspace = 0.25, hspace=0.5)

            #     plt.show()
            #     user_input = input('Press s to skip to end of test, or any other key to continue: ')
            #     p = not (user_input == 's' or user_input == 'S')


    print('\nPercent of the time the SMA is with the trend:\n')
    print(pct_with_trend_df)

    total_ave_pct_with_trend /= (len(COINS) * len(SMA_WINDOWS))
    print('\nTotal average percent = %.2f %%\n' % total_ave_pct_with_trend)


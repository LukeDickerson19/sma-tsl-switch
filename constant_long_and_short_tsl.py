import time
import sys
# sys.path.insert(0, './')
import subprocess
from poloniex import poloniex
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
pd.set_option('display.max_rows', 8)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1200)
import numpy as np

''' NOTES:


    TO DO:

        NOW:

            it might be useful to plot cur_sl_pl for all tsls on the same chart
            to see what is profiting and what is taking a loss at what times? ... hmmmmMMMMMM?????!!!!!
                the larger x values get to higher %cur_pl when the price moves their way
                but is the %tot_pl bigger for smaller x values when the price moves their way????
                    whichever one is biggest, just do a simple thing initially where it decides between
                    1 long and 1 short x value and switches between them based on some ........ thing

            maybe plot long stuff on the left and its synonomous short stuff on the right in a 2x3 grid

            make verticle lines on chart the points where the tsl was triggered

        BACK-BURNER:

            probably need to include the trading fee in order to make the current and total profit/loss charts useful at all
                see def get_tsl_dct_from_price_iteration(prices, Xs):
                    i think i did it right, might want to double check though


    DESCRIPTION:

        This script displays multiple TSLs of varying percentages
        over COIN2 vs COIN1(tether) historical price data with
        matplotlib to see if there is a way to manage which TSLs
        get what percent of the portfolio at what time to hopefully
        yield consistent profits statistically.
        It uses both long and short TSLs.

            initial questions:
                what do we do on a trend?
                what do we do on a transition between trends?
                what do we do on all x levels?


    POSSIBLE STRATEGIES:

        S1:

            whichever tsl has a greater current profit
                put money there
            if it switches, and the other is now profitting more
                switch also

        S2:

            start with a very small investment in both tsl
            when one of them starts to make profit
                (increase the investment in it)
                keep increasing the investment as it continues to profit
                eventually 1 of 2 things will happen
                    1: the tsl will be triggered
                        in which case all but the last investment will make profit
                        gotta make sure the profits from the investment before outweigh the loss of that last
                        basically the last 2 investments will cancel eachother out
                    2: there will be nothing more to invest
                        in which case all of them profitted.

        S3:
            Whichever one has as greater SMA slope of the total profits, do that one


    SIDE NOTES:

        if this works, look into doing it on robin hood with options for the .02 trades
            collateral and what not might b
            NO DAY TRADING on Robinhood :(

        when the price slope is steap, use a small x
            small x catches those gains
        when the price slope is shallow, use a larger x (not super large though, just like a medium one)
            medium x doesn't get caught on bs

        what if you plotted a trailing window of the sum of all the upward movements (and likewise for down)


    '''

# constants
QUERI_POLONIEX = False
MAKE_DCT = False # True when you want to iterate over the price data and make the stop losses, False when you want the df made from the csv files
PRICE_DATA_FILE = './data/price_data.csv'
LONG_TSLS_FILES = './data/long/'
SHORT_TSLS_FILES = './data/short/'
COIN1 = 'USDT'
COIN2 = 'BTC'
PAIR = COIN1 + '_' + COIN2
TF = 0.0025 # TF = trading fee


# setup connection to servers
def poloniex_server():

    # private.mail285@gmail.com
    API_KEY = '...'
    SECRET_KEY = '...'

    return poloniex(API_KEY, SECRET_KEY)


# get backtesting price data
def get_past_prices_from_poloniex(
    startTime, endTime, period, num_periods, conn):

    # get history data from startTime to endTime
    startTime_unix = time.mktime(startTime.timetuple())
    endTime_unix = time.mktime(endTime.timetuple())

    # get history data of this currency into the dictionary
    prices = conn.api_query("returnChartData", {
            'currencyPair': PAIR,
            'start': startTime_unix,
            'end': endTime_unix,
            'period': period
        })

    prices2 = []
    for t in num_periods:  # remove unneeded data
        price = prices[t]['close']
        prices2.append({'date': prices[t]['date'], 'price': price})

    prices3 = pd.DataFrame(prices2)
    prices3.to_csv(PRICE_DATA_FILE)

    return prices3
def get_past_prices_from_csv_file():

    return pd.read_csv(PRICE_DATA_FILE, index_col=[0])

# get TSL's data
def get_tsl_dct_from_price_iteration(prices, Xs):

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
        'price': prices['price'],
        'long' : {
            '%.2f%%' % (100*x) : pd.DataFrame(columns=columns)
            for x in Xs},
        'short' : {
            '%.2f%%' % (100*x) : pd.DataFrame(columns=columns)
            for x in Xs}
    }

    N = len(dct['price'])
    for i, price in dct['price'].iteritems():

        print('---------------- i = %d / %d ------- price = %f -----------------' % (i, N, price))

        # long tsl
        for x, (x_str, df) in zip(Xs, dct['long'].items()):

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
        for x, (x_str, df) in zip(Xs, dct['short'].items()):

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
    #     long/
    #         long<x0>.csv
    #         long<x1>.csv
    #         long<x2>.csv
    #         ...
    #     short/
    #         short<x0>.csv
    #         short<x1>.csv
    #         short<x2>.csv
    #         ...
    subprocess.run(['rm', '-rf', '%s*' % LONG_TSLS_FILES])
    for x, df in zip(Xs, dct['long'].values()):
        df.to_csv(LONG_TSLS_FILES + 'long%.2f.csv' % (100*x))
    subprocess.run(['rm', '-rf', '%s*' % SHORT_TSLS_FILES])
    for x, df in zip(Xs, dct['short'].values()):
        df.to_csv(SHORT_TSLS_FILES + 'short%.2f.csv' % (100*x))

    return dct
def get_tsl_dct_from_csv_files():
    dct = {
        'price' : get_past_prices_from_csv_file()['price'],
        'long'  : {
            '%.2f%%' % (100*x) :
                pd.read_csv(
                    LONG_TSLS_FILES + 'long%.2f.csv' % (100*x),
                    index_col=[0])
            for x in Xs},
        'short' : {
            '%.2f%%' % (100*x) :
                pd.read_csv(
                    SHORT_TSLS_FILES + 'short%.2f.csv' % (100*x),
                    index_col=[0])
            for x in Xs}
    }
    return dct

# display diagrams of price, TSL, and profit/loss data
def plot_all_dfs(Xs, dct, prices):

    # create date_labels and x_tick_indeces
    first_date = prices['date'].iloc[0]
    date_fmt = '%m-%d-%Y'
    date_labels = [first_date.strftime(date_fmt)]
    x_tick_indeces = [0]
    previous_month = first_date.strftime('%m')
    for i, row in prices.iterrows():
        current_month = row['date'].strftime('%m')
        if current_month != previous_month:
            date_labels.append(row['date'].strftime(date_fmt))
            x_tick_indeces.append(i)
        previous_month = current_month
    last_date = prices['date'].iloc[-1]
    if last_date != date_labels[-1]:
        date_labels.append(last_date.strftime(date_fmt))
        x_tick_indeces.append(prices['date'].tolist().index(last_date))
    # for i, l in zip(x_tick_indeces, date_labels):
    #     print(i, l)

    # for position_type in ['long', 'short']:
    #     for x, (x_str, df) in zip(Xs, dct[position_type].items()):
    #         plot_df1(dct['price'], position_type, x_str, df, date_labels, x_tick_indeces)
    #         break
    for x, (long_x_str, long_df), (short_x_str, short_df) in \
    zip(Xs, dct['long'].items(), dct['short'].items()):
        plot_df2(
            dct['price'],
            long_x_str,  long_df,
            short_x_str, short_df,
            date_labels, x_tick_indeces)
        # break
def plot_df1(prices, position_type, x_str, df, date_labels, x_tick_indeces):

    fig, ax = plt.subplots(nrows=3, ncols=1,
        num='%s: stop loss = %s' % (position_type, x_str), figsize=(14, 8))

    _legend_loc, _b2a = 'center left', (1, 0.5) # puts legend ouside plot

    ax[0].plot(prices, label='price')
    ax[0].plot(df['stop_loss'], label='stop loss')
    ax[0].legend(loc=_legend_loc, bbox_to_anchor=_b2a)
    ax[0].grid()
    # ax[0].yaxis.grid()  # draw horizontal lines
    ax[0].yaxis.set_zorder(-1.0)  # draw horizontal lines behind histogram bars
    ax[0].set_title('Price and TSL')
    ax[0].set_xticks(x_tick_indeces)
    ax[0].set_xticklabels('')

    ax[1].plot(df['cur_price_pl'], label='Current Price P/L')
    ax[1].plot(df['cur_sl_pl'], label='Current Stop Loss P/L')
    ax[1].legend(loc=_legend_loc, bbox_to_anchor=_b2a)
    ax[1].grid()
    # ax[1].yaxis.grid()  # draw horizontal lines
    # ax[1].yaxis.set_zorder(-1.0)  # draw horizontal lines behind histogram bars
    ax[1].set_title('Current TSL Profit and Price Profit')
    ax[1].set_xticks(x_tick_indeces)
    ax[1].set_xticklabels('')
    ax[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    ax[2].plot(df['tot_price_pl'], label='Total Price P/L')
    ax[2].plot(df['tot_sl_pl'], label='Total Stop Loss P/L')
    ax[2].legend(loc=_legend_loc, bbox_to_anchor=_b2a)
    ax[2].grid()
    # ax[2].yaxis.grid()  # draw horizontal lines
    ax[2].yaxis.set_zorder(-1.0)  # draw horizontal lines behind histogram bars
    ax[2].set_title('Total TSL Profit and Price Profit')
    ax[2].set_xticks(x_tick_indeces)
    ax[2].set_xticklabels(date_labels, ha='right', rotation=45)  # x axis should show date_labeles
    ax[2].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))


    plt.tight_layout()
    plt.show()
def plot_df2(
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

    # variables. ############
    startTime = datetime(2018, 3,  1, 0, 0, 0)  # year, month, day, hour, minute, second
    endTime   = datetime(2019, 5, 30, 0, 0, 0)
    # period = duration of time steps between rebalances
    #   300 s   900 s    1800 s   7200 s   14400 s   86400 s
    #   5 min   15 min   30 min   2 hrs    4 hrs     1 day
    period = 2 * 60 * 60  # duration of intervals between updates

    # determines the proper number of time steps from startTime to endTime for the given period
    num_periods = range(int((endTime - startTime).total_seconds() / period))

    # get backtest price data of COIN1 and COIN2 pair
    prices = \
        get_past_prices_from_poloniex(startTime, endTime, period, num_periods, poloniex_server()) \
        if QUERI_POLONIEX else get_past_prices_from_csv_file()
    # convert 'date' column from unix timestamp to datetime
    prices['date'] = prices['date'].apply(
        lambda unix_timestamp : datetime.fromtimestamp(unix_timestamp))
    # plt.plot(prices['price'])
    # plt.title('%s PriceChart' % PAIR)
    # plt.ylabel('Price')
    # plt.xlabel('Time')
    # plt.show()

    # get TSLs data
    # min_x, max_x, dx = 0.0025, 0.05, 0.0025
    # Xs = list(range(min_x, max_x, dx))
    Xs = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20]
    dct = get_tsl_dct_from_price_iteration(prices, Xs) \
        if MAKE_DCT else get_tsl_dct_from_csv_files()

    # display charts
    plot_all_dfs(Xs, dct, prices)



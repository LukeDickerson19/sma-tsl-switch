# SMA TSL SWITCH

#### Description:

Switch back and forth between a long and short TSL (Trailing Stop Loss), just tracking not investing. When SMA is going up, invest in the long TSL, and when SMA is going down invest in the short TSL. This strategy is very similar to the Parabolic SAR trading strategy, but it uses the SMA slope to attempt to avoid the situations where the SAR takes a loss.

https://www.investopedia.com/ask/answers/06/parabolicsar.asp


#### Results:

The backtest results say the strategy has a 3x annual return trading BTC/USDT with a TSL value of 1.0 % and an SMA window of 100 2 hr timesteps. The other cryptos have similar returns. I think this is too good to be true, but I've been looking into the way I'm calculating profits and it seems to be accurate. It currently calculates profits with percentages. Next steps are to redo the profit calculations with fake USDT instead of percentages to see if it changes the annual profits.


![results](https://github.com/PopeyedLocket/sma-tsl-switch/blob/master/images/asset-BTC_x-1_w-100.png?raw=true "Results")



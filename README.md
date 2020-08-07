# financial_scripts
This repository contains various backtesting, and other financial software which might be useful if you would like to test a stock/option/cryptocurrency trading algorithms.

# option_backtest_module.py

This script is used to test an option trading strategy which is based on buying an option closest to a given moneyness, at specific time of the day, day of week, and week of expiration, VIX level threshold, etc. 

The module calculates following statistics:
1. The dates and prices for the buy and sell side for each position
2. The number of trades that were taken
3. Number of winning trades
4. Number of losing trades
5. The profit or loss per each position
6. The max number of consecutive profitable trades
7. The max number of consecutive losing trades
8. The max profit on a single trade
9. The max loss on a single trade
10. The percentage of profitable trades
11. Average daily gain
12. Average daily loss
13. Average hold time per trade
14. Average hold time per winning trade
15. Average hold time per losing trade
16. Total gain or loss of the strategy over the specified time the strategy was run


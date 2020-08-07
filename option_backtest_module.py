import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        function_name = f.__name__
        end = time()
        print(f'Elapsed time for {function_name}: {end-start}')
        return result

    return wrapper


"""This module is used in order to test various strategies based on option
trading. 

Example:
    backtest = BacktestEngine(
        ticker='amzn',
        start_date='12.01.2016',
        end_date='29.03.2021',
        moneyness=10,
        sell_at_gain=0.05,
        sell_at_loss=0.05,
        expiration_week=0,
        vix_threshold=0,
        time_to_execute="10:00",
        option_type='call',
        week_days_trading=(0, 1, 2, 3, 4, 5, 6, 7),
        option_lot_size=100,
        path=r'D:/Soft/python/test'
    )

backtest.run_backtest()


"""


class StatisticsEngine:
    """This class is used in order to make plots for
    visualization of required results, record of statistics,
    calculation of various metrics and
    other required analysis"""

    def __init__(self, make_plot: bool = True):
        self.positions = ...
        self.result = {}
        self.make_plot = make_plot

    def __get__(self, instance, owner):
        if not self.result:
            self.result['number_of_positions'] = self.amount_of_positions(value=self.positions)
            self.result['number_of_profitable_positions'] = self.amount_profit_positions(value=self.positions)
            self.result['number_of_losing_positions'] = self.amount_loss_positions(value=self.positions)
            self.result['max_sequential_profit_positions'] = self.max_sequential_profit_positions(value=self.positions)
            self.result['max_sequential_losing_positions'] = self.max_sequential_loss_positions(value=self.positions)
            self.result['max_profit_by_position'] = self.max_profit_by_position(value=self.positions)
            self.result['max_loss_by_position'] = self.max_loss_by_position(value=self.positions)
            self.result['percent_of_profitable_positions'] = self.percent_profit_positions(value=self.positions)
            self.result['average_daily_profit'] = self.average_daily_profit(value=self.positions)
            self.result['average_daily_loss'] = self.average_daily_loss(value=self.positions)
            self.result['average_time_of_position'] = self.average_expectation_position(value=self.positions)
            self.result['average_time_of_profit_position'] = self.average_expectation_profit_position(
                value=self.positions)
            self.result['average_time_of_losing_position'] = self.average_expectation_loss_position(
                value=self.positions)
            self.result['profit_loss_of_strategy'] = self.profit_loss_strategy(value=self.positions)

        profit = np.array(self.positions['Profit'])
        result = pd.DataFrame.from_dict(self.result, orient='index')

        return result, profit

    def __set__(self, instance, value):
        data_for_stat = pd.DataFrame.from_dict(value)
        data_for_stat.index = data_for_stat.pop('Date')
        data_for_stat['days'] = [pd.to_datetime(data_for_stat.index[i]).date() for i in range(data_for_stat.shape[0])]
        self.positions = data_for_stat[['Profit', 'Transaction_id', 'days']]

    @staticmethod
    def amount_of_positions(value):
        value = np.array(value['Profit'])
        return sum(value != 0)

    @staticmethod
    def amount_profit_positions(value):
        value = np.array(value['Profit'])
        return sum(value > 0)

    @staticmethod
    def amount_loss_positions(value):
        value = np.array(value['Profit'])
        return sum(value < 0)

    @staticmethod
    def max_sequential_profit_positions(value):
        value = np.array(value['Profit'])
        value = value[value != 0]
        current_sequence = 0
        best_sequence = 0
        for result in value:
            if result > 0:
                current_sequence += 1
                if current_sequence > best_sequence:
                    best_sequence = current_sequence
            else:
                current_sequence = 0
        return best_sequence

    @staticmethod
    def max_sequential_loss_positions(value):
        value = np.array(value['Profit'])
        value = value[value != 0]
        current_sequence = 0
        best_sequence = 0
        for result in value:
            if result < 0:
                current_sequence += 1
                if current_sequence > best_sequence:
                    best_sequence = current_sequence
            else:
                current_sequence = 0
        return best_sequence

    @staticmethod
    def max_profit_by_position(value):
        value = np.array(value['Profit'])
        return max(value)

    @staticmethod
    def max_loss_by_position(value):
        value = np.array(value['Profit'])
        return min(value)

    @staticmethod
    def percent_profit_positions(value):
        value = np.array(value['Profit'])
        return sum(value > 0) / sum(value != 0) * 100

    @staticmethod
    def average_daily_profit(value):
        return value[value['Profit'] > 0].groupby('days').sum()['Profit'].mean()

    @staticmethod
    def average_daily_loss(value):
        return value[value['Profit'] < 0].groupby('days').sum()['Profit'].mean()

    @staticmethod
    def average_expectation_position(value):
        transaction_id = np.array(value['Transaction_id'])
        expectation_position = []
        counter = 0

        for i in range(1, len(transaction_id)):
            if transaction_id[i] != 0 and transaction_id[i] == transaction_id[i - 1]:
                counter += 1
            elif counter != 0:
                expectation_position.append(counter)
                counter = 0

        return np.mean(expectation_position)

    @staticmethod
    def average_expectation_profit_position(value):
        transaction_id = np.array(value['Transaction_id'])
        profit = np.array(value['Profit'])
        expectation_position = []
        counter = 0

        for i in range(1, len(transaction_id)):
            if transaction_id[i] != 0 and transaction_id[i] == transaction_id[i - 1]:
                counter += 1
            elif counter != 0:
                if profit[i - 1] > 0:
                    expectation_position.append(counter)
                counter = 0

        return np.mean(expectation_position)

    @staticmethod
    def average_expectation_loss_position(value):
        transaction_id = np.array(value['Transaction_id'])
        profit = np.array(value['Profit'])
        expectation_position = []
        counter = 0

        for i in range(1, len(transaction_id)):
            if transaction_id[i] != 0 and transaction_id[i] == transaction_id[i - 1]:
                counter += 1
            elif counter != 0:
                if profit[i - 1] < 0:
                    expectation_position.append(counter)
                counter = 0

        return np.mean(expectation_position)

    @staticmethod
    def profit_loss_strategy(value):
        return np.sum(value['Profit'])


class BacktestEngine:
    strat_statistics = StatisticsEngine()

    def __init__(
            self,
            ticker: str,
            start_date: str,
            end_date: str,
            moneyness: float,
            sell_at_gain: float,
            sell_at_loss: float,
            expiration_week: int = 0,
            vix_threshold: float = 0,
            time_to_execute: str = "06:00",
            option_type: str = 'call',
            week_days_trading: tuple = (0, 1, 2, 3, 4, 5, 6, 7),
            option_lot_size: int = 1,
            path: str = r'D:/'
    ):

        self.path = path
        self.ticker = ticker
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.moneyness = moneyness
        self.expiration_week = expiration_week
        self.vix_threshold = vix_threshold
        self.sell_at_gain = sell_at_gain
        self.sell_at_loss = sell_at_loss
        self.option_type = option_type[0].upper()
        self.week_days_trading = week_days_trading
        self.time_to_execute = pd.to_datetime(time_to_execute)
        self.option_lot_size = option_lot_size

        self.date_strings = None
        self.option_columns = None

        # transactions history
        self.trade_log = {'Profit': [],
                          'Position': [],
                          'Date': [],
                          'Strike': [],
                          'Transaction_id': [],
                          'ExpirationDate': [],
                          'PositionPrice': [],
                          'Current_option_close': [],
                          'Close_criterion': []}

        self.set_opt_date = set()
        self.list_opt_date = list()

        self.stock_data, self.option_data = self.load_trade_data()
        self.get_cut_stock_data()

        self.current_option = None
        self.position = False

        # positional statistics
        self.current_profit = 0
        self.position_price = 0
        self.open = None
        self.close = None
        self.vix = None
        self.trade_log_index = 0
        self.bought_option_strike = None
        self.bought_option_expiration = None
        self.opened_option_stats = None
        self.stock_data_i = None
        self.stock_data_row = None
        self.current_option_close = None

        self.statistics = None
        self.close_criterion = None

        self.matched_stock_date = None
        self.matched_stock_price = 0
        self.best_moneyness_delta = self.moneyness
        self.position_id = None
        self.current_transaction_num = 0
        self.current_result = 0

    def plot_strat_statistics(self):
        stats, profit = self.strat_statistics
        plt.plot(np.cumsum(profit), color='b')
        plt.title('Cumulative Sum strategy result')
        plt.xlabel('index')
        plt.ylabel('Cumulative Sum Value')
        plt.show()

    def get_cut_stock_data(self):
        max_date_time = max(self.option_data[:, self.option_columns['DateTime']])
        self.stock_data = self.stock_data[self.stock_data.index <= max_date_time]

    def save_strat_statistics(self):
        stats, _ = self.strat_statistics
        stats.to_csv(f'{self.path}/{self.ticker}_strat_statistics.csv')

    @timing
    def run_backtest(self) -> None:
        position_log = list()

        for stock_data_i, stock_data_row in self.stock_data.iterrows():
            self.stock_data_i = stock_data_i
            self.stock_data_row = stock_data_row

            if not self.position and self.stock_conditions_met(stock_data_i):
                self.matched_stock_date = stock_data_i
                self.matched_stock_price = stock_data_row['Close']
                matched_options = self.get_matched_options()

                if matched_options.size > 0:
                    self.position_id, self.opened_option_stats = self.get_best_option_index(matched_options)
                    if self.opened_option_stats.size > 0:
                        self.open_option()
                        position_log.append(1)

                self.record_algo_state()

            elif self.position and self.option_expired():
                self.close_expired_option()
                self.record_algo_state()
                self._reset_position_state()

            elif self.position and self.find_bought_option().size > 0:

                if self.check_stop_loss():
                    self.current_profit = -self.sell_at_loss * self.position_price * self.option_lot_size
                    self.record_algo_state()
                    self._reset_position_state()

                elif self.check_take_profit():
                    self.current_profit = self.sell_at_gain * self.position_price * self.option_lot_size
                    self.record_algo_state()
                    self._reset_position_state()

                else:
                    self.record_algo_state()

            else:
                self.record_algo_state()

        self.strat_statistics = self.trade_log
        self.save_trade_log()

    @timing
    def load_trade_data(self) -> tuple:

        stock_data = pd.read_csv(f'{self.path}/{self.ticker}.csv', parse_dates=['DateTime'])
        stock_data = stock_data.set_index('DateTime')
        stock_data = stock_data[stock_data.index <= self.end_date]
        stock_data = stock_data[stock_data.index >= self.start_date]

        option_data = pd.read_csv(f'{self.path}/{self.ticker}_option.csv', parse_dates=['DateTime', 'ExpirationDate'])
        self.option_columns = {k: v for k, v in zip(option_data.columns, range(len(option_data.columns)))}
        self.option_columns['ExpirationWeek'] = len(self.option_columns.keys())
        option_data = option_data[option_data['Type'] == self.option_type]

        option_data = np.array(option_data)
        option_data = self._find_expiration_time(option_data)
        option_data = self.get_expiration_week(option_data)

        self.list_opt_date = np.array([pd.to_datetime(x) for x in option_data[:, self.option_columns['DateTime']]])
        self.set_opt_date = set(self.list_opt_date)

        vix_data = pd.read_csv(f'{self.path}/VIX.csv',
                               parse_dates=['Date'])

        vix_data = vix_data.set_index('Date')
        vix_data = vix_data[['Close']]
        vix_data.columns = ['VIX']
        stock_data = self._merge_by_date(stock_data, vix_data)
        self.date_strings = {k: k.strftime('%m/%d/%Y %H:%M') for k in stock_data.index}

        return stock_data, option_data

    @timing
    def _find_expiration_time(self, option_data: np.ndarray):
        time_index = option_data[:, self.option_columns['DateTime']]
        strike = option_data[:, self.option_columns['Strike']].astype(str)
        exp_data = option_data[:, self.option_columns['ExpirationDate']]
        type_opt = option_data[:, self.option_columns['Type']]

        uniq_data = np.unique(option_data[:, [self.option_columns['Strike'],
                                              self.option_columns['ExpirationDate'],
                                              self.option_columns['Type']]].astype("<U22"), axis=0)
        u_strike = uniq_data[:, 0]
        u_exp_data = uniq_data[:, 1]
        u_type = uniq_data[:, 2]

        for i_strike, j_data, i_type in zip(u_strike, u_exp_data, u_type):
            locator = np.where((exp_data == pd.to_datetime(j_data)) & (strike == i_strike) & (type_opt == i_type))[0]
            locator_time = time_index[locator[-1]]
            option_data[locator, self.option_columns['ExpirationDate']] = locator_time

        return option_data

    @timing
    def option_expired(self):
        current_time = self.stock_data_i
        exp = self.bought_option_expiration

        return current_time >= exp

    @timing
    def get_expiration_week(self, option_data: np.ndarray) -> np.ndarray:
        expiration_week = []

        for row in option_data:
            current_date = row[self.option_columns['DateTime']]
            expiration_date = row[self.option_columns['ExpirationDate']]

            current_week = current_date.isocalendar()[1]
            exp_week = expiration_date.isocalendar()[1]

            expiration_week.append(exp_week-current_week)

        option_data = np.insert(option_data, option_data.shape[1], expiration_week, axis=1)

        return option_data

    @timing
    def stock_conditions_met(self, i) -> bool:
        date = i
        vix = self.stock_data_row['VIX']
        weekday = date.weekday()
        _time = date.time()

        required_date = True if self.start_date <= date < self.end_date else False
        required_vix = True if vix > self.vix_threshold else False
        required_day = True if weekday in self.week_days_trading else False
        required_time = True if self.time_to_execute.time() == _time else False

        return (required_date and
                required_time and
                required_day and
                required_vix)

    @timing
    def get_matched_options(self) -> np.ndarray:
        if self.matched_stock_date not in self.set_opt_date:
            matched_options = np.array([])
        else:
            matched_options = self.option_data[np.where(self.list_opt_date == self.matched_stock_date)[0]]

            matched_options = matched_options[np.where(matched_options[:, self.option_columns['ExpirationWeek']] ==
                                                       self.expiration_week)[0]]

        return matched_options

    @staticmethod
    def _merge_by_date(df1, df2, columns: tuple = ('VIX',)):

        for col in columns:
            to_merge = []
            for i in df1.index.date:
                try:
                    to_merge.append(df2.loc[i, col])
                except:
                    to_merge.append(0)
            df1[col] = to_merge

        return df1

    @staticmethod
    def check_stop_loss_call(current_low, current_high, stop_loss):
        if current_low <= stop_loss:
            return True

    @staticmethod
    def check_stop_loss_put(current_low, current_high, stop_loss):
        if current_high <= stop_loss:
            return True

    @timing
    def check_stop_loss_by_type(self, type_option: str):
        stop = {
            'C': self.check_stop_loss_call,
            'P': self.check_stop_loss_put
        }

        return stop[type_option]

    @timing
    def check_stop_loss(self):
        option = self.current_option
        current_low = option[0, self.option_columns['Low']]
        current_high = option[0, self.option_columns['High']]
        stop_loss = (1 - self.sell_at_loss) * self.position_price
        if self.check_stop_loss_by_type(self.option_type)(current_low, current_high, stop_loss):
            self.close_criterion = 'Stop_loss'
            return True

    @timing
    def find_bought_option(self):

        if self.stock_data_i in self.set_opt_date:
            current_option = self.option_data[
                np.where(self.list_opt_date == self.stock_data_i)]
            current_option = current_option[
                np.where(current_option[:, self.option_columns['Strike']] == self.bought_option_strike)]
            current_option = current_option[
                np.where(current_option[:, self.option_columns['ExpirationDate']] == self.bought_option_expiration)]
            self.current_option = current_option
            if self.current_option.size > 0:
                self.current_option_close = current_option[0, self.option_columns['Close']]
            else:
                self.current_option_close = None
        else:
            current_option = np.array([])

        return current_option

    @staticmethod
    def check_take_profit_call(current_low, current_high, take_profit):
        if current_high >= take_profit:
            return True

    @staticmethod
    def check_take_profit_put(current_low, current_high, take_profit):
        if current_low >= take_profit:
            return True

    @timing
    def check_take_profit_by_type(self, type_option: str):
        stop = {
            'C': self.check_take_profit_call,
            'P': self.check_take_profit_put
        }

        return stop[type_option]

    @timing
    def check_take_profit(self):
        option = self.current_option
        current_low = option[0, self.option_columns['Low']]
        current_high = option[0, self.option_columns['High']]
        take_profit = (self.sell_at_gain + 1) * self.position_price

        if self.check_take_profit_by_type(self.option_type)(current_low, current_high, take_profit):
            self.close_criterion = 'Take_profit'
            return True

    @timing
    def get_best_option_index(self, options: np.ndarray):

        best_moneyness_delta = np.inf
        best_index = np.array([])
        best_option = np.array([])
        current_moneyness = 0

        for row in options:
            if self.option_type == 'P':
                current_moneyness = row[self.option_columns['Strike']] - self.matched_stock_price
            elif self.option_type == 'C':
                current_moneyness = self.matched_stock_price - row[self.option_columns['Strike']]

            current_moneyness_delta = abs(self.moneyness - current_moneyness)

            if current_moneyness_delta < best_moneyness_delta and np.sign(current_moneyness) == np.sign(self.moneyness):
                best_moneyness_delta = current_moneyness_delta
                best_index = row[self.option_columns['DateTime']]
                best_option = row

        return best_index, best_option

    @timing
    def open_option(self) -> None:

        exp_date = self.opened_option_stats[self.option_columns['ExpirationDate']]

        if exp_date.date() > self.stock_data_i.date():
            self.position = True
            self.position_price = self.opened_option_stats[self.option_columns['Close']]
            self.bought_option_strike = self.opened_option_stats[self.option_columns['Strike']]
            self.bought_option_expiration = self.opened_option_stats[self.option_columns['ExpirationDate']]
            self.current_profit = 0
            self.current_transaction_num += 1

    @timing
    def close_expired_option(self):

        self.close_criterion = 'Expiration'

        if self.option_type == 'P':
            if self.stock_data_row['Close'] < self.bought_option_strike:
                self.current_profit = (self.bought_option_strike - self.stock_data_row['Close'] -
                                       self.position_price) * self.option_lot_size
            else:
                self.current_profit = -self.position_price * self.option_lot_size

        elif self.option_type == 'C':
            if self.stock_data_row['Close'] > self.bought_option_strike:
                self.current_profit = (self.stock_data_row['Close'] - self.bought_option_strike -
                                       self.position_price) * self.option_lot_size

            else:
                self.current_profit = -self.position_price * self.option_lot_size

    @timing
    def record_algo_state(self):
        self.trade_log['Position'].append(self.position)
        self.trade_log['Profit'].append(self.current_profit)
        self.trade_log['Date'].append(self.stock_data_i)
        self.trade_log['Strike'].append(self.bought_option_strike)
        self.trade_log['ExpirationDate'].append(self.bought_option_expiration)
        self.trade_log['PositionPrice'].append(self.position_price)
        self.trade_log['Current_option_close'].append(self.current_option_close)
        if self.position:
            self.trade_log['Transaction_id'].append(self.current_transaction_num)
        else:
            self.trade_log['Transaction_id'].append(0)

        self.trade_log['Close_criterion'].append(self.close_criterion)

    @timing
    def _reset_position_state(self):
        self.opened_option_stats = None
        self.position_id = None
        self.current_profit = 0
        self.bought_option_expiration = None
        self.bought_option_strike = None
        self.position = False
        self.position_price = 0
        self.matched_stock_date = None
        self.matched_stock_price = None
        self.close_criterion = None
        self.current_option_close = None

    @timing
    def save_trade_log(self):
        log_data = self.stock_data
        for k, v in self.trade_log.items():
            log_data[k] = v

        log_data.to_csv(f'{self.path}/backtest_{self.ticker}.csv')


if __name__ == '__main__':
    backtest = BacktestEngine(
        ticker='amzn',
        start_date='12.01.2016',
        end_date='29.03.2020',
        moneyness=10,
        sell_at_gain=0.15,
        sell_at_loss=0.05,
        expiration_week=0,
        vix_threshold=0,
        time_to_execute="10:00",
        option_type='call',
        week_days_trading=(0, 1, 2, 3, 4, 5, 6, 7),
        option_lot_size=100,
        path=r'D:/Soft/python/test'
    )

    backtest.run_backtest()
    print(backtest.strat_statistics[0])
    backtest.save_strat_statistics()
    backtest.plot_strat_statistics()

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from freqtrade.strategy import DecimalParameter, IntParameter, merge_informative_pair
from functools import reduce

log = logging.getLogger(__name__)


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

class air(IStrategy):
    minimal_roi = {
        "0": 10
    }

    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 20

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }

    stoploss = -0.25

    use_custom_stoploss = True

    clucha_bbdelta_close = DecimalParameter(0.01,0.05, default=0.01889, decimals=5, space='buy', optimize=True)
    clucha_bbdelta_tail = DecimalParameter(0.7, 1.2, default=0.72235, decimals=5, space='buy', optimize=True)
    clucha_close_bblower = DecimalParameter(0.001, 0.05, default=0.0127, decimals=5, space='buy', optimize=True)
    clucha_closedelta_close = DecimalParameter(0.001, 0.05, default=0.00916, decimals=5, space='buy', optimize=True)
    clucha_rocr_1h = DecimalParameter(0.1, 1.0, default=0.79492, decimals=5, space='buy', optimize=True)

    is_optimize_deadfish = True
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0, space='sell', optimize=is_optimize_deadfish)

    sell_fastx = IntParameter(50, 100, default=75, space='sell', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # buy_1 indicators
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Set Up Bollinger Bands
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()
        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        inf_tf = '1h'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)


        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        # ewo indicators
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)

        #local indicators
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)

        #cofi indicators
        dataframe['adx'] = ta.ADX(dataframe)

        #buy_33 indicators
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)

        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # loss sell indicators
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        dataframe['bb_width'] = (
                (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        is_ewo = (
            (dataframe['rocr_1h'].gt(self.clucha_rocr_1h.value)) &
            ((
                (dataframe['lower'].shift().gt(0)) &
                (dataframe['bbdelta'].gt(dataframe['ha_close'] * self.clucha_bbdelta_close.value)) &
                (dataframe['closedelta'].gt(dataframe['ha_close'] * self.clucha_closedelta_close.value)) &
                (dataframe['tail'].lt(dataframe['bbdelta'] * self.clucha_bbdelta_tail.value)) &
                (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
            ) |
            (
                (dataframe['ha_close'] < dataframe['ema_slow']) &
                (dataframe['ha_close'] < self.clucha_close_bblower.value * dataframe['bb_lowerband'])
            ))
        )

        conditions.append(is_ewo)
        dataframe.loc[is_ewo, 'enter_tag'] += 'buy_ewo'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        
        # sell signal
        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return "sell_fahmi"

        # sell fast
        if current_time - timedelta(minutes=60) > trade.open_date_utc:
            if (current_candle["fastk"] > self.sell_fastx.value) and (current_profit > -0.01):
                return "sell_kalah"

        # sell lama
        if current_time - timedelta(days=1) > trade.open_date_utc:
            if (current_candle["fastk"] > self.sell_fastx.value) and (current_profit > -0.05):
                return "sell_lama"

        # stoploss - deadfish
        if ((current_profit < self.sell_deadfish_profit.value)
                and (current_candle['bb_width'] < self.sell_deadfish_bb_width.value)
                and (current_candle['close'] > current_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value)
                and (current_candle['volume_mean_12'] < current_candle[
                    'volume_mean_24'] * self.sell_deadfish_volume_factor.value)):
            return "sell_stoploss_deadfish"

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')

        return dataframe

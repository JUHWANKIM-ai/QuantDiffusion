import torch
import math

def log_return_to_price(log_returns, initial_prices):
    """
    Convert log returns to price process.

    Args:
    - log_returns: Tensor of shape [N, T, d], log return process.
    - initial_prices: Tensor of shape [N, 1, d], initial prices.

    Returns:
    - prices: Tensor of shape [N, T+1, d], price process.
    """
    # Get the shapes
    N, T, d = log_returns.shape

    # Create a tensor to hold the prices, which will have one more time step than the log returns
    prices = torch.zeros((N, T + 1, d), device=log_returns.device)

    # Set the initial prices
    prices[:, 0, :] = initial_prices.squeeze(1)

    # Iteratively compute the price at each time step
    for t in range(1, T + 1):
        prices[:, t, :] = prices[:, t - 1, :] * torch.exp(log_returns[:, t - 1, :])

    return prices[:, 1:, :]



class TradingStrategy:
    def __init__(self, initial_capital=10000, min_trade_size=0.0001, max_capital_per_asset=0.1667):
        self.initial_capital = initial_capital
        self.min_trade_size = min_trade_size
        self.max_capital_per_asset = max_capital_per_asset * self.initial_capital
        self.positions = None  # Will be initialized later
        self.remaining_capital = None # Will be initialized later
        self.window_size = None # Will depend on the class

    def execute_trades_at_t(self, signals: torch.Tensor, prices: torch.Tensor, t: int):
        """
        Executes trades based on the signals for a batch of assets. (롱 온리)
        +1 = 롱 진입, -1 = 청산(보유분 매도 -> 현금), 0 = 유지.
        신호가 오면 기존 보유분을 먼저 청산한 뒤, +1 이면 다시 롱으로 진입한다.
        포지션은 항상 0 이상 (숏 없음).

        :param signals: Tensor of shape [N, d] where -1 = sell to flat, 0 = hold, 1 = go long.
        :param prices: Tensor of shape [N, d] representing the prices of the assets.
        :param t: execution time

        :return: Updated capital and positions tensors.
        """
        N, d = signals.shape  # Number of batches and assets

        updated_positions = self.positions[:, t, :].clone()  # (N, d)  # modified t-1-> t
        updated_capital = self.remaining_capital[:, t].clone()  # (N,)  # modified t-1-> t

        # 배치(N) 벡터화: 샘플 n 끼리는 독립이라 텐서 연산으로 한 번에 처리.
        # 자본이 자산 간 순차로 공유되므로 자산 루프(i)만 유지 (결과는 기존 이중루프와 동일).
        for i in range(d):
            sig = signals[:, i]                  # (N,)
            price = prices[:, i]                 # (N,)
            traded = sig != 0                    # (N,) bool
            if not bool(torch.any(traded)):
                continue
            zeros = torch.zeros_like(price)

            # 1) 기존 보유분 청산 (신호 있는 샘플만): 보유 롱을 매도해 현금화
            updated_capital = updated_capital + torch.where(traded, updated_positions[:, i] * price, zeros)
            updated_positions[:, i] = torch.where(traded, zeros, updated_positions[:, i])

            # 2) 롱 진입만 (sig == 1). 숏 없음 -> -1 은 청산(flat)으로만 작동.
            units = self.validate_buy(price, updated_capital)   # (N,) 양수
            long_m = (sig == 1)
            updated_positions[:, i] = updated_positions[:, i] + torch.where(long_m, units, zeros)
            updated_capital = updated_capital - torch.where(long_m, units * price, zeros)

        return updated_capital, updated_positions

    def close_position(self, prices):
        """
        Closes all open positions at the final time step.
        :param prices: Tensor of shape [N, d] representing the prices of the assets.
        """
        sod_positions = self.positions[:, -2, :].clone()  # Copy positions to update
        sod_capital = self.remaining_capital[:, -2].clone()  # Copy remaining capital to update

        for i in range(sod_positions.shape[1]):
            self.remaining_capital[:, -1] += sod_positions[:, i] * prices[:, i]
            self.positions[:, -1, i] = 0
        self.remaining_capital[:, -1] += sod_capital

    def validate_buy(self, price, current_capital):
        """
        진입 규모(양수 units) 산정. price/current_capital 이 배치 텐서(N,)여도 동작하도록 벡터화.
        (기존 스칼라 min/max 로직과 동일: min(max_units, capital//price) 를 min_trade_size 로 하한)
        """
        max_units = self.truncate(self.max_capital_per_asset / price)
        units = torch.minimum(max_units, self.truncate(current_capital / price))
        return torch.clamp(units, min=self.min_trade_size)

    def truncate(self, units):
        truncated_units = torch.trunc(units / self.min_trade_size) * self.min_trade_size
        return truncated_units

    def get_pnl_trajectory(self, price_series: torch.Tensor):
        """
        Calculate the PnL trajectory based on volatility trading strategy.
        :param price_series: Tensor of shape [N, T, d], where N = batch size, T = time steps, and d = number of assets.
        """
        N, T, d = price_series.shape

        self.remaining_capital = torch.zeros([N, T + 1]).to(price_series.device)
        self.remaining_capital[:, :self.window_size + 1] = self.initial_capital
        self.positions = torch.zeros([N, T + 1, d]).to(price_series.device)

        buy_sell_signals = self.get_buy_sell_signals(price_series)

        for t in range(self.window_size, T - 1):
            updated_capital, updated_positions = self.execute_trades_at_t(buy_sell_signals[:, t, :],
                                                                          price_series[:, t, :], t)

            # Update the positions and capital
            self.remaining_capital[:, t+1 ] = updated_capital 
            self.positions[:, t+1 ] = updated_positions 

        self.close_position(price_series[:, -1, :])
        pnl = self.compute_cumulative_pnl(price_series)
        pnl = pnl[:, self.window_size:]
        return pnl

    def compute_cumulative_pnl(self, price_series: torch.Tensor):
        """
        Compute the cumulative percentage PnL at each time step t.
        :param price_series: Tensor of shape [N, T, d] representing the price series of assets.
        :return: Tensor of shape [N, T+1, d] representing the cumulative PnL, starting with 1.
        """
        N, T, d = price_series.shape

        # Initialize the PnL tensor, starting with 1 at t=0 for all batches
        pnl = torch.ones([N, T + 1])

        # Loop over all time steps to calculate PnL for each batch
        for t in range(1, T + 1):
            # Portfolio value at time t (remaining capital + sum of positions held * prices across all assets)
            portfolio_value = self.remaining_capital[:, t] + (
                        self.positions[:, t, :] * price_series[:, t-1, :]).sum(dim=1)

            # Cumulative PnL at time t (current portfolio value relative to the initial capital)
            pnl[:, t] = portfolio_value / self.initial_capital

        return pnl

    def get_buy_sell_signals(self, price_series: torch.Tensor):
        raise NotImplementedError("This method should be implemented by derived strategies.")


class EqualWeightPortfolioStrategy(TradingStrategy):
    def __init__(self, initial_capital=10000, min_trade_size=0.0001, max_capital_per_asset=0.1667):
        """
        Initialize the equal weight portfolio strategy.
        :param initial_capital: Total initial capital available.
        :param min_trade_size: Minimum trade size allowed.
        :param max_capital_per_asset: Maximum capital that can be allocated to a single asset (unused in this strategy).
        """
        super().__init__(initial_capital, min_trade_size, max_capital_per_asset)

    def get_buy_sell_signals(self, price_series: torch.Tensor):
        """
        Generate buy signals at the beginning (allocate capital equally) and sell signals at the end.
        :param price_series: Tensor of shape [N, T, d], where N = batch size, T = time steps, and d = number of assets.
        :return: Signals of shape [N, T, d], where 1 indicates buy at the beginning, -1 indicates sell at the end, and 0 otherwise.
        """
        N, T, d = price_series.shape
        signals = torch.zeros_like(price_series).to(price_series.device)

        # Allocate capital equally at the start (buy signal at time t=0)
        signals[:, 0, :] = 1  # Buy equal amounts of all assets at t=0

        # Sell everything at the end (sell signal at time t=T-1)
        signals[:, T-1, :] = -1  # Sell all assets at t=T-1

        return signals

    def get_pnl_trajectory(self, price_series: torch.Tensor):
        """
        Calculate the PnL trajectory based on equal weight buy and hold strategy.
        :param price_series: Tensor of shape [N, T, d], where N = batch size, T = time steps, and d = number of assets.
        """
        N, T, d = price_series.shape

        # Initialize capital and positions for the strategy
        self.remaining_capital = torch.zeros([N, T+1], device=price_series.device)
        self.remaining_capital[:, 0] = self.initial_capital
        self.positions = torch.zeros([N, T+1, d], device=price_series.device)

        # Get buy and sell signals (buy at t=0, sell at t=T-1)
        buy_sell_signals = self.get_buy_sell_signals(price_series)

        # Allocate capital equally across all assets at t=0
        for n in range(N):  # Loop over each batch
            capital_per_asset = self.initial_capital / d  # Equal capital per asset
            self.remaining_capital[n, 1] = self.remaining_capital[n, 0]
            for i in range(d):  # Loop over each asset
                if buy_sell_signals[n, 0, i] == 1:  # Buy signal at t=0
                    units_to_buy = capital_per_asset / price_series[n, 0, i]  # Buy equal number of units
                    self.positions[n, 1, i] = units_to_buy  # Set positions after buy
                    self.remaining_capital[n, 1] -= units_to_buy * price_series[
                        n, 0, i]  # Deduct spent capital

        # The strategy holds the positions until the last time step
        for t in range(1, T-1):
            self.remaining_capital[:, t+1] = self.remaining_capital[:, t]
            self.positions[:, t+1] = self.positions[:, t]

        # Close positions at t=T-1
        for n in range(N):
            for i in range(d):
                if buy_sell_signals[n, T-1, i] == -1:  # Sell signal at t=T-1
                    self.remaining_capital[n, T] += self.positions[n, T-1, i] * price_series[n, T-1, i]
                    self.positions[n, T, i] = 0  # Set positions to zero after selling

        pnl = self.compute_cumulative_pnl(price_series)
        return pnl
    
class MeanReversionStrategy(TradingStrategy):
    # def __init__(self, initial_capital=10000, min_trade_size=0.0001,
    #              max_capital_per_asset=0.1667, window_size=30, threshold=0.03):
    def __init__(self, initial_capital=10000, min_trade_size=0.0001,
                 max_capital_per_asset=0.1667, window_size=10, threshold=0.05):
        """
        :param window_size: 이동평균을 구할 기간(일수)
        :param threshold: 평균 대비 몇 % 이상 벗어났다가 되돌아올 때 시그널을 낼지 (%) 예: 0.02 → 2%
        """
        super().__init__(initial_capital, min_trade_size, max_capital_per_asset)
        self.window_size = window_size
        self.threshold = threshold

    def get_buy_sell_signals(self, price_series: torch.Tensor):
        """
        price_series.shape = [N, T, d]

        1) 이전 시점(prev)에는 평균 대비 ±threshold 구간 밖에 있었다가,
        2) 현재 시점(curr)에는 평균 ±threshold 구간 안으로 되돌아올 때
           (Reversion Confirmation) 매수/매도 신호를 한 번만 발생시킵니다.
        """
        N, T, d = price_series.shape
        signals = torch.zeros((N, T, d), dtype=torch.int8, device=price_series.device)

        # 만약 t=0에 미리 균등 매수하려면 아래 한 줄을 해제하세요.
        signals[:, self.window_size, :] = 1

        # i: 자산 인덱스, t: 시점 인덱스
        for i in range(d):
            # t는 최소 window_size+1부터 시작해야 prev_window와 curr_window가 모두 잡힙니다.
            for t in range(self.window_size + 1, T):
                # (1) 이전 시점(prev)의 윈도우 평균(prev_mean)과 이전 가격(prev_price)
                prev_window = price_series[:, t - 1 - self.window_size : t - 1, i]  # shape [N, window_size]
                prev_mean   = prev_window.mean(dim=1)                               # shape [N]
                prev_price  = price_series[:, t - 1, i]                             # shape [N]

                # (2) 현재 시점(curr)의 윈도우 평균(curr_mean)과 현재 가격(curr_price)
                curr_window = price_series[:, t - self.window_size : t, i]          # shape [N, window_size]
                curr_mean   = curr_window.mean(dim=1)                               # shape [N]
                curr_price  = price_series[:, t, i]                                  # shape [N]

                # (3) Reversion Confirmation: “이전에는 평균±threshold 밖에 있었고, 현재는 평균±threshold 안으로 돌아오는 순간”을 포착

                # ▶ 매수 타이밍 (과매도 구간에서 되돌아올 때)
                #    - prev_price < prev_mean*(1−threshold)   (이전에 평균보다 (1−thr) 아래)
                #    - curr_price ≥ curr_mean*(1−threshold)   (현재 평균−threshold 이상으로 복귀)
                buy_cross = (prev_price <  prev_mean * (1 - self.threshold)) & \
                            (curr_price >= curr_mean * (1 - self.threshold))

                # ▶ 매도 타이밍 (과매수 구간에서 되돌아올 때)
                #    - prev_price > prev_mean*(1+threshold)   (이전에 평균보다 (1+thr) 위)
                #    - curr_price ≤ curr_mean*(1+threshold)   (현재 평균+threshold 이하로 복귀)
                sell_cross = (prev_price >  prev_mean * (1 + self.threshold)) & \
                             (curr_price <= curr_mean * (1 + self.threshold))

                signals[buy_cross,  t, i] =  1
                signals[sell_cross, t, i] = -1
        return signals



class TrendFollowingStrategy(TradingStrategy):
    def __init__(self, initial_capital=10000, min_trade_size=0.0001, max_capital_per_asset=0.1667, short_window=10,
                 long_window=30):
        """
        :param short_window: The short window for calculating the short-term moving average.
        :param long_window: The long window for calculating the long-term moving average.
        """
        super().__init__(initial_capital, min_trade_size, max_capital_per_asset)
        self.short_window = short_window
        self.long_window = long_window
        self.window_size = long_window

    def get_buy_sell_signals(self, price_series: torch.Tensor):
        """
        Generate buy/sell signals based on trend following logic.
        A buy signal is triggered when the short-term MA crosses above the long-term MA for the first time.
        A sell signal is triggered when the short-term MA crosses below the long-term MA for the first time.

        :param price_series: Tensor of shape [N, T, d], where N = batch size, T = time steps, and d = number of assets.
        :return: Signals of shape [N, T, d], where 1 indicates buy, -1 indicates sell, and 0 indicates no action.
        """
        N, T, d = price_series.shape
        signals = torch.zeros_like(price_series).to(price_series.device)

        signals[:, self.window_size, :] = 1

        # Calculate short-term and long-term moving averages for each asset
        for i in range(d):  # Loop over each asset
            short_ma = torch.zeros(N, T).to(price_series.device)
            long_ma = torch.zeros(N, T).to(price_series.device)

            for t in range(self.long_window, T):
                short_ma[:, t] = price_series[:, t - self.short_window:t, i].mean(1)
                long_ma[:, t] = price_series[:, t - self.long_window:t, i].mean(1)

                # Only generate signals at crossover points
                if t > self.long_window:  # Need at least one previous data point to check crossover
                    prev_short_ma = short_ma[:, t - 1]
                    prev_long_ma = long_ma[:, t - 1]

                    # Buy signal: short-term MA crosses above long-term MA
                    buy_mask = (prev_short_ma <= prev_long_ma) & (short_ma[:, t] > long_ma[:, t])
                    signals[buy_mask, t, i] = 1

                    # Sell signal: short-term MA crosses below long-term MA
                    sell_mask = (prev_short_ma >= prev_long_ma) & (short_ma[:, t] < long_ma[:, t])
                    signals[sell_mask, t, i] = -1
        return signals


# High vol -> sell
class VolatilityTradingStrategy(TradingStrategy):
    def __init__(self, initial_capital=10000, min_trade_size=0.0001,
                 max_capital_per_asset=0.1667, window_size=10,
                 upper_threshold=0.30, lower_threshold=0.15):
    # def __init__(self, initial_capital=10000, min_trade_size=0.0001,
    #              max_capital_per_asset=0.1667, window_size=30,
    #              upper_threshold=0.25, lower_threshold=0.10):
        super().__init__(initial_capital, min_trade_size, max_capital_per_asset)
        self.window_size     = window_size
        self.upper_threshold = upper_threshold  # 위험 증가 기준
        self.lower_threshold = lower_threshold  # 시장 안정 기준

    def compute_return_rates(self, price_series: torch.Tensor):
        returns = price_series[:, 1:, :] / price_series[:, :-1, :] - 1.0
        zero    = torch.zeros((price_series.size(0), 1, price_series.size(2)),
                              device=price_series.device)
        return torch.cat((zero, returns), dim=1)  # shape [N, T, d]

    def get_buy_sell_signals(self, price_series: torch.Tensor):
        N, T, d = price_series.shape
        signals = torch.zeros((N, T, d), dtype=torch.int8, device=price_series.device)


        signals[:, self.window_size, :] = 1

        # 1) rolling volatility 계산
        ret = self.compute_return_rates(price_series)  # [N, T, d]
        windows = ret.unfold(1, self.window_size, 1)  
        vol = windows.std(dim=3) * math.sqrt(252)      # 일간 기준 연율화 변동성

        # 2) crossover 방식으로 신호 생성 (Risk-Averse)
        for i in range(d):
            for t in range(self.window_size + 1, T):
                idx_prev = t - self.window_size - 1
                idx_curr = t - self.window_size

                prev_v = vol[:, idx_prev, i]
                curr_v = vol[:, idx_curr, i]

                # 변동성 상단 돌파 → 위험 증가 → 청산(Sell)
                sell_cross = (prev_v <= self.upper_threshold) & (curr_v > self.upper_threshold)

                # 변동성 하단 돌파 → 시장 안정 → 진입(Buy)
                buy_cross  = (prev_v >= self.lower_threshold) & (curr_v < self.lower_threshold)

                signals[buy_cross,  t, i] =  1
                signals[sell_cross, t, i] = -1

        return signals



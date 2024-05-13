import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from dotenv import load_dotenv
load_dotenv()


        
from binance.client import Client
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

class CryptoGraph:
    # TODO: what if I use a tumbler that choose neither real or mean costs 
    def __init__(self, params, start, end, is_meanCosts_const=True, is_weight_const=True, is_meanPrice_const=True, is_step_const=True,is_static=False):
        self.prng = np.random.RandomState(params['seed'])  
        self.graph = nx.DiGraph()
        self.params = params
        self.meanCosts = params['mean_price']
        self.realCosts = params['conversion_rates']
        self.volatilities = params['volatility']


        #these variables are responsible for price modeling
        self.meanCosts_log = params['mean_price_log']
        self.volatilities_log = params['volatility_log']
        

        self.dist = defaultdict(lambda: defaultdict(int))
        self.spreads = defaultdict(lambda: defaultdict(float))
        self.neighbors = defaultdict(list)
        self.vertices = list(params['currencies'])
        self.currency_index_map = {currency: index for index, currency in enumerate(params['currencies'])}

        self.is_meanCosts_const = is_meanCosts_const
        self.is_weight_const = is_weight_const
        self.is_meanPrice_const = is_meanPrice_const
        self.is_step_const = is_step_const

        self.start_node = 0
        self.end_node = 0 
        self.steps = 0  
        self.vertexCount = 0  
        self.Horizon = 1  
        self.mPathsList = []  
        self.nPaths = 0  
        self.set_real_costs()
        self.set_mean_costs()
        self.set_volatility()
        self.is_static = is_static

    
        self.build_graph(params['conversion_rates'], start, end)

        self.test_log_rates = self.realCosts
        self.use_mean = True
        
        

    def get_statistics(self): 
        # print(f"all nodes: {self.currency_index_map}")
        for (from_currency, to_currency), rate in self.realCosts.items(): 
                mean_price_from = self.meanCosts[(from_currency, to_currency)]

                # print(f"{from_currency} -> {to_currency} |LOG| mean = {mean_price_from} | current rate = {rate} ")
                # print(f"{from_currency} -> {to_currency} |   | mean = {np.exp(-mean_price_from)} | current rate = {np.exp(-rate)} ")

    def set_volatility(self):
        items = list(self.volatilities.items())
        for (from_currency, to_currency), rate in items:
            # print('from_currency, to_currency:', from_currency, to_currency, " ------------=------------", rate )
            self.volatilities[(to_currency,from_currency)] = rate

    def set_real_costs(self): 
        items = list(self.realCosts.items())
        for (from_currency, to_currency), rate in items:
            self.realCosts[(from_currency,to_currency)] = -np.log(rate)

    def set_mean_costs(self):
        items = list(self.meanCosts.items())
        for (from_currency, to_currency), rate in items:
            self.meanCosts[(from_currency,to_currency)] = -np.log(rate)
            
        
    def build_graph(self, conversion_rates, start_node, end_node, is_static=False):
        for (from_currency, to_currency), rate in conversion_rates.items():
                mean_price_from = self.meanCosts[(from_currency, to_currency)]
                volatility_from = self.volatilities[(from_currency, to_currency)]

                mean_price_from_log = self.meanCosts_log[(from_currency, to_currency)]
                volatility_from_log = self.volatilities_log[(from_currency, to_currency)]

                if from_currency not in self.graph:
                    self.graph.add_node(from_currency)
                if to_currency not in self.graph:
                    self.graph.add_node(to_currency)

                self.graph.add_edge(from_currency, to_currency, weight=rate,
                                    mean_price=mean_price_from, volatility=volatility_from, 
                                    mean_price_log=mean_price_from_log, volatility_log=volatility_from_log)

                self.spreads[from_currency][to_currency] = volatility_from_log
                self.dist[from_currency][to_currency] = rate
                self.neighbors[from_currency].append(to_currency)

        self.vertexCount = len(self.graph.nodes)
        self.Horizon = self.vertexCount + 1
        if self.is_step_const: 
            self.steps = 1000 #nx.shortest_path_length(self.graph, start_node, end_node)
        else: 
            self.steps = nx.shortest_path_length(self.graph, start_node, end_node)

        r = end_node
        self.spreads[r][r] = 0
        if self.is_static:
            self.graph.add_edge(r, r, weight=0,
                                        mean_price=0, volatility=0, 
                                        mean_price_log=0, volatility_log=0)

        self.neighbors[r].append(r)
        self.meanCosts[(r, r)] = 0
        self.volatilities[(r, r)] = 0
        self.dist[r][r] = 0

        self.start_node = self.vertices.index(start_node)
        self.end_node = self.vertices.index(end_node)
        self.get_statistics()
            
        # input ("Finish buildig the graph! enter to continue")


    def update_graph(self, alpha=0.05, type="gbm"):
        type = "gbm"
        """
        Updates the graph to reflect new market data by adjusting log prices directly.
        Each edge's weight (log cost) and log mean price are updated using an exponential moving average (EMA).
        
        Parameters:
            alpha (float): The smoothing factor used in the EMA, representing the degree of weight decrease,
                        a constant smoothing factor between 0 and 1. A higher alpha discounts older observations faster.
        """
    
        if not (0 < alpha <= 1):
            raise ValueError("Alpha should be between 0 and 1.")

        try:
            for u, v, attrs in self.graph.edges(data=True):
            
                if 'mean_price' not in attrs or 'volatility' not in attrs:
                    print ("ENTER THERE! FIX THE ERROR!")
                    raise KeyError(f"Edge ({u}, {v}) data does not include 'mean_price' and/or 'volatility'.")

                if type != "gbm": 
                    new_price = self.generate_future_prices_ou(np.exp(-1 * attrs['weight']), np.exp(-attrs['mean_price']), attrs['volatility'], num_periods=1)

                else: 
                    new_price = self.generate_future_prices_gbm(np.exp(-1 * attrs['weight']), attrs['mean_price_log'], attrs['volatility_log'], num_periods=1)
                # print(f"OLD = {new_price[0]} NEW PRICE FOR {u} / {v} = {new_price[-1]}")
                # print(f"Does the new one is greater than old one: {new_price[-1] > new_price[0]} ")
                new_price = new_price[1]
                new_log_price = -np.log(new_price)
                if self.use_mean: 
                    # print(f"old mean proce for {u} -> {v}  :", attrs['mean_price'], f"new log price: {new_log_price}")
                    if not self.is_meanPrice_const: 
                        attrs['mean_price'] = alpha * new_log_price + (1 - alpha) * attrs['mean_price']
                    
                    if not self.is_weight_const: 
                        attrs['weight'] = new_log_price
                    if type != "gbm": 
                        if not self.is_meanCosts_const: 
                            self.meanCosts[(u, v)] = alpha * new_log_price + (1 - alpha) * attrs['mean_price']
                            self.realCosts[(u, v)] = new_log_price
                    else:
                        if not self.is_meanCosts_const: 
                            self.meanCosts[(u, v)] = alpha * new_log_price + (1 - alpha) * self.meanCosts[(u, v)]
                            self.realCosts[(u, v)] = new_log_price

                else: 
                    if not self.is_meanPrice_const: 
                        attrs['mean_price'] = (1 - alpha)  * new_log_price + alpha * attrs['mean_price']
                    if not self.is_weight_const:
                        attrs['weight'] = new_log_price
                    if not self.is_meanCosts_const: 
                        self.meanCosts[(u, v)] = attrs['mean_price']
                        self.realCosts[(u, v)] = new_log_price
            
            self.get_statistics()
            # # input ("enter to continue")

        except Exception as e:
            print(f"An error occurred while updating the graph: {e}")
        else:
            print("Graph update complete.")
    
    def generate_future_prices_ou(self, current_price, mean_price, volatility, num_periods):
        """
        Simulates future prices using the Ornstein-Uhlenbeck process, a mean-reverting model.
        
        Parameters:
            current_price (float): The starting price.
            mean_price (float): The long-term mean price towards which the price reverts.
            reversion_speed (float): The speed at which the price reverts to the mean.
            volatility (float): The standard deviation of the returns.
            num_periods (int): The number of periods to simulate.

        Returns:
            list[float]: A list of simulated prices over the specified number of periods.
        """

        prices = [current_price]
        max_price_change = 0.05
        for _ in range(num_periods):
            dt = 1  # Time increment, usually set to 1 for simplicity
            random_shock = np.random.normal(0, volatility * np.sqrt(dt))
            reversion_speed = 0.5 
            price_change = reversion_speed * (mean_price - prices[-1]) * dt + random_shock
            price_change = np.clip(price_change, -prices[-1] * max_price_change, prices[-1] * max_price_change)
      
            new_price = prices[-1] + price_change
            prices.append(new_price)

        return prices
    
    def generate_future_prices_gbm(self, current_price, drift, volatility, num_periods):
        """
        Simulates future prices using the Geometric Brownian Motion model, which includes a drift and volatility component.
        
        Parameters:
            current_price (float): The starting price.
            drift (float): The expected return, representing the drift component of GBM.
            volatility (float): The standard deviation of the returns, representing the volatility.
            num_periods (int): The number of future price periods to simulate.

        Returns:
            list[float]: A list of simulated prices over the specified number of periods.
        """
        prices = [current_price]
        dt = 1  # Assuming each period is one unit of time

        for _ in range(num_periods):
            random_shock = np.random.normal(0, 1)
            price_change_factor = np.exp((drift - 0.5 * volatility**2) * dt + volatility * random_shock * np.sqrt(dt))
            new_price = prices[-1] * price_change_factor
            if new_price == np.inf: 
                prices = self.generate_future_prices(current_price, drift, volatility, num_periods)
                # print(f"prices = {prices}!!!!")
                # input ()
                new_price = prices[-1]
            prices.append(new_price)

        # print(f"********************************PRICES = {prices}")
        return prices

    def generate_future_prices(self, current_price, avg_return, volatility, num_periods, max_price_change=0.05):
        prices = [current_price]
        for _ in range(num_periods):
            # Generate a random return within a bounded range
            random_return = np.random.normal(avg_return, volatility)
            bounded_random_return = np.clip(random_return, -max_price_change, max_price_change)
            
            # Calculate the new price while limiting the maximum change
            new_price = prices[-1] * (1 + bounded_random_return)
            prices.append(new_price)

        return prices
    


    def display_graph(self):
        """
        Displays the graph visually using matplotlib, including edge labels for conversion rates.
        """
        edge_labels = {(u, v): f"Rate: {np.exp(-self.graph[u][v]['weight']):.4f}"
                        for u, v in self.graph.edges()}
        pos = nx.circular_layout(self.graph)  # Используем circular_layout вместо spring_layout
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=16)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='green')
        plt.title('Cryptocurrency Conversion Market')
        plt.show()


    def get_deadline(self):
        cost_range = self.params['costMax'] - self.params['costMin']
        return self.params['costMin'] + cost_range * self.params['deadlinePerc'] * self.steps

    def get_currency_index(self, currency):
        return self.currency_index_map[currency]

    def get_currency_name(self, index):
        for currency, idx in self.currency_index_map.items():
            if idx == index:
                return currency
        return None  # Если индекс не найден, возвращаем None или можно выбрать другое поведение

    def apply_to_market(self, path): 
        total_cost = 0 
        counter = 1 
        pnl = 1
        for i in range(len(path)-1): 
            cost = self.test_log_rates[(path[i], path[i+1])]
            print (f"{path[i]} to { path[i+1]} is {cost}") 
            total_cost += cost 
            print(f'{counter}) Convert {pnl} {path[i]} to {path[i+1]} for {np.exp(-total_cost)} with convertion rate {np.exp(-cost)}')
            counter += 1
            pnl *= np.exp(cost)

        return total_cost


    # def truebellman(self, target_node):
    #     print(f"Enter into truebellman mehtod! Try to execute that")
    #     print(f"All nodes: {self.graph.nodes}")
    #     # input ("get all nodes!")
    #     inflist = [np.inf]*len(self.graph.nodes)
    #     # input ("Press enter to continue!")
    #     # vt - list for values at time t for all the nodes w.r.t. to target_node
    #     vt = {k: v for k, v in zip(self.graph.nodes, inflist)} 
    #     vt[target_node] = 0
        
    #     # decision function for nodes w.r.t. to target_node
    #     dt = {k:v for k,v in zip(self.nodes, self.nodes)}
        
    #      # updating vt
    #     for t in range(1, len(self.nodes)):            
    #         for v in self.nodes:
    #             for w in self.edges[v]:
    #                 # Bellman' equation 
    #                 if (vt[v] > vt[w] + self.distances[(v, w)]):
    #                     vt[v] = vt[w] + self.distances[(v, w)]
    #                     dt[v] = w 
    #     # print(vt)
    #     # print(g.distances)

    #     v_aux = {k:-1 if v == np.inf else v for k,v in vt.items()}
    #     max_node = max(v_aux, key=v_aux.get)
    #     max_dist = v_aux[max_node]

    #     return(max_node,max_dist)  




    def true_bellman_ford(self, target_node):
        # print("Entering true_bellman_ford method! Beginning execution.")
        # print(f"All nodes: {list(self.graph.nodes)}")
        # input ("Press Enter to continue after reviewing all nodes.")

        # Convert node indexes to node names if necessary
        target_node = self.get_currency_name(target_node)
        # print('Target node: ', target_node)

        # Initialize distances from the target node to all other nodes as infinity, except for the target node itself which is zero.
        distances = {node: float('inf') for node in self.graph.nodes}
        distances[target_node] = 0
        # print(f"Initial distances: {distances}")

        # Initialize predecessors for each node
        predecessors = {node: None for node in self.graph.nodes}

        # Relax edges repeatedly
        for _ in range(len(self.graph.nodes) - 1):
            for u in self.graph.nodes:
                for v in self.graph[u]:
                    weight = 1 # Assume each edge has a 'weight' attribute
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        predecessors[v] = u
                        # print(f"Updated distance for {v} based on edge {u} -> {v} with weight {weight}")

        # print(f"All distances: {distances}")
        # print(f"All predecessors: {predecessors}")

        # Check for the furthest node from the target that can be reached
        reachable_nodes = {node: dist for node, dist in distances.items() if dist < float('inf')}
        if reachable_nodes:
            furthest_node = max(reachable_nodes, key=reachable_nodes.get)
            max_distance = reachable_nodes[furthest_node]
        else:
            furthest_node, max_distance = None, None

        # print(f"The furthest reachable node from {target_node} is {furthest_node} with a distance of {max_distance}.")
        return furthest_node, max_distance


    def bellman(self, target_node):
        # Initialize all nodes' distances from the target node to infinity
        distances = {node: float('inf') for node in self.graph.nodes}
        target_node = self.get_currency_name(target_node)
        distances[target_node] = 0  # Distance to itself is zero
        
        # Initialize the decision function for nodes with respect to the target node
        predecessors = {node: None for node in self.graph.nodes}
        
        # Relax edges repeatedly
        for _ in range(len(self.graph.nodes) - 1):
            for v in  self.graph.nodes:
                for w in self.graph[v]:
                    # Use mean costs from the graph for edge weights
                    mean_cost = self.graph[v][w]['mean_price']
                    if distances[w] + mean_cost < distances[v]:
                        # print(f"add from {v} to {w}: {mean_cost} => distances : {distances[v]}")
                        distances[v] = distances[w] + mean_cost
                        predecessors[v] = w

        # print("distances: ", distances)
        return distances


class BinancePriceFetcher:

    def __init__(self, cryptocurrencies,api_key=API_KEY, api_secret=API_SECRET, period=1):
        self.client = Client(api_key, api_secret) 
        self.cryptocurrencies = cryptocurrencies
        self.period = period
        self.filtered_pairs = []
        self.pairs = self.get_pairs_for_cryptos(cryptocurrencies)
        

    def get_pairs_for_cryptos(self, crypto_list):
        """Получение списка торговых пар для заданного списка криптовалют."""
        # Получаем информацию о всех торговых парах на бирже
        exchange_info = self.client.get_exchange_info()
        symbols = exchange_info.get('symbols', [])

        # Фильтрация торговых пар, содержащих криптовалюты из списка
        filtered_pairs = []
        for symbol in symbols:
            base_asset = symbol['baseAsset']
            quote_asset = symbol['quoteAsset']
            if base_asset in crypto_list and quote_asset in crypto_list:
                filtered_pairs.append(f"{base_asset}{quote_asset}")
                self.filtered_pairs.append((f"{base_asset}", f"{quote_asset}"))
                

        return filtered_pairs

    def get_last_hour_prices(self, symbol):
        """ Получает минутные цены за последний час для заданной пары символов. """
        # Установка временных рамок для запроса
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=self.period)
        # Форматируем время в строку, подходящую для API
        start_str = int(start_time.timestamp() * 1000)
        end_str = int(end_time.timestamp() * 1000)
    
        # Получаем свечи за последний час
        klines = self.client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, startTime=start_str, endTime=end_str)
    
        # Преобразуем данные в DataFrame для удобства
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df['close'] = df['close'].astype(float)
        df.drop(columns=['volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], inplace=True)
        
        # Получаем bid и ask цены для каждой свечи
        depth = self.client.get_order_book(symbol=symbol)

        bid_price = float(depth['bids'][0][0]) if depth['bids'] else None
        ask_price = float(depth['asks'][0][0]) if depth['asks'] else None
        
       # Добавляем bid цену в DataFrame
# Добавляем bid цену в DataFrame
        new_row_bid = {'close': bid_price}
        new_index_bid = pd.to_datetime(depth['lastUpdateId'], unit = 'ms')
        df.loc[new_index_bid] = new_row_bid

        # Добавляем ask цену в DataFrame
        new_row_ask = {'close': ask_price}
        new_index_ask = pd.to_datetime(depth['lastUpdateId']+ 1, unit='ms')
        df.loc[new_index_ask] = new_row_ask
        # index=[new_index_ask])
        return df, df[['close']]

    def log_return(self, price: pd.Series) -> pd.Series:
        return price.transform(np.log).diff()
    
    def avg_amplitude(self, log_ret: pd.Series) -> float:
        return log_ret.transform(np.fabs).mean()


    def create_statistics(self):
         # Set seed for reproducibility

        statistics = {
            'currencies': self.cryptocurrencies,
            'conversion_rates': {},
            'mean_price': {},
            'volatility': {}, 
            'conversion_rates_log': {},
            'mean_price_log': {},
            'volatility_log': {}
        }
        
        for base_currency, quote_currency in self.filtered_pairs:
            symbol = base_currency + quote_currency
            _, last_prices = self.get_last_hour_prices(symbol)

            # Calculate volatility (standard deviation of log returns)
            # log_returns = np.log(last_prices / last_prices.shift(1))
            # volatility = log_returns.std()
        
            volatility = last_prices['close'][0:-3].std()
            volatility_1 = (1 / last_prices['close'][0:-3]).std()

            mean_price = last_prices['close'][0:-3].mean()
            mean_price_1 = (1 / last_prices['close'][0:-3]).mean()    
            # print(f"******{symbol}****** BID = {last_prices['close'].iloc[-2]} and ASK = {last_prices['close'].iloc[-1]} ")
            bid_price = last_prices['close'].iloc[-2]
            ask_price = 1 / (last_prices['close'].iloc[-1])
        
            statistics['conversion_rates'][(base_currency, quote_currency)] = bid_price
            statistics['mean_price'][(base_currency, quote_currency)] = mean_price
            statistics['volatility'][(base_currency, quote_currency)] = volatility

            statistics['conversion_rates'][(quote_currency, base_currency)] = ask_price
            statistics['mean_price'][(quote_currency, base_currency)] = mean_price_1
            statistics['volatility'][(quote_currency, base_currency)] = volatility_1

            volatility_log = self.log_return(last_prices['close'][0:-3]).std()
            volatility_1_log = self.log_return((1 / last_prices['close'][0:-3])).std()

            mean_price_log = (self.log_return(last_prices['close'][0:-3])).mean()
            mean_price_1_log =  (self.log_return((1 / last_prices['close'][0:-3]))).mean()

        
            statistics['mean_price_log'][(base_currency, quote_currency)] = mean_price_log
            statistics['volatility_log'][(base_currency, quote_currency)] = volatility_log

            statistics['mean_price_log'][(quote_currency, base_currency)] = mean_price_1_log
            statistics['volatility_log'][(quote_currency, base_currency)] = volatility_1_log

        return statistics



    def plot_candlestick_chart(self, df, name):
        """ Отрисовка свечного графика на основе DataFrame со столбцами 'open', 'high', 'low', 'close'. """
        # Устанавливаем индекс DataFrame в формате datetime, если это ещё не сделано
        if not isinstance(df.index, pd.DatetimeIndex):
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)

        # Убедимся, что все данные имеют числовой тип
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)

        # Настройка стиля и отрисовка графика
        mpf.plot(df, type='candle', style='charles',
                title=f'Minute Candlestick Chart for {name}',
                ylabel='QUOTE')
    

    def prepare_end_node(self,params, start_node, end_node): 
        params['currencies'].append(end_node)
        try:
            for base_currency, quote_currency in self.filtered_pairs:
                # print(" base_currency, quote_currency: ",  base_currency, quote_currency)
                if base_currency == start_node: 
                    params['conversion_rates'][(end_node, quote_currency)] = params['conversion_rates'][(base_currency, quote_currency)]
                    params['mean_price'][(end_node, quote_currency)] = params['mean_price'][(base_currency, quote_currency)]
                    params['volatility'][(end_node, quote_currency)] = params['volatility'][(base_currency, quote_currency)]

                    params['conversion_rates'][(quote_currency, end_node)] = params['conversion_rates'][(quote_currency, base_currency)]
                    params['mean_price'][(quote_currency, end_node)] = params['mean_price'][(quote_currency, base_currency)]
                    params['volatility'][(quote_currency, end_node)] = params['volatility'][(quote_currency, base_currency)]

                    # params['conversion_rates_log'][(end_node, quote_currency)] = params['conversion_rates_log'][(base_currency, quote_currency)]
                    params['mean_price_log'][(end_node, quote_currency)] = params['mean_price_log'][(base_currency, quote_currency)]
                    params['volatility_log'][(end_node, quote_currency)] = params['volatility_log'][(base_currency, quote_currency)]
      
                    # params['conversion_rates_log'][(quote_currency, end_node)] = params['conversion_rates'][(quote_currency, base_currency)]
                    params['mean_price_log'][(quote_currency, end_node)] = params['mean_price_log'][(quote_currency, base_currency)]
                    params['volatility_log'][(quote_currency, end_node)] = params['volatility_log'][(quote_currency, base_currency)]

                elif quote_currency == start_node:
                    params['conversion_rates'][(base_currency, end_node)] = params['conversion_rates'][(base_currency, quote_currency)]
                    params['mean_price'][(base_currency, end_node)] = params['mean_price'][(base_currency, quote_currency)]
                    params['volatility'][(base_currency, end_node)] = params['volatility'][(base_currency, quote_currency)]
                    
                    params['conversion_rates'][(end_node, base_currency)] = params['conversion_rates'][(quote_currency, base_currency)]
                    params['mean_price'][(end_node, base_currency)] = params['mean_price'][(quote_currency, base_currency)]
                    params['volatility'][(end_node, base_currency)] = params['volatility'][(quote_currency, base_currency)]
                    
                    params['mean_price_log'][(base_currency, end_node)] = params['mean_price_log'][(base_currency, quote_currency)]
                    params['volatility_log'][(base_currency, end_node)] = params['volatility_log'][(base_currency, quote_currency)]
                    
                    params['mean_price_log'][(end_node, base_currency)] = params['mean_price_log'][(quote_currency, base_currency)]
                    params['volatility_log'][(end_node, base_currency)] = params['volatility_log'][(quote_currency, base_currency)]
        except Exception as e: 
            print("error: ", e)
        return params




class CryptoGraphDynamic:
    # TODO: what if I use a tumbler that choose neither real or mean costs 
    def __init__(self, params, start, end, is_meanCosts_const=True, is_weight_const=True, is_meanPrice_const=True, is_step_const=True):
        self.prng = np.random.RandomState(params['seed'])  
        self.graph = nx.DiGraph()
        self.params = params
        self.meanCosts = params['mean_price']
        self.realCosts = params['conversion_rates']
        self.volatilities = params['volatility']


        #these variables are responsible for price modeling
        self.meanCosts_log = params['mean_price_log']
        self.volatilities_log = params['volatility_log']
        

        self.dist = defaultdict(lambda: defaultdict(int))
        self.spreads = defaultdict(lambda: defaultdict(float))
        self.neighbors = defaultdict(list)
        self.vertices = list(params['currencies'])
        self.currency_index_map = {currency: index for index, currency in enumerate(params['currencies'])}

        self.is_meanCosts_const = is_meanCosts_const
        self.is_weight_const = is_weight_const
        self.is_meanPrice_const = is_meanPrice_const
        self.is_step_const = is_step_const

        self.start_node = 0
        self.end_node = 0 
        self.steps = 0  
        self.vertexCount = 0  
        self.Horizon = 1  
        self.mPathsList = []  
        self.nPaths = 0  
        self.set_real_costs()
        self.set_mean_costs()
        self.set_volatility()

        
        self.build_graph(params['conversion_rates'], start, end)

        self.test_log_rates = self.realCosts
        self.use_mean = True
        
        

    def get_statistics(self): 
        # print(f"all nodes: {self.currency_index_map}")
        for (from_currency, to_currency), rate in self.realCosts.items(): 
                mean_price_from = self.meanCosts[(from_currency, to_currency)]

                # print(f"{from_currency} -> {to_currency} |LOG| mean = {mean_price_from} | current rate = {rate} ")
                # print(f"{from_currency} -> {to_currency} |   | mean = {np.exp(-mean_price_from)} | current rate = {np.exp(-rate)} ")

    def set_volatility(self):
        items = list(self.volatilities.items())
        for (from_currency, to_currency), rate in items:
            # print('from_currency, to_currency:', from_currency, to_currency, " ------------=------------", rate )
            self.volatilities[(to_currency,from_currency)] = rate

    def set_real_costs(self): 
        items = list(self.realCosts.items())
        for (from_currency, to_currency), rate in items:
            self.realCosts[(from_currency,to_currency)] = -np.log(rate)

    def set_mean_costs(self):
        items = list(self.meanCosts.items())
        for (from_currency, to_currency), rate in items:
            self.meanCosts[(from_currency,to_currency)] = -np.log(rate)
            
        
    def build_graph(self, conversion_rates, start_node, end_node):
        for (from_currency, to_currency), rate in conversion_rates.items():
                mean_price_from = self.meanCosts[(from_currency, to_currency)]
                volatility_from = self.volatilities[(from_currency, to_currency)]

                mean_price_from_log = self.meanCosts_log[(from_currency, to_currency)]
                volatility_from_log = self.volatilities_log[(from_currency, to_currency)]

                if from_currency not in self.graph:
                    self.graph.add_node(from_currency)
                if to_currency not in self.graph:
                    self.graph.add_node(to_currency)

                self.graph.add_edge(from_currency, to_currency, weight=rate,
                                    mean_price=mean_price_from, volatility=volatility_from, 
                                    mean_price_log=mean_price_from_log, volatility_log=volatility_from_log)

                self.spreads[from_currency][to_currency] = volatility_from_log
                self.dist[from_currency][to_currency] = rate
                self.neighbors[from_currency].append(to_currency)

        self.vertexCount = len(self.graph.nodes)
        self.Horizon = self.vertexCount + 1
        if self.is_step_const: 
            self.steps = 1000 #nx.shortest_path_length(self.graph, start_node, end_node)
        else: 
            self.steps = nx.shortest_path_length(self.graph, start_node, end_node)

        r = end_node
        self.spreads[r][r] = 0

        self.neighbors[r].append(r)
        self.meanCosts[(r, r)] = 0
        self.volatilities[(r, r)] = 0
        self.dist[r][r] = 0

        self.start_node = self.vertices.index(start_node)
        self.end_node = self.vertices.index(end_node)
        self.get_statistics()
            
        # input("Finish buildig the graph! enter to continue")


    def update_graph(self, alpha=0.05, type="gbm"):
        type = "gbm"
        """
        Updates the graph to reflect new market data by adjusting log prices directly.
        Each edge's weight (log cost) and log mean price are updated using an exponential moving average (EMA).
        
        Parameters:
            alpha (float): The smoothing factor used in the EMA, representing the degree of weight decrease,
                        a constant smoothing factor between 0 and 1. A higher alpha discounts older observations faster.
        """
    
        if not (0 < alpha <= 1):
            raise ValueError("Alpha should be between 0 and 1.")

        try:
            for u, v, attrs in self.graph.edges(data=True):
            
                if 'mean_price' not in attrs or 'volatility' not in attrs:
                    print ("ENTER THERE! FIX THE ERROR!")
                    raise KeyError(f"Edge ({u}, {v}) data does not include 'mean_price' and/or 'volatility'.")

                if type != "gbm": 
                    new_price = self.generate_future_prices_ou(np.exp(-1 * attrs['weight']), np.exp(-attrs['mean_price']), attrs['volatility'], num_periods=1)

                else: 
                    new_price = self.generate_future_prices_gbm(np.exp(-1 * attrs['weight']), attrs['mean_price_log'], attrs['volatility_log'], num_periods=1)
                # print(f"OLD = {new_price[0]} NEW PRICE FOR {u} / {v} = {new_price[-1]}")
                # print(f"Does the new one is greater than old one: {new_price[-1] > new_price[0]} ")
                new_price = new_price[1]
                new_log_price = -np.log(new_price)
                if self.use_mean: 
                    # print(f"old mean proce for {u} -> {v}  :", attrs['mean_price'], f"new log price: {new_log_price}")
                    if not self.is_meanPrice_const: 
                        attrs['mean_price'] = alpha * new_log_price + (1 - alpha) * attrs['mean_price']
                    
                    if not self.is_weight_const: 
                        attrs['weight'] = new_log_price
                    if type != "gbm": 
                        if not self.is_meanCosts_const: 
                            self.meanCosts[(u, v)] = alpha * new_log_price + (1 - alpha) * attrs['mean_price']
                            self.realCosts[(u, v)] = new_log_price
                    else:
                        if not self.is_meanCosts_const: 
                            self.meanCosts[(u, v)] = alpha * new_log_price + (1 - alpha) * self.meanCosts[(u, v)]
                            self.realCosts[(u, v)] = new_log_price

                else: 
                    if not self.is_meanPrice_const: 
                        attrs['mean_price'] = (1 - alpha)  * new_log_price + alpha * attrs['mean_price']
                    if not self.is_weight_const:
                        attrs['weight'] = new_log_price
                    if not self.is_meanCosts_const: 
                        self.meanCosts[(u, v)] = attrs['mean_price']
                        self.realCosts[(u, v)] = new_log_price
            
            self.get_statistics()
            # input("enter to continue")

        except Exception as e:
            print(f"An error occurred while updating the graph: {e}")
        else:
            # print("Graph update complete.")
            print()
    
    def generate_future_prices_ou(self, current_price, mean_price, volatility, num_periods):
        """
        Simulates future prices using the Ornstein-Uhlenbeck process, a mean-reverting model.
        
        Parameters:
            current_price (float): The starting price.
            mean_price (float): The long-term mean price towards which the price reverts.
            reversion_speed (float): The speed at which the price reverts to the mean.
            volatility (float): The standard deviation of the returns.
            num_periods (int): The number of periods to simulate.

        Returns:
            list[float]: A list of simulated prices over the specified number of periods.
        """

        prices = [current_price]
        max_price_change = 0.05
        for _ in range(num_periods):
            dt = 1  # Time increment, usually set to 1 for simplicity
            random_shock = np.random.normal(0, volatility * np.sqrt(dt))
            reversion_speed = 0.5 
            price_change = reversion_speed * (mean_price - prices[-1]) * dt + random_shock
            price_change = np.clip(price_change, -prices[-1] * max_price_change, prices[-1] * max_price_change)
      
            new_price = prices[-1] + price_change
            prices.append(new_price)

        return prices
    
    def generate_future_prices_gbm(self, current_price, drift, volatility, num_periods):
        """
        Simulates future prices using the Geometric Brownian Motion model, which includes a drift and volatility component.
        
        Parameters:
            current_price (float): The starting price.
            drift (float): The expected return, representing the drift component of GBM.
            volatility (float): The standard deviation of the returns, representing the volatility.
            num_periods (int): The number of future price periods to simulate.

        Returns:
            list[float]: A list of simulated prices over the specified number of periods.
        """
        prices = [current_price]
        dt = 1  # Assuming each period is one unit of time

        for _ in range(num_periods):
            random_shock = np.random.normal(0, 1)
            price_change_factor = np.exp((drift - 0.5 * volatility**2) * dt + volatility * random_shock * np.sqrt(dt))
            new_price = prices[-1] * price_change_factor
            prices.append(new_price)

        return prices

    def generate_future_prices(self, current_price, avg_return, volatility, num_periods, max_price_change=0.05):
        prices = [current_price]
        for _ in range(num_periods):
            # Generate a random return within a bounded range
            random_return = np.random.normal(avg_return, volatility)
            bounded_random_return = np.clip(random_return, -max_price_change, max_price_change)
            
            # Calculate the new price while limiting the maximum change
            new_price = prices[-1] * (1 + bounded_random_return)
            prices.append(new_price)

        return prices
    


    def display_graph(self):
        """
        Displays the graph visually using matplotlib, including edge labels for conversion rates.
        """
        edge_labels = {(u, v): f"Rate: {np.exp(-self.graph[u][v]['weight']):.4f}"
                        for u, v in self.graph.edges()}
        pos = nx.circular_layout(self.graph)  # Используем circular_layout вместо spring_layout
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=16)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='green')
        plt.title('Cryptocurrency Conversion Market')
        plt.show()


    def get_deadline(self):
        cost_range = self.params['costMax'] - self.params['costMin']
        return self.params['costMin'] + cost_range * self.params['deadlinePerc'] * self.steps

    def get_currency_index(self, currency):
        return self.currency_index_map[currency]

    def get_currency_name(self, index):
        for currency, idx in self.currency_index_map.items():
            if idx == index:
                return currency
        return None  # Если индекс не найден, возвращаем None или можно выбрать другое поведение

    def apply_to_market(self, path): 
        total_cost = 0 
        counter = 1 
        pnl = 1
        for i in range(len(path)-1): 
            cost = self.test_log_rates[(path[i], path[i+1])]
            print (f"{path[i]} to { path[i+1]} is {cost}") 
            total_cost += cost 
            print(f'{counter}) Convert {pnl} {path[i]} to {path[i+1]} for {np.exp(-total_cost)} with convertion rate {np.exp(-cost)}')
            counter += 1
            pnl *= np.exp(cost)

        return total_cost


class BinancePriceFetcherDynamic:

    def __init__(self, cryptocurrencies,api_key=API_KEY, api_secret=API_SECRET, period=1):
        self.client = Client(api_key, api_secret) 
        self.cryptocurrencies = cryptocurrencies
        self.period = period
        self.filtered_pairs = []
        self.pairs = self.get_pairs_for_cryptos(cryptocurrencies)
        

    def get_pairs_for_cryptos(self, crypto_list):
        """Получение списка торговых пар для заданного списка криптовалют."""
        # Получаем информацию о всех торговых парах на бирже
        exchange_info = self.client.get_exchange_info()
        symbols = exchange_info.get('symbols', [])

        # Фильтрация торговых пар, содержащих криптовалюты из списка
        filtered_pairs = []
        for symbol in symbols:
            base_asset = symbol['baseAsset']
            quote_asset = symbol['quoteAsset']
            if base_asset in crypto_list and quote_asset in crypto_list:
                filtered_pairs.append(f"{base_asset}{quote_asset}")
                self.filtered_pairs.append((f"{base_asset}", f"{quote_asset}"))
                

        return filtered_pairs

    def get_last_hour_prices(self, symbol):
        """ Получает минутные цены за последний час для заданной пары символов. """
        # Установка временных рамок для запроса
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=self.period)
        # Форматируем время в строку, подходящую для API
        start_str = int(start_time.timestamp() * 1000)
        end_str = int(end_time.timestamp() * 1000)
    
        # Получаем свечи за последний час
        klines = self.client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, startTime=start_str, endTime=end_str)
    
        # Преобразуем данные в DataFrame для удобства
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df['close'] = df['close'].astype(float)
        df.drop(columns=['volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], inplace=True)
        
        # Получаем bid и ask цены для каждой свечи
        depth = self.client.get_order_book(symbol=symbol)

        bid_price = float(depth['bids'][0][0]) if depth['bids'] else None
        ask_price = float(depth['asks'][0][0]) if depth['asks'] else None
        
       # Добавляем bid цену в DataFrame
# Добавляем bid цену в DataFrame
        new_row_bid = {'close': bid_price}
        new_index_bid = pd.to_datetime(depth['lastUpdateId'], unit = 'ms')
        df.loc[new_index_bid] = new_row_bid

        # Добавляем ask цену в DataFrame
        new_row_ask = {'close': ask_price}
        new_index_ask = pd.to_datetime(depth['lastUpdateId']+ 1, unit='ms')
        df.loc[new_index_ask] = new_row_ask
        # index=[new_index_ask])
        return df, df[['close']]

    def log_return(self, price: pd.Series) -> pd.Series:
        return price.transform(np.log).diff()
    
    def avg_amplitude(self, log_ret: pd.Series) -> float:
        return log_ret.transform(np.fabs).mean()


    def create_statistics(self):
         # Set seed for reproducibility

        statistics = {
            'currencies': self.cryptocurrencies,
            'conversion_rates': {},
            'mean_price': {},
            'volatility': {}, 
            'conversion_rates_log': {},
            'mean_price_log': {},
            'volatility_log': {}
        }
        
        for base_currency, quote_currency in self.filtered_pairs:
            symbol = base_currency + quote_currency
            _, last_prices = self.get_last_hour_prices(symbol)

            # Calculate volatility (standard deviation of log returns)
            # log_returns = np.log(last_prices / last_prices.shift(1))
            # volatility = log_returns.std()
        
            volatility = last_prices['close'][0:-3].std()
            volatility_1 = (1 / last_prices['close'][0:-3]).std()

            mean_price = last_prices['close'][0:-3].mean()
            mean_price_1 = (1 / last_prices['close'][0:-3]).mean()    
            bid_price = last_prices['close'].iloc[-2]
            ask_price = 1 / (last_prices['close'].iloc[-1])
        
            statistics['conversion_rates'][(base_currency, quote_currency)] = bid_price
            statistics['mean_price'][(base_currency, quote_currency)] = mean_price
            statistics['volatility'][(base_currency, quote_currency)] = volatility

            statistics['conversion_rates'][(quote_currency, base_currency)] = ask_price
            statistics['mean_price'][(quote_currency, base_currency)] = mean_price_1
            statistics['volatility'][(quote_currency, base_currency)] = volatility_1

            volatility_log = self.log_return(last_prices['close'][0:-3]).std()
            volatility_1_log = self.log_return((1 / last_prices['close'][0:-3])).std()

            mean_price_log = (self.log_return(last_prices['close'][0:-3])).mean()
            mean_price_1_log =  (self.log_return((1 / last_prices['close'][0:-3]))).mean()

        
            statistics['mean_price_log'][(base_currency, quote_currency)] = mean_price_log
            statistics['volatility_log'][(base_currency, quote_currency)] = volatility_log

            statistics['mean_price_log'][(quote_currency, base_currency)] = mean_price_1_log
            statistics['volatility_log'][(quote_currency, base_currency)] = volatility_1_log

        return statistics



    def plot_candlestick_chart(self, df, name):
        """ Отрисовка свечного графика на основе DataFrame со столбцами 'open', 'high', 'low', 'close'. """
        # Устанавливаем индекс DataFrame в формате datetime, если это ещё не сделано
        if not isinstance(df.index, pd.DatetimeIndex):
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)

        # Убедимся, что все данные имеют числовой тип
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)

        # Настройка стиля и отрисовка графика
        mpf.plot(df, type='candle', style='charles',
                title=f'Minute Candlestick Chart for {name}',
                ylabel='QUOTE')
    

    def prepare_end_node(self,params, start_node, end_node): 
        params['currencies'].append(end_node)
        try:
            for base_currency, quote_currency in self.filtered_pairs:
                # print(" base_currency, quote_currency: ",  base_currency, quote_currency)
                if base_currency == start_node: 
                    params['conversion_rates'][(end_node, quote_currency)] = params['conversion_rates'][(base_currency, quote_currency)]
                    params['mean_price'][(end_node, quote_currency)] = params['mean_price'][(base_currency, quote_currency)]
                    params['volatility'][(end_node, quote_currency)] = params['volatility'][(base_currency, quote_currency)]

                    params['conversion_rates'][(quote_currency, end_node)] = params['conversion_rates'][(quote_currency, base_currency)]
                    params['mean_price'][(quote_currency, end_node)] = params['mean_price'][(quote_currency, base_currency)]
                    params['volatility'][(quote_currency, end_node)] = params['volatility'][(quote_currency, base_currency)]

                    # params['conversion_rates_log'][(end_node, quote_currency)] = params['conversion_rates_log'][(base_currency, quote_currency)]
                    params['mean_price_log'][(end_node, quote_currency)] = params['mean_price_log'][(base_currency, quote_currency)]
                    params['volatility_log'][(end_node, quote_currency)] = params['volatility_log'][(base_currency, quote_currency)]
      
                    # params['conversion_rates_log'][(quote_currency, end_node)] = params['conversion_rates'][(quote_currency, base_currency)]
                    params['mean_price_log'][(quote_currency, end_node)] = params['mean_price_log'][(quote_currency, base_currency)]
                    params['volatility_log'][(quote_currency, end_node)] = params['volatility_log'][(quote_currency, base_currency)]

                elif quote_currency == start_node:
                    params['conversion_rates'][(base_currency, end_node)] = params['conversion_rates'][(base_currency, quote_currency)]
                    params['mean_price'][(base_currency, end_node)] = params['mean_price'][(base_currency, quote_currency)]
                    params['volatility'][(base_currency, end_node)] = params['volatility'][(base_currency, quote_currency)]
                    
                    params['conversion_rates'][(end_node, base_currency)] = params['conversion_rates'][(quote_currency, base_currency)]
                    params['mean_price'][(end_node, base_currency)] = params['mean_price'][(quote_currency, base_currency)]
                    params['volatility'][(end_node, base_currency)] = params['volatility'][(quote_currency, base_currency)]
                    
                    params['mean_price_log'][(base_currency, end_node)] = params['mean_price_log'][(base_currency, quote_currency)]
                    params['volatility_log'][(base_currency, end_node)] = params['volatility_log'][(base_currency, quote_currency)]
                    
                    params['mean_price_log'][(end_node, base_currency)] = params['mean_price_log'][(quote_currency, base_currency)]
                    params['volatility_log'][(end_node, base_currency)] = params['volatility_log'][(quote_currency, base_currency)]
        except Exception as e: 
            print("error: ", e)
        return params




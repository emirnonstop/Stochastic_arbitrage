"""
Stochastic Shortest Paths - Learning the costs
Dynamic Model - search for the parameter theta, 
which represents the percentile of the distribution 
of each cost to ensure the smallest possible penalty.
Run using a python command.

Author: Andrei Graur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

from CryptoGraph import CryptoGraph, BinancePriceFetcher
from Model_btc import StaticModel_Dynamic
from Policy import LookaheadPolicy


API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')


if __name__ == "__main__":
    # Load parameters from an Excel file
    file = 'Parameters_d.xlsx'
    seed = 189654913
    METRIC = "PERCENTILE"

    crypto_list = ["BTC", "ARB", "OP", "STRK", "USDT", "ETH"]
    fetcher = BinancePriceFetcher(crypto_list, API_KEY, API_SECRET)
    params = fetcher.create_statistics()
    
    seed = 189654913
    METRIC = "PERCENTILE"

    # Read algorithm parameters from the Excel sheet
    # parameter_data = pd.read_excel(file, sheet_name='Parameters')
    # parameters = {item[0]: item[1] for item in parameter_data.set_index('Index').to_dict('split')['data']}
    # parameters['seed'] = seed

    time.sleep(3)
    fetcher_tester = BinancePriceFetcher(crypto_list, API_KEY, API_SECRET)
    params_tester = fetcher_tester.create_statistics()

    # params_tester['seed'] = seed
    # params_tester['costMin'] = -10
    # params_tester['costMax'] = 10 
    # params_tester ['deadlinePerc'] = 0.6
    # params_tester ['maxVolatility'] = 100
    # params_tester ['maxSpreadPerc'] = 100

    params['seed'] = seed
    params['costMin'] = -10
    params['costMax'] = 10 
    params ['deadlinePerc'] = 0.6
    params ['maxVolatility'] = 100
    params ['maxSpreadPerc'] = 100


    start_node = "BTC"
    end_node = "f" + start_node
    params = fetcher.prepare_end_node(params, start_node, end_node)

    # params_tester = fetcher_tester.prepare_end_node(params_tester, start_node, end_node)

    # print("GLOBAL PARAMS:")
    # for p in params:
    #     print(p)
    #     print(params[p])

    crypto_graph = CryptoGraph(params, start=start_node, end=end_node,is_meanCosts_const=False, is_weight_const=False, is_meanPrice_const=False, is_step_const=True)
    G = crypto_graph

    crypto_graph.update_graph()  # to simulate dynamic price changes
    # crypto_graph.display_graph()
    print("Graph initialized and prices updated. Start node: {}, End node: {}.".format(crypto_graph.vertices[0], crypto_graph.vertices[-1]))


    state_names = ['node']
    init_state = {'node': crypto_graph.vertices[0]}
    decision_names = ['nextNode']

    parDf = pd.read_excel(file, sheet_name = 'Parameters')
    parDict=parDf.set_index('Index').T.to_dict('list')
    parameters = {key:v for key, value in parDict.items() for v in value}
    parameters['seed'] = seed
    theta_list = parameters['theta_cost_set'].split()

    M = StaticModel_Dynamic(state_names, decision_names, init_state, parameters, crypto_graph)
    # model = StaticModel_Dynamic(state_names, decision_names, init_state, parameters, crypto_graph)
    
    # Prepare lists to hold the results
    x = []
    avgCostList = []
    avgPenaltyList = []
    avgStepsList = []
    
    paths = []

    # Iterating over theta
    for theta in theta_list:

        theta = float(theta)
        M.start_new_theta(theta)
        x.append(theta)
            
        cost, penalty, steps, way = M.runTrials(parameters['nIterations'],crypto_graph.get_deadline())
        
        paths.append(way)
        avgCostList.append(cost)
        avgPenaltyList.append(penalty)
        avgStepsList.append(steps)

        print("Avg total cost with parameter {0} is {1:.3f}. Probability of being late is {2:.2f} and avg number of steps is {3:.2f}\n ".format(theta, cost, penalty,steps))
        input()
        # exit()
        

    print("ThetaCost ",x)
    print("AvgCost ",avgCostList)
    print("ProbLateness ",avgPenaltyList)
    print("AvgSteps ",avgStepsList)

    #Ploting the results
    fig1, axsubs = plt.subplots(1,2)
    fig1.suptitle('Comparison of theta^cost -  origin {}, destination {}, dist {} - deadline {} and number of iterations {}'.format(M.G.start_node,M.G.end_node,M.G.steps,G.get_deadline(),parameters['nIterations']) )
  

    axsubs[0].plot(x, avgCostList)
    axsubs[0].set_title('Average Cost')
    axsubs[0].set_xlabel('Percentile')
    axsubs[0].set_ylabel('$')

    
    plt.show()

    params_tester['seed'] = seed
    params_tester['costMin'] = -10
    params_tester['costMax'] = 10 
    params_tester ['deadlinePerc'] = 0.6
    params_tester ['maxVolatility'] = 100
    params_tester ['maxSpreadPerc'] = 100

    params_tester = fetcher_tester.prepare_end_node(params_tester, start_node, end_node)


    crypto_graph_tester = CryptoGraph(params_tester, start=start_node, end=end_node,is_meanCosts_const=False, is_weight_const=False, is_meanPrice_const=False, is_step_const=True)
    for path in paths: 
        print ("----ARBITRAGE PATH----")
        print(path)
        print ("--------------------------------------")
    
        arb_cost = crypto_graph.apply_to_market(path)
        delta = np.exp(-arb_cost) - 1 
        print((f" 'T'   delta of portfolio is {delta * 100} %").upper())
    
        arb_cost = crypto_graph_tester.apply_to_market(path)
        delta = np.exp(-arb_cost) - 1 
        print((f" 'T+1' delta of portfolio is {delta * 100} %").upper())
        print('\n')



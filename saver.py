import numpy as np
import pandas as pd
from copy import copy
from collections import (namedtuple, defaultdict)
import matplotlib.pyplot as plt
import os
import datetime
import builtins
import time

from StaticModelAdaptive import StaticModel
from PolicyAdaptive import Policy
from CryptoGraph import BinancePriceFetcher, CryptoGraph
from algo import find_arbitrage, all_negative_cycles

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')



def determenistic_main(gragh, tester_graph=1, start_node="BTC"):   
    print(f"************************ DETERMENTISTIC SEARCH ************************")
    print(f"================================BINANCE================================")
    print(f"Date - {datetime.datetime.now()}")
    path = find_arbitrage(g=gragh,
                        find_all=True, 
                        )

    # res = all_negative_cycles(gragh)


def run_static_main(params,params_tester, start_node, end_node, parameters):
    print("Starting adaptive stochastic shortest path with parameters")
    state_names = ['CurrentNode', 'CurrentNodeLinksCost']
    decision_names = ['NextNode']

    crypto_graph = CryptoGraph(params, start=start_node, end=end_node,is_meanCosts_const=False, is_weight_const=False, is_meanPrice_const=False, is_step_const=True)
    G = crypto_graph

    M = StaticModel(state_names, decision_names,  params, crypto_graph)
    policy_names = ['PureExploitation']
    P = Policy(M, policy_names)
    theta_list = M.init_args['theta_set'].split()
    # print("policy was chosen! I need to go next!")    

    print (f"***********Theta_list >>> {theta_list}") 
    obj_along_iterations = {theta:[] for theta in theta_list}
    vbar_along_iterations = {theta:[] for theta in theta_list}
    paths = {theta:[] for theta in theta_list}

    for theta in theta_list:
        # model = copy(M)
        M = StaticModel(state_names, decision_names,  params, crypto_graph)
        theta_list = M.init_args['theta_set'].split()
        policy_names = ['PureExploitation']
        P = Policy(M, policy_names)
        model = copy(M)
        model.theta_step = float(theta)
        model.prng = np.random.RandomState(model.init_args['seed'])
        
        print("\n\n***********Starting iterations for theta {}".format(model.theta_step))
    
        for ite in list(range(model.init_args['nIterations'])):
            model.visited = set()
            P.model.visited = set()
            # model.visited.add(start_node)
            # P.model.visited.add(start_node)
            model.obj = 0 
            model.state = model.build_state(model.init_state)
            # print(f"model.state.CurrentNode: {model.state.CurrentNode}")
            idx  = G.get_currency_name(model.state.CurrentNode)

            print("\tTheta {}  - Iteration {} - Stepsize {:.2f} - InitState {} - Vbar {:.2f}".format(model.theta_step,model.n,model.alpha(),model.state.CurrentNode,model.V_t[idx]))
            step = 1
            path = [model.G.get_currency_name(model.G.start_node)]
            while model.state.CurrentNode != model.init_args['target_node']:
                # print(f"Current Node: {model.state.CurrentNode} and target node : {model.init_args['target_node']} ")
                # input ()
                # calling policy and choosing a decision
                decision,vhat = P.make_decision(model)
                path.append(decision)
                print(f"*********** from {model.g.get_currency_name(model.state.CurrentNode)} ---->  {decision}")
                x = model.build_decision({'NextNode': decision})
                jdx = G.get_currency_name(model.state.CurrentNode)
                
                print("\t\t Theta {}  - Iteration {} - Step {} - Current State {} - vbar = {:.2f}".format(model.theta_step,model.n,step, model.state.CurrentNode,model.V_t[jdx]))
                
                vbar = model.update_VFA(vhat)

                print("\t\tDecision={}, vhat {:.2f} and new vbar for current state {:.2f}".format(x[0],vhat,model.V_t[G.get_currency_name(model.state.CurrentNode)]))

                # transition to the next state w/ the given decision
                model.transition_fn(x)
                step += 1

                
            print("Finishing Theta {} and Iteration {} with {} steps. Total cost: {} \n".format(model.theta_step,model.n,step-1,model.obj))
            model.n+=1
            obj_along_iterations[theta].append(model.obj)
            # print(f"obj_along_iterations[theta].append(model.obj): {obj_along_iterations}")
            # print(f"model.origin_node: {model.origin_node}")
            crypto = model.g.get_currency_name(model.origin_node)
            # print(f"model.V_t[crypto]): {model.V_t[crypto]}")
            vbar_along_iterations[theta].append(model.V_t[crypto])
            # print(f"vbar_along_iterations: {vbar_along_iterations}")
            print(f"**********************The arbitrage path: {path}")
            paths[theta] = path
            print(f"********************** {model.obj} ")
            print(f"********************** {model.V_t[crypto]}")
            print(f"\n\n\n\n")
            # input()


    params_tester = fetcher_tester.prepare_end_node(params_tester, start_node, end_node)
    crypto_graph_tester = CryptoGraph(params_tester, start=start_node, end=end_node,is_meanCosts_const=False, is_weight_const=False, is_meanPrice_const=False, is_step_const=True)

    for theta in theta_list: 
        if (model.init_args['nIterations'] == 1): 
            print("********************** THETA -  ",theta)
            print("********************** OBJ -  ", obj_along_iterations[theta])
            print("********************** V_BAR -  ", vbar_along_iterations[theta])
            print("********************** PATH  ",paths[theta])
        
            arb_cost = crypto_graph.apply_to_market(path)
            delta = np.exp(-arb_cost) - 1 
            print((f"********************** 'T'  delta of portfolio is {delta * 100} %").upper())
        
            arb_cost = crypto_graph_tester.apply_to_market(path)
            delta = np.exp(-arb_cost) - 1 
            print((f"********************** T+1' delta of portfolio is {delta * 100} %").upper())
            print('\n')
    # #Ploting the results
    # fig1, axsubs = plt.subplots(1,2)
    # fig1.suptitle('Comparison of theta^step - stepsize type {} - origin {}, target {}, dist {} '.format(M.init_args['stepsize_rule'],M.origin_node,M.target_node,M.dist) )

    # color_list = ['b','g','r','m','c']
    # nThetas = list(range(len(theta_list)))
    # Iterations = list(range(model.init_args['nIterations']))

    # # print("obj_along_iterations", obj_along_iterations)
    # totals_obj = [np.array(obj_along_iterations[theta]).sum() for theta in theta_list]
    # totals_V = [np.array(vbar_along_iterations[theta]).sum() for theta in theta_list]

    # for theta,t in zip(theta_list,nThetas):
        

    #     axsubs[0].plot(Iterations, np.array(obj_along_iterations[theta]).cumsum(), "{}-".format(color_list[t]),label = "{}".format(theta))
    #     axsubs[0].plot(Iterations, obj_along_iterations[theta], "{}-".format(color_list[t]),label = "{}".format(theta))

    #     axsubs[0].set_title('Objective function')
    #     axsubs[0].legend()
    #     axsubs[0].set_xlabel('Iterations')
    #     axsubs[0].set_ylabel('Cost')

    #     axsubs[1].plot(Iterations, vbar_along_iterations[theta], "{}o-".format(color_list[t]),label = "{}".format(theta))
    #     axsubs[1].set_title('Vbar')
    #     axsubs[1].legend()
    #     axsubs[1].set_xlabel('Iterations')
    #     axsubs[1].set_ylabel('Cost')


    #     # axsubs.plot(Iterations, obj_along_iterations[theta], "{}o-".format(color_list[t]),label = "{}".format(theta))
    #     # axsubs[0].set_title('Cost')
    #     # axsubs.legend()
    #     # axsubs.set_xlabel('Iterations')
        # axsubs.set_ylabel('Cost')

# plt.show()


    

def run_determenistic_main(params, start_node, end_node, parameters):
    state_names = ['CurrentNode', 'CurrentNodeLinksCost']
    decision_names = ['NextNode']

    crypto_graph = CryptoGraph(params, start=start_node, end=end_node ,is_meanCosts_const=False, is_weight_const=False, is_meanPrice_const=False, is_step_const=True)
    determenistic_main(crypto_graph.graph)


def run_dynamic_main(): 
    state_names = ['CurrentNode', 'CurrentNodeLinksCost']
    decision_names = ['NextNode']



if __name__ == "__main__":

    seed = 89720123


    file = 'Parameters.xlsx'
    parDf = pd.read_excel(file, sheet_name = 'parameters')
    parDict=parDf.set_index('Index').T.to_dict('list')
    parameters = {key:v for key, value in parDict.items() for v in value}
    parameters.update({'seed':seed})

    # print("------------PARAMETERS------------")
    # for param in parameters: 
    #     print(param)
   

    state_names = ['CurrentNode', 'CurrentNodeLinksCost']
    decision_names = ['NextNode']
    


    """------------Create a crypto graph-------------"""

    for i in range(10):
        print("****************************** ARBITRAGE SEARCH ******************************")
        cryptocurrencies = ['USDT','ETH', 'AVAX', 'LTC', 'BTC', 'SOL', 'ADA', 'MATIC', 'ATOM', 'DOGE', 'SEI',
        'SUI', 'ARB', 'TWT', 'DOT', 'FLOKI', "TON", "BONK"]

        crypto_list = ["BTC", "ARB", "OP", "STRK", "USDT", "ETH"]
        # crypto_list = cryptocurrencies
        fetcher = BinancePriceFetcher(crypto_list, API_KEY, API_SECRET)
        params = fetcher.create_statistics()

        # Read algorithm parameters from the Excel sheet
        # parameter_data = pd.read_excel(file, sheet_name='Parameters')
        # parameters = {item[0]: item[1] for item in parameter_data.set_index('Index').to_dict('split')['data']}
        # parameters['seed'] = seed

        time.sleep(10)
        fetcher_tester = BinancePriceFetcher(crypto_list, API_KEY, API_SECRET)
        params_tester = fetcher_tester.create_statistics()

        params['seed'] = seed
        params['costMin'] = -10
        params['costMax'] = 10 
        params ['deadlinePerc'] = 0.6
        params ['maxVolatility'] = 100
        params ['maxSpreadPerc'] = 100
        params ['theta_set'] = parameters['theta_set'] 
        params ['stepsize_rule'] = parameters['stepsize_rule'] 
        params ['nIterations'] =  parameters['nIterations'] 
        
        params_tester = params


    

        # params_tester = fetcher_tester.prepare_end_node(params_tester, start_node, end_node)

        # print("------------PARAMS------------")
        # for p in params:
        #     print(p)

        start_node = "BTC"
        run_determenistic_main(params=params, 
                        start_node=start_node,
                        end_node=start_node,
                        parameters=parameters)
        
        end_node = "f" + start_node
        params = fetcher.prepare_end_node(params, start_node, end_node)
        run_static_main(params=params, 
                        params_tester=params_tester,
                        start_node=start_node,
                        end_node=start_node,
                        parameters=parameters)
        
        exit()


---------------------------------


import numpy as np
import pandas as pd
from copy import copy
from collections import (namedtuple, defaultdict)
import matplotlib.pyplot as plt
import os
import datetime
import builtins
import time

from StaticModelAdaptive import StaticModel
from PolicyAdaptive import Policy
from CryptoGraph import BinancePriceFetcher, CryptoGraph
from algo import find_arbitrage, all_negative_cycles

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')



def determenistic_main(gragh, tester_graph=1, start_node="BTC"):   
    print(f"************************ DETERMENTISTIC SEARCH ************************")
    print(f"================================BINANCE================================")
    print(f"Date - {datetime.datetime.now()}")
    path = find_arbitrage(g=gragh,
                        find_all=True, 
                        )

    # res = all_negative_cycles(gragh)





if __name__ == "__main__":

    seed = 89720123


    file = 'Parameters.xlsx'
    parDf = pd.read_excel(file, sheet_name = 'parameters')
    parDict=parDf.set_index('Index').T.to_dict('list')
    print("Starting adaptive stochastic shortest path with parameters")
    parameters = {key:v for key, value in parDict.items() for v in value}
    parameters.update({'seed':seed})

    # print("------------PARAMETERS------------")
    # for param in parameters: 
    #     print(param)
   

    state_names = ['CurrentNode', 'CurrentNodeLinksCost']
    decision_names = ['NextNode']
    


    """------------Create a crypto graph-------------"""

    for i in range(10):
        print("****************************** ARBITRAGE SEARCH ******************************")
        cryptocurrencies = ['USDT','ETH', 'AVAX', 'LTC', 'BTC', 'SOL', 'ADA', 'MATIC', 'ATOM', 'DOGE', 'SEI',
        'SUI', 'ARB', 'TWT', 'DOT', 'FLOKI', "TON", "BONK"]

        crypto_list = ["BTC", "ARB", "OP", "STRK", "USDT", "ETH"]
        crypto_list = cryptocurrencies
        fetcher = BinancePriceFetcher(crypto_list, API_KEY, API_SECRET)
        params = fetcher.create_statistics()

        # Read algorithm parameters from the Excel sheet
        # parameter_data = pd.read_excel(file, sheet_name='Parameters')
        # parameters = {item[0]: item[1] for item in parameter_data.set_index('Index').to_dict('split')['data']}
        # parameters['seed'] = seed

        time.sleep(10)
        fetcher_tester = BinancePriceFetcher(crypto_list, API_KEY, API_SECRET)
        params_tester = fetcher_tester.create_statistics()

        params['seed'] = seed
        params['costMin'] = -10
        params['costMax'] = 10 
        params ['deadlinePerc'] = 0.6
        params ['maxVolatility'] = 100
        params ['maxSpreadPerc'] = 100
        params ['theta_set'] = parameters['theta_set'] 
        params ['stepsize_rule'] = parameters['stepsize_rule'] 
        params ['nIterations'] =  parameters['nIterations'] 
        
        params_d = params 
        params_tester = params

        start_node = "BTC"
        end_node = "f" + start_node
        # params = fetcher.prepare_end_node(params, start_node, end_node)

        params_tester = fetcher_tester.prepare_end_node(params_tester, start_node, end_node)

        # print("------------PARAMS------------")
        # for p in params:
        #     print(p)
            

        crypto_graph_d = CryptoGraph(params_d, start=start_node, end=start_node,is_meanCosts_const=False, is_weight_const=False, is_meanPrice_const=False, is_step_const=True)
        
        end_node = "f" + start_node
        params = fetcher.prepare_end_node(params, start_node, end_node)
        crypto_graph = CryptoGraph(params, start=start_node, end=end_node,is_meanCosts_const=False, is_weight_const=False, is_meanPrice_const=False, is_step_const=True)
        
        G = crypto_graph
        # determenistic_main(crypto_graph_d.graph)


        """----------------------------------------------"""

        # create the model, given the above policy
        M = StaticModel(state_names, decision_names,  params, crypto_graph)
        policy_names = ['PureExploitation']
        P = Policy(M, policy_names)
        theta_list = M.init_args['theta_set'].split()
        # print("policy was chosen! I need to go next!")    

        print (f"***********Theta_list >>> {theta_list}") 
        obj_along_iterations = {theta:[] for theta in theta_list}
        vbar_along_iterations = {theta:[] for theta in theta_list}
        paths = {theta:[] for theta in theta_list}


        for theta in theta_list:
            
            # model = copy(M)
            M = StaticModel(state_names, decision_names,  params, crypto_graph)
            theta_list = M.init_args['theta_set'].split()
            policy_names = ['PureExploitation']
            P = Policy(M, policy_names)
            model = copy(M)
            model.theta_step = float(theta)
            model.prng = np.random.RandomState(model.init_args['seed'])
            
            print("\n\n\n\n\n\n***********Starting iterations for theta {}".format(model.theta_step))
        
            for ite in list(range(model.init_args['nIterations'])):
                model.visited = set()
                P.model.visited = set()
                # model.visited.add(start_node)
                # P.model.visited.add(start_node)
                model.obj = 0 
                model.state = model.build_state(model.init_state)
                # print(f"model.state.CurrentNode: {model.state.CurrentNode}")
                idx  = G.get_currency_name(model.state.CurrentNode)

                print("\tTheta {}  - Iteration {} - Stepsize {:.2f} - InitState {} - Vbar {:.2f}".format(model.theta_step,model.n,model.alpha(),model.state.CurrentNode,model.V_t[idx]))
                step = 1
                path = [model.G.get_currency_name(model.G.start_node)]
                while model.state.CurrentNode != model.init_args['target_node']:
                    # print(f"Current Node: {model.state.CurrentNode} and target node : {model.init_args['target_node']} ")
                    # input ()
                    # calling policy and choosing a decision
                    decision,vhat = P.make_decision(model)
                    path.append(decision)
                    print(f"*********** from {model.g.get_currency_name(model.state.CurrentNode)} ---->  {decision}")
                    x = model.build_decision({'NextNode': decision})
                    jdx = G.get_currency_name(model.state.CurrentNode)
                    
                    print("\t\t Theta {}  - Iteration {} - Step {} - Current State {} - vbar = {:.2f}".format(model.theta_step,model.n,step, model.state.CurrentNode,model.V_t[jdx]))
                    
                    vbar = model.update_VFA(vhat)

                    print("\t\tDecision={}, vhat {:.2f} and new vbar for current state {:.2f}".format(x[0],vhat,model.V_t[G.get_currency_name(model.state.CurrentNode)]))

                    # transition to the next state w/ the given decision
                    model.transition_fn(x)
                    step += 1

                    
                print("Finishing Theta {} and Iteration {} with {} steps. Total cost: {} \n".format(model.theta_step,model.n,step-1,model.obj))
                model.n+=1
                obj_along_iterations[theta].append(model.obj)
                # print(f"obj_along_iterations[theta].append(model.obj): {obj_along_iterations}")
                # print(f"model.origin_node: {model.origin_node}")
                crypto = model.g.get_currency_name(model.origin_node)
                # print(f"model.V_t[crypto]): {model.V_t[crypto]}")
                vbar_along_iterations[theta].append(model.V_t[crypto])
                # print(f"vbar_along_iterations: {vbar_along_iterations}")
                print(f"**********************The arbitrage path: {path}")
                paths[theta] = path
                print(f"********************** {model.obj} ")
                print(f"********************** {model.V_t[crypto]}")
                print(f"\n\n\n\n")
                # input()

            
            
        
            
        #Ploting the results
        fig1, axsubs = plt.subplots(1,2)
        fig1.suptitle('Comparison of theta^step - stepsize type {} - origin {}, target {}, dist {} '.format(M.init_args['stepsize_rule'],M.origin_node,M.target_node,M.dist) )
    
        color_list = ['b','g','r','m','c']
        nThetas = list(range(len(theta_list)))
        Iterations = list(range(model.init_args['nIterations']))

        # print("obj_along_iterations", obj_along_iterations)
        totals_obj = [np.array(obj_along_iterations[theta]).sum() for theta in theta_list]
        totals_V = [np.array(vbar_along_iterations[theta]).sum() for theta in theta_list]

            

        params_tester['seed'] = seed
        params_tester['costMin'] = -10
        params_tester['costMax'] = 10 
        params_tester ['deadlinePerc'] = 0.6
        params_tester ['maxVolatility'] = 100
        params_tester ['maxSpreadPerc'] = 100
        params_tester ['theta_set'] = parameters['theta_set'] 
        params_tester ['stepsize_rule'] = parameters['stepsize_rule'] 
        params_tester ['nIterations'] =  parameters['nIterations']

        params_tester = fetcher_tester.prepare_end_node(params_tester, start_node, end_node)
        crypto_graph_tester = CryptoGraph(params_tester, start=start_node, end=end_node,is_meanCosts_const=False, is_weight_const=False, is_meanPrice_const=False, is_step_const=True)
        
        # for theta in theta_list: 
        #     if (model.init_args['nIterations'] == 1): 
        #         print("********************** THETA -  ",theta)
        #         print("********************** OBJ -  ", obj_along_iterations[theta])
        #         print("********************** V_BAR -  ", vbar_along_iterations[theta])
        #         print("********************** PATH ",paths[theta])
        #     else: 
        #         print("********************** THETA -  ",theta)
        #         print("********************** OBJ -  ", obj_along_iterations[theta].sum())
        #         print("********************** V_BAR -  ", vbar_along_iterations[theta].sum())
        #         print("********************** PATH  ",paths[theta])

        for theta in theta_list: 
            if (model.init_args['nIterations'] == 1): 
                print("********************** THETA -  ",theta)
                print("********************** OBJ -  ", obj_along_iterations[theta])
                print("********************** V_BAR -  ", vbar_along_iterations[theta])
                print("********************** PATH  ",paths[theta])
            
                arb_cost = crypto_graph.apply_to_market(path)
                delta = np.exp(-arb_cost) - 1 
                print((f"********************** 'T'  delta of portfolio is {delta * 100} %").upper())
            
                arb_cost = crypto_graph_tester.apply_to_market(path)
                delta = np.exp(-arb_cost) - 1 
                print((f"********************** T+1' delta of portfolio is {delta * 100} %").upper())
                print('\n')


    
    # for theta,t in zip(theta_list,nThetas):
        

    #     axsubs[0].plot(Iterations, np.array(obj_along_iterations[theta]).cumsum(), "{}-".format(color_list[t]),label = "{}".format(theta))
    #     axsubs[0].plot(Iterations, obj_along_iterations[theta], "{}-".format(color_list[t]),label = "{}".format(theta))

    #     axsubs[0].set_title('Objective function')
    #     axsubs[0].legend()
    #     axsubs[0].set_xlabel('Iterations')
    #     axsubs[0].set_ylabel('Cost')

    #     axsubs[1].plot(Iterations, vbar_along_iterations[theta], "{}o-".format(color_list[t]),label = "{}".format(theta))
    #     axsubs[1].set_title('Vbar')
    #     axsubs[1].legend()
    #     axsubs[1].set_xlabel('Iterations')
    #     axsubs[1].set_ylabel('Cost')


    #     # axsubs.plot(Iterations, obj_along_iterations[theta], "{}o-".format(color_list[t]),label = "{}".format(theta))
    #     # axsubs[0].set_title('Cost')
    #     # axsubs.legend()
    #     # axsubs.set_xlabel('Iterations')
    #     # axsubs.set_ylabel('Cost')

    # plt.show()


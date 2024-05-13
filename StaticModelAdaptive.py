"""
Stochastic Shortest Path Extension
Using point estimates

"""
from collections import (namedtuple, defaultdict)

import numpy as np
import builtins


class StaticModel():
    """
    Base class for model
    
    """

    def __init__(self, state_names, x_names, params, G):
        """
        Initializes the model

        :param state_names: list(str) - state variable dimension names
        :param x_names: list(str) - decision variable dimension names
        :param s_0: dict - need to contain at least information to populate initial state using s_names
        :param exog_info_fn: function -
        :param transition_fn: function -
        :param objective_fn: function -
        :param seed: int - seed for random number generator
        """

        self.init_args = params
        # for param in params: 
        #     print(param)
        self.visited = set()

        # self.init_state = s_0
        self.prng = np.random.RandomState(self.init_args['seed'])
        self.state_names = state_names
        self.x_names = x_names
        self.State = namedtuple('State', state_names)
        self.Decision = namedtuple('Decision', x_names)

        # Creating the graph and computing V_0
        self.G = G
        self.g , self.V_t, self.origin_node, self.target_node, self.dist = self.createStochasticGraph()

        # now i have the graph, also origin and end node + dist


        #  push start and end nodes to params!
        self.init_args.update({'start_node':self.origin_node,'target_node':self.target_node})

        self.exog_info = self.g

        #Constructing the initial state
        self.init_state =  {'CurrentNode': self.init_args['start_node'], 'CurrentNodeLinksCost': self.exog_info_fn(self.init_args['start_node'])}
        self.state = self.build_state(self.init_state)
        # print(f"Initial State: {self.state} ")
        self.print_State()
        
        # value of objective function
        self.obj = 0.0
        
        # current iteration
        self.n = 1
        #The stepsize will be set outside the constructor
        self.theta_step = 1
        # policy function, given by Bellman's equation
        self.policy = None

        # print("Graph init was completed!")

        

    def build_state(self, info):
        return self.State(*[info[k] for k in self.state_names])

    def build_decision(self, info):
        # print(f"self.x_names: {self.x_names}")
        # print(f"info: {info}")
        res = self.Decision(*[info[k] for k in self.x_names])
        # print(f"self.Decision(*[info[k] for k in self.x_names]): {res}")
        return self.Decision(*[info[k] for k in self.x_names])
    
    def print_State(self):
        a  =1 
        # print(" CurrentNode: {} and costs on its edges: ".format(self.state.CurrentNode))
        # print(" ****",printFormatedDict(self.state.CurrentNodeLinksCost))
    
    def update_VFA(self,vhat):
        # print(f"----    UPDATE VFA  ----")
        # print(self.V_t)
        # print(self.state.CurrentNode)
        currency = self.g.get_currency_name(self.state.CurrentNode)
        # print(f"\t\t\t\t AlPHA = {self.alpha()} ")
        self.V_t[currency] = (1-self.alpha())*self.V_t[currency] + self.alpha()*vhat
        return self.V_t[currency]

    def exog_info_fn(self, i, type="gbm"):
        # print("FUCKING: i", i)
        if builtins.type(i) == int:
            i = self.g.get_currency_name(i)
        # if type(i) == int: 
        #     i = self.g.get_currency_name(i)
            
        # if type(i) != str: 
        #     i = self.g.get_currency_name(i)
        # print(f"self.g.edges[i]: {self.g.neighbors[i]}")
        weights = {} 
        for j in self.g.neighbors[i]: 
            if type == 'gbm': 
                new_price = self.g.generate_future_prices_gbm(np.exp(-1 * self.g.graph[i][j]['weight']), self.g.graph[i][j]['mean_price_log'], self.g.graph[i][j]['volatility_log'], num_periods=1)
                # print(f"OLD = {new_price[0]} NEW PRICE FOR {i} / {j} = {new_price[-1]}")
                weights[j] = -np.log(new_price[-1])
            else: 
                new_price = self.g_test.realCosts[(i, j)]
                # print(f"NEW PRICE FOR {i} / {j} = {new_price}")
                weights[j] = -np.log(new_price)
            
        # print(f"weights: {weights}" )
        return weights 

    def transition_fn(self, decision):
        # print(f"\n\n\n ---------transition_fn---------")
        # print(f"decision: {decision}")
        # print(f"self.obj: {self.obj}")
        # print(f"decision.NextNode: {decision.NextNode}")
        # print(f"self.state.CurrentNodeLinksCost :{self.state.CurrentNodeLinksCost}")
       # input("Press enter to continue building transition_fn")
        self.obj = self.obj + self.state.CurrentNodeLinksCost[decision.NextNode]
        # print(f"d     self.ob afte all : {self.obj}")
        self.state = self.build_state({'CurrentNode': self.g.get_currency_index(decision.NextNode), 'CurrentNodeLinksCost': self.exog_info_fn(decision.NextNode)})
        # print(f"self.state : {self.state }")
        return self.state

    def objective_fn(self):
        return self.obj



    def createStochasticGraph(self):
        # create a random graph of n nodes and make sure there is a feasible path from node '0' to node 'n-1' 
        g = self.G
        # g = randomgraphChoice(self.prng,self.init_args['nNodes'], self.init_args['probEdge'],self.init_args['LO_UPPER_BOUND'],self.init_args['HI_UPPER_BOUND'])
        # print("Created the graph")  

        maxSteps = 0
        max_origin_node = self.G.start_node
        # print(f"max_origin_node: {self.G.get_currency_name(max_origin_node)}, { max_origin_node }")
        max_target_node = self.G.end_node
        # print(f"max_target_node: {self.G.get_currency_name(max_target_node)}, {max_target_node}")

        max_node, max_dist = self.G.true_bellman_ford(max_target_node)
        # print(f"max_node, max_dist: {max_node, max_dist}")
        maxSteps = max_dist


        # for target_node in g.nodes:
        #     # find the max number of steps bewteen to the target_node and the origin_node that achieves that
        #     max_node,max_dist = g.truebellman(target_node)
            
        #     if max_dist > maxSteps:   
        #         maxSteps = max_dist
        #         max_origin_node = max_node
        #         max_target_node = target_node

        # print("origin_node: {} -  target_node: {}  - distance: {}".format(max_origin_node,max_target_node,maxSteps))
        # print("origin_node: {} -  target_node: {}  - distance: {}".format(self.G.get_currency_name(max_origin_node) ,self.G.get_currency_name(max_target_node),maxSteps))

        V_0 = g.bellman(max_target_node)

        # print("Computed V_0")
        # print(printFormatedDict(V_0))

        return g,V_0,max_origin_node,max_target_node,maxSteps




    def alpha(self):
        if self.init_args['stepsize_rule']=='Constant':
            return self.theta_step
        else:
            return self.theta_step  

    




# Stochastic Graph class 
class StochasticGraph:
    def __init__(self):
        self.nodes = list()
        self.edges = defaultdict(list)
        self.lower = {}
        self.distances = {}
        self.upper = {}

    def add_node(self, value):
        self.nodes.append(value)
    
    # create edge with normal weight w/ given mean and var
    def add_edge(self, from_node, to_node, lower, upper):
        self.edges[from_node].append(to_node)
        self.distances[(from_node, to_node)] = 1
        self.lower[(from_node, to_node)] = lower
        self.upper[(from_node, to_node)] = upper
    
    # return the expected length of the shortest paths w.r.t. given node
    def bellman(self, target_node):
        inflist = [np.inf]*len(self.nodes)
        # vt - value list at time t for all the nodes w.r.t. to target_node
        vt = {k: v for k, v in zip(self.nodes, inflist)}
        vt[target_node] = 0
        
        # decision function for nodes w.r.t. to target_node
        dt = {k:v for k,v in zip(self.nodes, self.nodes)}
        
        # updating vt
        for t in range(1, len(self.nodes)):            
            for v in self.nodes:
                for w in self.edges[v]:
                    # Bellman' equation 
                    if (vt[v] > vt[w] + 0.5*(self.lower[(v,w)] + self.upper[(v,w)])):
                        vt[v] = vt[w] + 0.5*(self.lower[(v,w)] + self.upper[(v,w)])
                        dt[v] = w 
        # print(vt)
        # print(g.distances)
        return(vt)   
    
    def truebellman(self, target_node):
        inflist = [np.inf]*len(self.nodes)
       # input("Press enter to continue!")
        # vt - list for values at time t for all the nodes w.r.t. to target_node
        vt = {k: v for k, v in zip(self.nodes, inflist)} 
        vt[target_node] = 0
        
        # decision function for nodes w.r.t. to target_node
        dt = {k:v for k,v in zip(self.nodes, self.nodes)}
        
         # updating vt
        for t in range(1, len(self.nodes)):            
            for v in self.nodes:
                for w in self.edges[v]:
                    # Bellman' equation 
                    if (vt[v] > vt[w] + self.distances[(v, w)]):
                        vt[v] = vt[w] + self.distances[(v, w)]
                        dt[v] = w 
        # print(vt)
        # print(g.distances)

        v_aux = {k:-1 if v == np.inf else v for k,v in vt.items()}
        max_node = max(v_aux, key=v_aux.get)
        max_dist = v_aux[max_node]

        return(max_node,max_dist)  
    




def printFormatedDict(dictInput):
    nodeList = [node for node in dictInput.keys()] 

    for node in nodeList:
        a = 1
        # print("\t\tkey_{} = {:.2f}".format(str(node),dictInput[str(node)]))

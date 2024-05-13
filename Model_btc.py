

from collections import (namedtuple, defaultdict)
import math
import numpy as np
import pandas as pd
import xlrd
import time
from Policy import LookaheadPolicy

    
# Ensure to handle cases where no unvisited neighbors are available

class StaticModel_Dynamic():
    """
    Base class for the static model
    """
    def __init__(self, state_names, x_names, s_0, params, G):
        """
        Initializes the model
        :param state_names: list(str) - state variable dimension names
        :param x_names: list(str) - decision variable dimension names
        :param s_0: dict - contains the inital state information
        :param s_0[meanCosts]: dict- meanCosts[k][l] is the mean of the cost on the link k-l 
        :param s_0[spreads]: dict - spreads[k][l] represents the spread of the distribution of  
        cost on link k-l 
        :param Horizon: int - the horizon over which we are looking ahead
        :param vertexCount - the number of nodes in our network 
        :param seed: int - seed for random number generator
        """
        self.init_args = params
        self.visited = set()  # Set to track visited nodes
        
        self.init_state = s_0
        self.state_names = state_names
        self.State = namedtuple('State', state_names)
        self.state = self.build_state(self.init_state)
        self.x_names = x_names
        self.Decision = namedtuple('Decision', x_names)
        
        self.G = G
        self.theta = 0.5
        self.n = 0
        self.time = 1
        self.obs = 1
        self.estimated_costs = defaultdict(dict)
        self.costs = defaultdict(dict)
        self.prng = np.random.RandomState(params['seed'])
        
        
    def start_new_theta(self,theta):
        """This method generates new delta. Delta is a learning rate!"""
        self.theta = theta
        self.estimated_costs = defaultdict(dict)
        self.n = 0
        self.obs = 1
        self.prng = np.random.RandomState(self.init_args['seed'])
        # print("*****************Reseting model for theta {:.2f}".format(self.theta))
        

    def update_estimated_costs(self):
        alpha = self.get_step_size()
        
        self.G.update_graph(self.theta)
        for k in range(self.G.vertexCount ):
            k = self.G.get_currency_name(k)
            for l in self.G.neighbors[k]:
                m_hat = self.sample_from_uniform(k,l)	
				

                if self.G.is_meanCosts_const: 
                    m_hat = self.G.meanCosts[(k, l)]
                elif self.G.is_meanPrice_const: 
                    m_hat  = self.G.graph.get_edge_data(k, l)['mean_price']
                else: 
                    self.estimated_costs[k][l] = self.G.meanCosts[(k, l)]
                    continue

                if alpha < 1:
                        self.estimated_costs[k][l] = (1-alpha)* self.estimated_costs[k][l] + alpha * m_hat
                else:
                    self.estimated_costs[k][l] = m_hat
        
        self.estimated_costs[self.G.end_node][self.G.end_node] = 0
        
    def sample_from_uniform(self,fromNode,toNode):
        # return self.G.meanCosts[(fromNode, toNode)]

        if self.G.is_meanCosts_const: 
            m_hat = self.G.meanCosts[(fromNode, toNode)]
        elif self.G.is_meanPrice_const: 
            m_hat  = self.G.graph.get_edge_data(fromNode, toNode)['mean_price']

        spread = self.G.spreads[fromNode][toNode]
        # Изменение строк
        deviation = self.prng.uniform(- spread, spread) * self.G.meanCosts[(fromNode, toNode)]
        m_hat = self.G.meanCosts[(fromNode, toNode)] + deviation    

        # print(f"self.G.meanCosts[(fromNode, toNode)]: {self.G.meanCosts[(fromNode, toNode)]} ||| m_hat: {m_hat}")

        return m_hat


    def get_step_size(self):
        #alpha = 1/self.n
        #alpha = 1./self.time
        alpha = 1./self.obs
        return alpha    


    def build_state(self, info):
        res = self.State(*[info[k] for k in self.state_names])
        return res

    def build_decision(self, info):
        return self.Decision(*[info[k] for k in self.x_names])

    # exog_info_fn: function - returns the real experienced cost of traversing a link 
    # from 'fromNode' to 'toNode' 
    def exog_info_fn(self, fromNode, toNode):
        cost_hat = self.sample_from_uniform(fromNode,toNode)
        return cost_hat

    # transition_fn: function - updates the state within the model andeturns new state
    def transition_fn(self, decision):
        self.state = self.build_state({'node':decision})
        self.time += 1
        self.obs += 1
        self.visited.add(decision)  # Add to visited set

        
    # :param objective_fn: function - returns the cost we would experience by taking 'decision'
    # as our next node from the current state 'state'
    def objective_fn(self, decision):
        cost = self.exog_info_fn(self.state.node, decision)
        return cost 
    '''
    the function for running trials; it simulates solving the problem a bunch of 
    times (nbTrials times), then takes the squared mean of the costs incurred, 
    and then returns that mean value
    '''
    def runTrials(self,nbTrials,deadline):
        totalPenalty = 0.0
        totalCost = 0.0
        totalTime = 0.0

        for i in range(nbTrials):
            way = [] 
            way.append(self.init_state)
            self.state = self.build_state(self.init_state)
            # print("self.state: ", self.state)
            self.time = 1
            self.n += 1
            cost=0.0
            costs = []
            

            #Following a path  - the policy function is a lookahead 
            end_node = self.G.get_currency_name(self.G.end_node)
            # print("end_node: ", end_node)
            while self.state.node != end_node:
                # print(f"1. Update estimated costs! ")
                self.update_estimated_costs()
                # print(self.estimated_costs)
                P = LookaheadPolicy(self) 
                # print(f"2. Make a decision what cryptocurrency to go!")
                decision = P.get_decision("PERCENTILE")    
                
                way.append(decision)
                #self.build_decision({'nextNode':decision})
                stepCost = self.objective_fn(decision)
                # print(f"---> Decision from {self.state.node} is {decision} for {stepCost}")
                cost += stepCost
                # print(cost, stepCost)
                costs.append(stepCost)
                #print("\t Theta {:.2f}, Iteration {}, Time {}, CurrNode {}, Decision {}, Step Cost {:.2f} Cum Cost {:.2f}".format(self.theta,self.n,self.time,self.state.node,decision,stepCost,cost))
                self.transition_fn(decision)
                # print("decision: ", decision, "self.state.node:", self.state.node, "end_node: ", end_node)
                # input() 
                # print("UPDATE COSTS")
                self.update_estimated_costs()

            #end of path calculations
            totalCost += cost

            # if cost > deadline:
            #    #latenessSquared = (cost - deadline) ** 2
            #    latenessSquared = 1
            #    totalPenalty += latenessSquared
            # else:
            #    latenessSquared=0
            latenessSquared=0
            totalTime += self.time-1
            
            # print("End of Theta {:.2f}, Iteration {}. Cost: {:.2f}, Lateness: {:.2f} and number of steps {}".format(self.theta,self.n,cost,math.sqrt(latenessSquared),self.time-1))
            # print(way)
            # print(costs)
            self.visited = set()

        # print(f"print costs: {cost}")
        
        #end of trials
        avgCost = totalCost/nbTrials
        avgPenalty = totalPenalty / nbTrials
        avgTime = totalTime / nbTrials

        # print(" self.theta: ",  self.theta)
        way[0] = way[0]['node']
        return avgCost,avgPenalty,avgTime, way
            
                    
                    
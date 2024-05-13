import numpy as np

from collections import (namedtuple, defaultdict)

class Policy():
    """
    Base class for Static Stochastic Shortest Path Model policy
    """

    def __init__(self, model, policy_names):
        """
        Initializes the policy
        :param policy_names: list(str) - list of policies
        :param model: the model that the policy is being implemented on
        """
        self.model = model
        self.policy_names = policy_names
        self.Policy = namedtuple('Policy', policy_names)

    def build_policy(self, info):
        # this function builds the policies depending on the parameters provided
        # print(self.Policy(*[info[k] for k in self.policy_names]))
        return self.Policy(*[info[k] for k in self.policy_names])


    def make_decision(self,M):
        i = M.state.CurrentNode
        i = M.g.get_currency_name(i)
        
        costs = {} 
    #     print(f"M.state.CurrentNodeLinksCost : {M.state.CurrentNodeLinksCost}")
    #    # input()
    #     print(f"M.V_t: {M.V_t}")
        
       # input()
        for j in M.g.neighbors[i]: 
           
            costs[j] = M.state.CurrentNodeLinksCost[j] +  M.V_t[j]
            # print(f"costs[j] = M.state.CurrentNodeLinksCost[j] +  M.V_t[j]")
       
            # print(f"{costs[j]} = {M.state.CurrentNodeLinksCost[j]} + {M.V_t[j]}") 
        # costs = {j:M.state.CurrentNodeLinksCost[j] + M.V_t[j] for j in M.g.neighbors[i]}

        # print("all costs!", costs)
        
        unvisited_costs = {decision: cost for decision, cost in costs.items() if decision not in self.model.visited}
        # input()
        # print(f"unvisited: {unvisited_costs}")
        # print(f"visited: {self.model.visited}")
        # input()
        optimal_decision = min(unvisited_costs, key=costs.get)
        self.model.visited.add(optimal_decision)

        # rint(f"optimal_decision: {optimal_decision}, {costs[optimal_decision]}")
        
        return optimal_decision, costs[optimal_decision]
    
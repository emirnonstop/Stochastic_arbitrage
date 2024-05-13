''' 

The code for the lookahead policy we use in our 
Static Model

'''
import time 
import numpy as np

# the lookahead policy
class LookaheadPolicy():
    def __init__(self, model):
        self.model = model
    
    def get_decision(self, METRIC):
       
        node_index = self.model.G.get_currency_index(self.model.state.node)  # Получение индекса текущего узла
        decisions = [[0] * self.model.G.vertexCount for _ in range(self.model.G.Horizon + 1)]
        # print("self.model.G.vertexCount", self.model.G.vertexCount)
        # print("self.model.G.Horizon + 1)]", self.model.G.Horizon + 1)

        # for i in (decisions): 
        #     print (i)
        V = np.ones((self.model.G.Horizon + 1, self.model.G.vertexCount)) * np.inf
        # print("self.model.G.Horizon + 1)]", self.model.G.Horizon + 1)
        # print("self.model.G.vertexCount", self.model.G.vertexCount)

        # print ("self.model.G.end_node: ",self.model.G.end_node )
        V[:, self.model.G.end_node] = 0  # Costs at the destination are zero

        # Algorithm uses the "stepping backwards in time" method
        lookAheadTime = self.model.G.Horizon - 1

        while lookAheadTime >= 0:
            for k in self.model.G.vertices:
                k_index = self.model.G.get_currency_index(k)
                # k  = self.model.G.get_currency_index(k)  # Получение индекса текущего узла
                # print ("K = ", k)
                argMin = -1
                minVal = np.inf
                # print(f"Global search for {k} || self.model.G.neighbors = {self.model.G.neighbors[k]})")
                # print ("G.neighbors:", self.model.G.neighbors)
                for l in self.model.G.neighbors[k]:
                    # print(f"I consider the crypto {l}")
                    l_index = self.model.G.get_currency_index(l)  # Получение индекса текущего узла
                   
                    if l not in self.model.visited:  
                        if METRIC == "PERCENTILE": 
                            
                            volatility = self.model.G.volatilities[(k, l)]
                            # print(f"k: {k} ||| l : {l}")
                            mean = self.model.estimated_costs[k][l]
                            # print("\n GLOBAL CHECK ->>>>>>")
                            # print(mean)
                            # print(self.model.G.meanCosts[(k, l)])
                            # print("GLOBAL CHECK ->>>>>>")
                            l_n = l
                            l = self.model.G.get_currency_index(l)  # Получение индекса текущего узла
                            # print(f"this is a node number l: {l} {l_n} ")

                            transition_cost = V[lookAheadTime + 1][l] + self.use_percentile_val(self.model.theta, volatility, mean)
                            # print(f"transition_cost from {k} to {self.model.G.get_currency_name(l)} is {transition_cost} ")
                        else:
                            l_index = self.model.G.get_currency_index(l)  # Получение индекса текущего узла
                            # print(f"{k} -> {l} the transaction_cost = {transition_cost}")
                            transition_cost = V[lookAheadTime + 1][l_index] + self.model.G.meanCosts[(k, l)]
                            # print(f"{k} -> {l} the transaction_cost = {transition_cost} and  self.model.G.meanCosts[(k, l)] =  {self.model.G.meanCosts[(k, l)]} ")
                            l = l_index
                        if minVal > transition_cost:
                            # print(f"Update state! {self.model.G.get_currency_name(l)} ")
                            argMin = l
                            minVal = transition_cost
                k = self.model.G.get_currency_index(k)  # Получение индекса текущего узла
                V[lookAheadTime][k] = minVal
                decisions[lookAheadTime][k] = argMin
            lookAheadTime -= 1

        # for v in V:
        #     print(v)
    
        # print(f"All decisions")
        # for decision in decisions: 
        #     currency_names= []
        #     for dec in decision: 
        #         currency_names.append(dec)
        #     print(currency_names)

        node = self.model.state.node
        # print(node)
        node = self.model.G.get_currency_index(node)
    
        # if METRIC == "PERCENTILE": 
        decision = self.model.G.get_currency_name(decisions[0][node])
        # else: 
        #     decision = decisions[0][node]
        # print(decision)
        return decision

    def use_percentile_val(self, theta, spread, mean):
        mean = np.exp(-mean)
        point_val = 1 - spread + (2 * spread) * theta
        used_cost = mean * point_val
        used_cost = -np.log(used_cost)
        return used_cost

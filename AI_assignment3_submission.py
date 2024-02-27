from random import choices, randint
from random import choice
import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    nodes = ["alarm","faulty alarm", "gauge","faulty gauge", "temperature"]    
    BayesNet = BayesianModel()
    for i in range(len(nodes)):
        BayesNet.add_node(nodes[i])
    BayesNet.add_edge("temperature","faulty gauge")
    BayesNet.add_edge("faulty gauge","gauge")
    BayesNet.add_edge("temperature","gauge")
    BayesNet.add_edge("gauge","alarm")
    BayesNet.add_edge("faulty alarm","alarm")
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    cpd_t = TabularCPD('temperature', 2, values=[[0.8], [0.2]]) 
    cpd_fa = TabularCPD('faulty alarm', 2, values=[[0.85], [0.15]])   
    cpd_fg = TabularCPD('faulty gauge', 2, values=[[0.95, 0.20], \
                    [0.05, 0.80]], evidence=['temperature'], evidence_card=[2])
    cpd_g = TabularCPD('gauge', 2, values=[[0.95, 0.05, 0.20, 0.80], \
                    [0.05, 0.95, 0.80, 0.20]], evidence=['faulty gauge','temperature'], evidence_card=[2, 2])
    cpd_a = TabularCPD('alarm', 2, values=[[0.90, 0.10, 0.55, 0.45], \
                    [0.10, 0.90, 0.45, 0.55]], evidence=['faulty alarm', 'gauge'], evidence_card=[2, 2])
    bayes_net.add_cpds(cpd_t, cpd_fg, cpd_g, cpd_fa,cpd_a)
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    alarm_prob = marginal_prob['alarm'].values
    return alarm_prob[1]


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    gauge_prob = marginal_prob['gauge'].values
    return gauge_prob[1]


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'],evidence={'alarm':1,'faulty alarm':0,'faulty gauge':0}, joint=False)
    temp_prob = conditional_prob['temperature'].values
    return temp_prob[1]


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    nodes = ["A","B", "C","AvB","BvC","CvA"]   
    for i in range(len(nodes)):
        BayesNet.add_node(nodes[i])
    BayesNet.add_edge("A","AvB")
    BayesNet.add_edge("B","AvB")
    BayesNet.add_edge("B","BvC")
    BayesNet.add_edge("C","BvC")
    BayesNet.add_edge("A","CvA")
    BayesNet.add_edge("C","CvA")

    cpd_a = TabularCPD('A', 4, values=[[0.15], [0.45],[.30],[.10]])     
    cpd_b = TabularCPD('B', 4, values=[[0.15], [0.45],[.30],[.10]])        
    cpd_c = TabularCPD('C', 4, values=[[0.15], [0.45],[.30],[.10]]) 
  
    cpd_AvB = TabularCPD('AvB', 3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1], \
                    [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1],\
                    [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]], evidence=['A','B'], evidence_card=[4,4])
    cpd_BvC = TabularCPD('BvC', 3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1], \
                    [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1],\
                    [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]], evidence=['B','C'], evidence_card=[4,4])
    cpd_CvA = TabularCPD('CvA', 3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1], \
                    [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1],\
                    [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]], evidence=['C','A'], evidence_card=[4,4])

    BayesNet.add_cpds(cpd_a, cpd_b, cpd_c, cpd_AvB, cpd_BvC, cpd_CvA)

    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]    
    solver = VariableElimination(bayes_net)
    for i in range(3):
        A_CA = solver.query(variables=['AvB'],evidence={'BvC':i,'CvA':2}, joint=False)
        B_CA = solver.query(variables=['BvC'],evidence={'CvA':2}, joint=False)
        B_C = solver.query(variables=['AvB'],evidence={'CvA':2}, joint=False)
        posterior[i] = A_CA['AvB'].values[0]* B_CA['BvC'].values[i]/B_C['AvB'].values[0]
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """ 
    
    if initial_state  == None or initial_state == []:
        A = randint(0,3)
        B = randint(0,3)
        C = randint(0,3)
        AvB = 0
        BvC = randint(0,2)
        CvA = 2
        initial_state = [A, B, C, AvB, BvC, CvA]

    #select A, B, or C uniformly at random
    ind = 3
    while ind==3 or ind==5:
        ind = randint(0,5)
    A_cpd = bayes_net.get_cpds('A')      
    team_table = A_cpd.values
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    
    #P(A | B, C, AvB, CvA)
    #P(A|B, C, AvB=0, BvC, CvA=2) -> P(A|B, AvB=0) P(A|C, CvA=2)
    #P(B | A, C, AvB, BvC) -> P(B|A, AvB=0) P(B|C, BvC)
    #P(C | A, B, BvC, CvA) -> P(A|B, AvB=0) P(C|A, CvA=2)
    #(BvC | B, C)
    
    probs = []
    nums = []
    
    if ind < 3:
        probs = [0,0,0,0]
        nodes = [0,1,2]
        nodes.pop(ind)
        for i in range(4):
            nums.append(i)
            if ind == 0:#AvB[A,B]*CvA[B,A]
                matches = match_table[initial_state[3],i,initial_state[nodes[0]]] *match_table[initial_state[5],initial_state[nodes[1]],i]
            elif ind == 1:#AvB[A,B]*BvC[B,A]
                matches = match_table[initial_state[3],initial_state[nodes[0]],i] * match_table[initial_state[initial_state[4],i,nodes[1]]] 
            elif ind == 2:#AvB[A,B]*BvC[B,C]
                matches = match_table[initial_state[4],initial_state[1],i] * match_table[initial_state[5],i,initial_state[0]]
            P_t = team_table[i]
            probs[i] = P_t * team_table[initial_state[nodes[0]]] * team_table[initial_state[nodes[1]]] * matches
            
    else:        
        probs = [0,0,0]
        for i in range(3):
            nums.append(i)
            probs[i] = team_table[initial_state[1]] * team_table[initial_state[2]] * match_table[i,initial_state[1],initial_state[2]]
    
    
    sum_probs = sum(probs)
    probs = probs/sum_probs
    index = choices(nums,probs)
    initial_state[ind] = index[0]
    
    sample = tuple(initial_state)
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values    
    if initial_state  == None or initial_state == []:
        A = randint(0,3)
        B = randint(0,3)
        C = randint(0,3)
        AvB = 0
        BvC = randint(0,2)
        CvA = 2
        initial_state = [A, B, C, AvB, BvC, CvA]
    A = randint(0,3)
    B = randint(0,3)
    C = randint(0,3)
    AvB = 0
    BvC = randint(0,2)
    CvA = 2
    cand_state = [A, B, C, AvB, BvC, CvA]
    #old jumping distribution
    pold = team_table[initial_state[0]]
    sample = tuple(initial_state)    
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    raise NotImplementedError        
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    raise NotImplementedError
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    return "Shruti Mhasawade"

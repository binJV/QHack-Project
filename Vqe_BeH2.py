#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt


# In[2]:


symbols= ["H","Be","H"]
coordinates = np.array([[ 0.69683,   1.16085,   0.],[-0.49317, 1.16085,0. ],[-1.68317,   1.16085,   0.]])
#qchem library computes our required no of Qubits and the hamiltonian of molecule
Ham, qubits= qchem.molecular_hamiltonian(symbols,coordinates,charge=0)
print(qubits,Ham)


# In[3]:


hf = qchem.hf_state(electrons=6, orbitals = qubits)
print(hf)


# In[4]:


num_wires =qubits
dev = qml.device("default.qubit", wires = num_wires)
@qml.qnode(dev)
def exp_energy(state):   #the state is representates as a jordan wigner representation
    qml.BasisState(np.array(state), wires= range(num_wires)) #prepare state
    return qml.expval(Ham)


# In[5]:


exp_energy(hf)


# In[6]:


# Step 2: Prepare a trial Ground state as the above energe is not ground state energy
def ansatz(params):
    qml.BasisState(hf,wires=range(num_wires))
    qml.DoubleExcitation(params[0], wires=[0,1,6,7])
    qml.DoubleExcitation(params[1], wires=[0,1,8,9])
    qml.DoubleExcitation(params[2], wires=[0,1,10,11])
    qml.DoubleExcitation(params[3], wires=[0,1,12,13])
    qml.DoubleExcitation(params[4], wires=[2,3,6,7])
    qml.DoubleExcitation(params[5], wires=[2,3,8,9])
    qml.DoubleExcitation(params[6], wires=[2,3,10,11])
    qml.DoubleExcitation(params[7], wires=[2,3,12,13])
    qml.DoubleExcitation(params[8], wires=[4,5,6,7])
    qml.DoubleExcitation(params[9], wires=[4,5,8,9])
    qml.DoubleExcitation(params[10], wires=[4,5,10,11])
    qml.DoubleExcitation(params[11], wires=[4,5,12,13])
    
    
    


# In[12]:


#Minimizing the Expectation Value of the Hamiltonian
#build a circuit that returns the expectation value of hamiltonian and returns a ground state  that is a cost function
@qml.qnode(dev)
def cost_function(params):
    ansatz(params)
    return qml.expval(Ham)



# In[19]:


# minimize the cost function or the expectation value what we mean is that based on theta of double excitation states our
#probabilities of the created state is dependent on these theta so a particular amount of contribution will give minimum energy.

theta = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],requires_grad = True)  # start at theta = -2 and then use opt to reach minima with some stepsize by some iteration
opt = qml.GradientDescentOptimizer(stepsize=0.1)
energy = [cost_function(theta)]
angle = [theta]
max_iterations = 100


for n in range (max_iterations):
    theta, prev_cost = opt.step_and_cost(cost_function,theta)
    energy.append(cost_function(theta))
    angle.append(theta)
    
    
    if n%2==0:    #prints theta and cost after every 10 iterations
        print(f"Step = {n}, Energy = {energy[-1]:.8f} Ha")


# In[111]:


new=np.zeros(max_iterations+1)
print(angle[3][3])
for i in range (max_iterations+1):
    new[i]=angle[i][4]
plt.plot(new,energy)
#plt.plot(angle,energy)


# In[40]:


print(f"Final Ground energy:  {energy[-1]:.8f} Ha")
print(f"Final angle Parameters: = {theta[0]:.8f} {theta[1]:.8f} {theta[2]:.8f} {theta[3]:.8f} {theta[4]:.8f} {theta[5]:.8f} {theta[6]:.8f} {theta[7]:.8f} {theta[8]:.8f} {theta[9]:.8f} {theta[10]:.8f} {theta[11]:.8f} ")


# In[28]:


# what is actual ground state? ask from pennylane
@qml.qnode(dev)
def ground_state(params):   
    ansatz(params)
    return qml.state()


# In[29]:


ground_state(theta)


# In[ ]:





# In[ ]:





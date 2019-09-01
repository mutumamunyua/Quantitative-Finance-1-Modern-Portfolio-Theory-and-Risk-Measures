#!/usr/bin/env python
# coding: utf-8


#This code is based on modern portfolio theory


#Load necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
import scipy
import scipy.linalg
import scipy.optimize as scopt
from scipy.optimize import minimize
from scipy.stats import norm
import math
from numpy import linalg as LA
import scipy.linalg as la
from math import fabs
import mpld3
from mpld3 import plugins

np.random.seed(123)

# Turn off progress printing 
solvers.options['show_progress'] = False


#Part 1: Modern Portfolio Theory and Risk Measures
## NUMBER OF ASSETS
assets = ['A', 'B', 'C']

# load simulated data with specified mean, sd and correlation
stdata = pd.read_csv('/Users/------.csv') 

# subset assets data only
dat = stdata.loc[:,'A':'C']

# calculate mean return
returns = dat.mean() 
#Display results
print(returns)

# get covariance matrix
cov = dat.cov()
#Display results
print(cov)

#  create a random set of portfolios by varying simulated weights, here 5000
np.random.seed(42)
num_ports = 5000
all_weights = np.zeros((num_ports, len(assets)))

# Define Confidence Interval
alpha = 0.99

#create buckets for output data
#return
ret_arr = []

#volatility
vol_arr = []

# Weight
weights_all = []

#Sharpe Ratio
sharpe_ratio = []

#Value at Risk-VAR
VAR = []

#Expected Shortfall-ES
ES = []


#Simulate weights and get returns and volatilities, VAR and ES
# the loop will create several weights and get parameters of interest

for i in range(num_ports):
    # Weights
    weights = np.array(np.random.random(3))
    weights = weights/np.sum(weights)
    
    # Expected return
    ret = np.dot(weights, returns)
    #print(ret)
    
    # Expected volatility
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    
    #Sharpe ratio
    sharpe = ret / vol
    
    # Compute VAR
    val = -ret + norm.ppf(alpha)*vol
    #print(VAR)
    
    # Compute ES
    expsh =  -ret + (1-alpha)**-1 * norm.pdf(norm.ppf(alpha)) * vol  
    
    # Append all data to buckets     
    ret_arr.append(ret)
    vol_arr.append(vol)
    weights_all.append(weights)
    sharpe_ratio.append(sharpe)
    VAR.append(val)
    ES.append(expsh)
    
#create a dictionary for returns, volatilities, sharpe ratio,
# VAR and ES for all portfolios
portfolio = {'Returns': ret_arr,
            'Volatility': vol_arr,
            'Sharpe Ratio': sharpe_ratio,
            'Value at Risk': VAR,
            'Expected Shortfall': ES}

# add ticker and weight in the portfolio dictionary
for counter,symbol in enumerate(assets):
    portfolio[symbol+'Weight'] = [Weight[counter] for Weight in weights_all]

#Change data to pandas dataframe and label df
df = pd.DataFrame(portfolio)

#label
column_order = ['Returns', 'Volatility', 'Sharpe Ratio', 'Value at Risk', 'Expected Shortfall'] + [stock+'Weight' for stock in assets]


#reorder dataframe columns
df = df[column_order]
#print(df.head())

# find the minimum variance portfolio
min_volatility = df['Volatility'].min()

# use the min values to locate and create the special portfolio
min_variance_port = df.loc[df['Volatility'] == min_volatility]
#Display results
print('Minimum Variance is:', min_volatility)

# #Display details of the min var portfolios
print(min_variance_port.T)


# To plot the efficient frontier we define several functions
#create a function to get the return volatility and 
#sharpe ratio from a given set of weights

#function to get the return volatility
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(dat.mean() * weights) 
    vol = np.sqrt(np.dot(weights.T, np.dot(dat.cov(), weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])

#function get the negative sharpe ratio (this will be used for minimization)
def neg_sharpe(weights):
# 2 is the sharpe ratio index from the get_ret_vol_sr
    return get_ret_vol_sr(weights)[2] * -1

# Ensure sum of weights equals 1
def check_sum(weights):
    #return 0 if sum of the weights is 1
    return np.sum(weights)-1

# Define optimization variables and assign initial weights
cons = ({'type':'eq','fun':check_sum})
bounds = ((0,1), (0,1), (0,1))
init_guess = [0.3, 0.3, 0.4]

# Optimize to get the min-var portfolio using Sequential Least Squares
# Programming Method (SLSQP)

opt_results = minimize(neg_sharpe, init_guess, method = 'SLSQP', bounds=bounds, constraints=cons)
print(opt_results)

get_ret_vol_sr(opt_results.x)

# Get the efficient frontier
# Define the return-risk plane boundaries

frontier_y = np.linspace(0.1, 0.30, 200)

#define a function to minimize volatility
def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]

frontier_x = []

#Get the efficient frontier

for possible_return in frontier_y:
    cons = ({'type':'eq', 'fun':check_sum},
            {'type':'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    result = minimize(minimize_volatility,init_guess,method='SLSQP', bounds=bounds, constraints=cons)
    frontier_x.append(result['fun'])

# plot frontier, min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='red', marker='D', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Opportunity Set with Efficient Frontier')
plt.plot(frontier_x,frontier_y, 'r--', linewidth=3)
plt.show()
print('(a) Scatter plot of portfolios with efficient frontier shown in dashed red line')
print('(b)The approximate location of the minimum variance portfolio is shown by the red dot')


#Show the head of dat showing VARs and ES
print('data head showing imputed VARs and ES', df.head(5))

# find the minimum VAR and ES
min_var = df['Value at Risk'].min()
min_es = df['Expected Shortfall'].min()

# use the min values to locate and create the two special portfolios
min_var_port = df.loc[df['Value at Risk'] == min_var]
min_es_port = df.loc[df['Expected Shortfall'] == min_es]

print('(c) Portfolios with te smallest VAR', min_var_port.T)
print('(c) Portfolios with te smallest ES', min_es_port.T)


# In[53]:


# (e) Determine if the correlation matrix ρ is positive definite

rho = np.matrix([[1,-0.3,-0.5],[-0.3,1,-0.6],[-0.5,-0.6,1]])

#calculate the eigen values that satisfy,  det(A − λI) = 0 and then calulate 
# eigen vectors x where (A − λI)x = 0
# Using numpy

w, v = LA.eig(rho)

print('Eigen Values for rho are :', w)
print('The Eigen vectors for rho are:', v)
print('The correlation matrix (rho) is symmetric and positive definite')


# In[54]:


#take a 'Cholesky' decomposition:

# The Gershgorin theorem applied to rho implies that the eigenvalues lie within
# the union of D1(1, r = 0.8), D2(1, r = 0.9), and D3(1, r = 1.1). 
# There are three eigenvalues in D1UD2UD3 D(1, r = 0.8), D(1, r = 0.9), and D(1, r = 1.1).

L = la.cholesky(rho)
print(L)

# Get lower triangle decomposition
Ll = scipy.linalg.cholesky(rho, lower=True)
print(Ll)

print('There are 3 eigen values in the union disks D1, D2 and D3')


# In[55]:


# Question 2: VAR and ES sensitivities

## NUMBER OF ASSETS
assets2 = ['1', '2', '3']
num2 = len(assets)
#print(num2)


# Step 1 input data for 3 stocks A, B and C
wgts = np.array([0.5,0.2,0.3])
mu2 = np.array([0, 0, 0])
vol2 = np.array([0.3,0.2,0.15])
vol_d = np.diag(vol2)
#print(vol_d)

rho2 = np.matrix([[1,0.8,0.5],[0.8,1,0.3],[0.5,0.3,1]])

cov2 = np.dot(vol_d, np.dot(rho2, vol_d))
#print(cov2)
covv = np.squeeze(np.asarray(cov2))

# Expected volatility of Portfolio
sigma2 = np.sqrt(np.dot(wgts.T, np.dot(covv, wgts)))
#print(sigma2)


cov_asset_Portfolio = np.dot(wgts.T, covv)


VAR_p = []
ES_p = []

#Calculate the VAR and ES sensitivies and 

for i in range(num2):
    # VAR (i)
    varr = -mu2[i] + norm.ppf(alpha)*wgts[i]*(sigma2)**-1
    
    # Compute ES
    ess =  -mu2[i] + ((1-alpha)*(sigma2))**-1 * norm.pdf(norm.ppf(alpha)) * wgts[i]  
    
    #Add Data to Buckets
    VAR_p.append(varr)
    ES_p.append(ess)

print('VAR Sensitivities:', VAR_p)
print('ES Sensitivitoes:', ES_p)


# In[56]:


#create a dictionary for Parial VARS and ES for all portfolios
partialVARES = {'VAR Sensitivities': VAR_p,
            'ES Sensitivities': ES_p}

print(partialVARES)


# In[57]:


# (2b) Determine if the correlation matrix ρ is positive definite

#calculate the eigen values that satisfy,  det(A − λI) = 0 and then calulate 
# eigen vectors x where (A − λI)x = 0
# Using numpy

w2, v2 = LA.eig(rho2)

print('Eigen Values for rho2 are :', w2)
print('The Eigen vectors for2 rho are:', v2)


# In[58]:


#take a 'Cholesky' decomposition:

# The Gershgorin theorem applied to rho implies that the eigenvalues lie within
# the union of D1(1, r = 0.8), D2(1, r = 0.9), and D3(1, r = 1.1). 
# There are three eigenvalues in D1UD2UD3 D(1, r = 0.8), D(1, r = 0.9), and D(1, r = 1.1).

# Get the default upper triengle decomposition
L2 = la.cholesky(rho2)
print(L2)

# Get lower triangle decomposition
L2l = scipy.linalg.cholesky(rho2, lower=True)
print(L2l)

print('There are 3 eigen values in the union disks D1, D2 and D3')


# In[59]:


# Question 5(a) LVAR
# Define Variable based on LVAR formula
# LVAR = VAR + L2 = W[]


# a
w5 = 16000000
alpha = 0.99
za = 2.33
mu5 = 0.01
prho = 0.03
smu =  0.0035 # 35 bps = 35*0.01% = 0.35%
srho = 0.015 # 150 bps = 150*0.01% = 1.5%

LVAR1 = (w5*((za*prho)-mu5))
L1 = ((w5*0.5)*(smu+(za*srho)))
LVAR = LVAR1 + L1
LVAR1p = (LVAR1/LVAR)*100
L1p = (L1/LVAR)*100


#create a dictionary for Liquidity Adjusted VARS
LVARS1 = {'VAR': LVAR1,
            'Liquidity Adjustment': L1,
               'Percentage of VAR in LVAR': LVAR1p,
            'Percentage of Liquidity Adjustment in LVAR': L1p,
               'Liquidity Adjustment Value at Risk': LVAR}

print(LVARS1)


# In[60]:


# Question 5 (b) LVAR
# Define Variable based on LVAR formula
# LVAR = VAR + L2 = W[]


# b
w5b = 40000000
alphab = 0.95
zab = 1.645
prhob = 0.03
srhob = 0.0055 # 150 bps = 55*0.01% = 0.55%

LVARb = (w5b*((zab*prhob)))
L2 = ((w5b*0.5)*(srhob))
LVAR2 = LVARb + L2
LVAR2p = 100*(LVARb/LVAR2)
L2p = 100*(L2/LVAR2)

#create a dictionary for Liquidity Adjusted VARS and ES for all portfolios
LVARS2 = {'VAR': LVARb,
            'Liquidity Adjustment': L2,
               'Percentage of VAR in LVAR': LVAR2p,
            'Percentage of Liquidity Adjustment in LVAR': L2p,
               'Liquidity Adjustment Value at Risk': LVAR2}

print(LVARS2)


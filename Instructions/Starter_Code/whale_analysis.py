#!/usr/bin/env python
# coding: utf-8

#  #  A Whale off the Port(folio)
#  ---
# 
#  In this assignment, you'll get to use what you've learned this week to evaluate the performance among various algorithmic, hedge, and mutual fund portfolios and compare them against the S&P 500 Index.

# In[1]:


# Initial imports
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Cleaning
# 
# In this section, you will need to read the CSV files into DataFrames and perform any necessary data cleaning steps. After cleaning, combine all DataFrames into a single DataFrame.
# 
# Files:
# 
# * `whale_returns.csv`: Contains returns of some famous "whale" investors' portfolios.
# 
# * `algo_returns.csv`: Contains returns from the in-house trading algorithms from Harold's company.
# 
# * `sp500_history.csv`: Contains historical closing prices of the S&P 500 Index.

# ## Whale Returns
# 
# Read the Whale Portfolio daily returns and clean the data

# In[2]:


# Reading whale returns
whale_csv_path = Path("Resources/whale_returns.csv")
whale_df  = pd.read_csv(whale_csv_path, index_col="Date", parse_dates=True,infer_datetime_format=True)
whale_df.head()


# In[3]:


get_ipython().run_line_magic('ls', '')


# In[4]:


# Count nulls
whale_df.isnull().sum()
whale_df.head()


# In[5]:


# Drop nulls and check again
whale_df.dropna(inplace=True)
whale_df.isnull().sum()


# ## Algorithmic Daily Returns
# 
# Read the algorithmic daily returns and clean the data

# In[6]:


# Reading algorithmic returns
algo_csv_path = Path("Resources/algo_returns.csv")
algo_df  = pd.read_csv(algo_csv_path, index_col="Date", parse_dates=True,infer_datetime_format=True)
algo_df.head()

# drop nulls
algo_df.dropna(inplace=True)
algo_df.isnull().sum()


# ## S&P 500 Returns
# 
# Read the S&P 500 historic closing prices and create a new daily returns DataFrame from the data. 

# In[7]:


# Reading S&P 500 Closing Prices
sp500_csv_path = Path("Resources/sp500_history.csv")
sp500_df  = pd.read_csv(sp500_csv_path, index_col="Date", parse_dates=True,infer_datetime_format=True)
sp500_df.head()

#drop nulls
sp500_df.dropna(inplace=True)
sp500_df.isnull().sum()



# In[8]:


# Check Data Types using two methods
sp500_df.dtypes
sp500_df.info()


# In[9]:


# Fix Data Types-  remove dollar signs and non-numeric characters; keeping date format as it is as widely used
sp500_df['Close'] = sp500_df['Close'].str.replace('$','')
sp500_df['Close'] = sp500_df['Close'].str.replace(',','')
sp500_df['Close'] = sp500_df['Close'].str.replace(' ','')

#convert close to a numeric data type
sp500_df['Close'] = pd.to_numeric(sp500_df['Close'])


# In[10]:


# Calculate Daily Returns
sp500_df['Returns'] = sp500_df['Close'].pct_change()


# In[11]:


# Drop nulls
sp500_df.dropna(inplace=True)
sp500_df['Returns'].head()


# In[12]:


# Rename `Close` Column to be specific to this portfolio.
sp500_df = sp500_df.rename(columns={'Daily Returns': 'Closing Price'})
sp500_df.head()


# ## Combine Whale, Algorithmic, and S&P 500 Returns

# In[13]:


# Join Whale Returns, Algorithmic Returns, and the S&P 500 Returns into a single DataFrame with columns for each portfolio's returns.

port_returns_df = pd.concat([whale_df, algo_df, sp500_df[['Returns']]], axis=1)

#rename columns
port_returns_df.rename(columns={'Returns':'Whale Returns', 'Returns': 'Algorithmic Returns', 'Returns': 'S&P500 Returns'}, inplace=True)

port_returns_df.head()



# In[14]:


#portfolio nulls to be dropped 

port_returns_df.isnull().sum()
port_returns_df.dropna(inplace=True)

port_returns_df.head()


# # Conduct Quantitative Analysis
# 
# In this section, you will calculate and visualize performance and risk metrics for the portfolios.

# port_returns_df.head()---

# ## Performance Anlysis
# 
# #### Calculate and Plot the daily returns.

# In[15]:


# Plot daily returns of all portfolios
import matplotlib.pyplot as plt

port_returns_df.plot(subplots=True, title="Daily Returns of Portfolios")
plt.show()


# #### Calculate and Plot cumulative returns.

# In[16]:


# Calculate cumulative returns of all portfolios
port_cummulative_returns_df = (1 + port_returns_df).cumprod() -1

# Plot cumulative returns
port_cummulative_returns_df.plot(subplots=True, title="Cummulative Returns of Portfolios")
plt.show()


# ---

# ## Risk Analysis
# 
# Determine the _risk_ of each portfolio:
# 
# 1. Create a box plot for each portfolio. 
# 2. Calculate the standard deviation for all portfolios
# 4. Determine which portfolios are riskier than the S&P 500
# 5. Calculate the Annualized Standard Deviation

# ### Create a box plot for each portfolio
# 

# In[17]:


# Box plot to visually show risk
plt.figure(figsize=(10,5)) # this is set to show good width of plot
port_returns_df.boxplot()
# expand the box to give more space and clarity
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.tight_layout()
#add title to boxplot
plt.title("Boxplot of Portfolio Returns")



# cleaner view by letting text on x-axis spacing and wrap around
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.subplots_adjust(bottom=0.3)

#show the plot
plt.show()


# ### Calculate Standard Deviations

# In[18]:


# Calculate the daily standard deviations of all portfolios
port_std_df = port_returns_df.std()


port_std_df.plot(subplots=True, title="Daily Standard Deviations")

# expand the box to give more space and clarity
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.tight_layout()
plt.figure(figsize=(10,5)) # this is set to show good width of plot

plt.show()
port_std_df.head()


# ### Determine which portfolios are riskier than the S&P 500

# In[19]:


# Calculate  the daily standard deviation of S&P 500
sp500_std_dev = sp500_df['Returns'].std()
print(f"The daily standard deviation of S&P 500 is {sp500_std_dev:.6f}")

# Determine which portfolios are riskier than the S&P 500
# calculate standard deviation for each item in the portfolio and use if else to find which is riskier than SP500

port_std_df.apply(lambda x: 'more risky' if x > sp500_std_dev else 'less risky')


# ### Calculate the Annualized Standard Deviation

# In[20]:


# Calculate the annualized standard deviation (252 trading days)
port_std_df = port_returns_df.std()*np.sqrt(252)



port_std_df.plot(subplots=True, title="Annanualized Standard Deviations")

# expand the box to give more space and clarity
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.tight_layout()
plt.figure(figsize=(10,5)) # this is set to show good width of plot

plt.show()
port_std_df.head()


# ---

# ## Rolling Statistics
# 
# Risk changes over time. Analyze the rolling statistics for Risk and Beta. 
# 
# 1. Calculate and plot the rolling standard deviation for all portfolios using a 21-day window
# 2. Calculate the correlation between each stock to determine which portfolios may mimick the S&P 500
# 3. Choose one portfolio, then calculate and plot the 60-day rolling beta between it and the S&P 500

# ### Calculate and plot rolling `std` for all portfolios with 21-day window

# In[21]:


# Calculate the rolling standard deviation for all portfolios using a 21-day window
port_std_df = port_returns_df.rolling(window=21).std()
port_std_df.head()




# Plot the rolling standard deviation

port_std_df.plot(subplots=True, title="Rolling Standard Deviations")

# expand the box to give more space and clarity
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.tight_layout()
plt.figure(figsize=(10,5)) # this is set to show good width of plot

plt.show()


# In[ ]:


### Calculate and plot the correlation


# In[22]:


# Calculate the correlation
correlation_matrix = port_returns_df.corr()

# Display de correlation matrix
import seaborn as sns
sns.heatmap(correlation_matrix, annot=True)


# ### Calculate and Plot Beta for a chosen portfolio and the S&P 500

# In[36]:


# Calculate covariance of a single portfolio - 'Algo 1'

# Create a rolling window of the required size
window = 60

# Calculate the rolling covariance
rolling_cov = port_returns_df['Algo 1'].rolling(window).cov(port_returns_df['S&P500 Returns'])

# Calculate rolling variance of S&P 500
rolling_var = port_returns_df['S&P500 Returns'].rolling(window).var()


# Calculate the rolling beta
rolling_beta = rolling_cov / rolling_var

# Plot the rolling beta
rolling_beta.plot()
plt.xlabel('Time')
plt.ylabel('Beta')
plt.show()


# ## Rolling Statistics Challenge: Exponentially Weighted Average 
# 
# An alternative way to calculate a rolling window is to take the exponentially weighted moving average. This is like a moving window average, but it assigns greater importance to more recent observations. Try calculating the [`ewm`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html) with a 21-day half life for each portfolio, using standard deviation (`std`) as the metric of interest.

# In[48]:


# Use `ewm` to calculate the rolling window

# Set the half-life for the EWMA
half_life = 21

# Calculate the EWMA of the std for Algo 1
ewma_std_Algo1 = port_returns_df['Algo 1'].ewm(halflife=half_life).std()

# Calculate the EWMA of the std for S&P500 Returns
ewma_std_sp500 = port_returns_df['S&P500 Returns'].ewm(halflife=half_life).std()

# Plot the EWMA of the std for Algo 1 and S&P500 Returns
ewma_std_Algo1.plot(label='Algo 1')
ewma_std_sp500.plot(label='S&P500 Returns')
plt.xlabel('Time')
plt.ylabel('EWMA of Std Algo 1 and S&P500')

# Move the legend to the right of the plot
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()


# Plot the EWMA of the std for each portfolio


# Calculate the EWMA of the std for all items in portfolio
ewma_std_All = port_returns_df.ewm(halflife=half_life).std()

# Plot the EWMA of the std for each portfolio

# Set the size of the plot
plt.figure(figsize=(105,25))

ewma_std_All.plot()
plt.xlabel('Time')
plt.ylabel('EWMA of Std')
# Move the legend to the right of the plot
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()



# ---

# # Sharpe Ratios
# In reality, investment managers and thier institutional investors look at the ratio of return-to-risk, and not just returns alone. After all, if you could invest in one of two portfolios, and each offered the same 10% return, yet one offered lower risk, you'd take that one, right?
# 
# ### Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot

# In[26]:


# Annualized Sharpe Ratios


# In[26]:


# Annualized Sharpe Ratios


# In[26]:


# Annualized Sharpe Ratios


# In[27]:


# Visualize the sharpe ratios as a bar plot


# ### Determine whether the algorithmic strategies outperform both the market (S&P 500) and the whales portfolios.
# 
# Write your answer here!

# ---

# # Create Custom Portfolio
# 
# In this section, you will build your own portfolio of stocks, calculate the returns, and compare the results to the Whale Portfolios and the S&P 500. 
# 
# 1. Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
# 2. Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock
# 3. Join your portfolio returns to the DataFrame that contains all of the portfolio returns
# 4. Re-run the performance and risk analysis with your portfolio to see how it compares to the others
# 5. Include correlation analysis to determine which stocks (if any) are correlated

# ## Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.

# In[28]:


# Reading data from 1st stock


# In[29]:


# Reading data from 2nd stock


# In[30]:


# Reading data from 3rd stock


# In[31]:


# Combine all stocks in a single DataFrame


# In[32]:


# Reset Date index


# In[33]:


# Reorganize portfolio data by having a column per symbol


# In[34]:


# Calculate daily returns

# Drop NAs

# Display sample data


# ## Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock

# In[35]:


# Set weights
weights = [1/3, 1/3, 1/3]

# Calculate portfolio return

# Display sample data


# ## Join your portfolio returns to the DataFrame that contains all of the portfolio returns

# In[36]:


# Join your returns DataFrame to the original returns DataFrame


# In[37]:


# Only compare dates where return data exists for all the stocks (drop NaNs)


# ## Re-run the risk analysis with your portfolio to see how it compares to the others

# ### Calculate the Annualized Standard Deviation

# In[38]:


# Calculate the annualized `std`


# ### Calculate and plot rolling `std` with 21-day window

# In[39]:


# Calculate rolling standard deviation

# Plot rolling standard deviation


# ### Calculate and plot the correlation

# In[40]:


# Calculate and plot the correlation


# ### Calculate and Plot Rolling 60-day Beta for Your Portfolio compared to the S&P 500

# In[41]:


# Calculate and plot Beta


# ### Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot

# In[42]:


# Calculate Annualized Sharpe Ratios


# In[43]:


# Visualize the sharpe ratios as a bar plot


# ### How does your portfolio do?
# 
# Write your answer here!

# In[ ]:





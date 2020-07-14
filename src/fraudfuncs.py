import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eda(data, country):
    """Wrapper function to calculate several exploratory analyses at once for the country provided"""
    sub_data = data[data['input country code'] == country]
    print(f"Number of {country.upper()} transactions: {len(sub_data)},\nProportion of total transactions: {round(len(sub_data)/len(data), 2)}")
    
    total_fraud_rate = (len(data[data['isfraud'] == 1])/len(data))*100
    sub_fraud_rate = (len(sub_data[sub_data['isfraud'] == 1])/len(sub_data))*100
    print(f"{country.upper()} fraud rate: {round(sub_fraud_rate, 3)}%, {round(((sub_fraud_rate - total_fraud_rate)/total_fraud_rate)*100, 2)}% different than average")
    
    for feature in ['Email First Seen Days', 'IP Distance From Address', 'Transaction Risk Score']:
        plot_continuous_histograms(data, sub_data, country, feature)
    for feature in ['Primary Phone to Name', 'Email to Name']:
        plot_categorical(data, sub_data, country, feature)
    
    correlations(sub_data)    

def plot_continuous_histograms(data, sub_data, country, feature):
        """Plots a histogram of a continuous feature for both the entire data set and the selected country"""
        fig, ax = plt.subplots(1, 2, figsize = (14, 5))
        data[feature].hist(grid = False, bins = 20, ax = ax[0])
        sub_data[feature].hist(grid = False, bins = 20, ax = ax[1])
        ax[0].set_title(f'Histogram of Total Data for {feature}')
        ax[1].set_title(f'Histogram of {country.upper()} Data for {feature}')
        ax[0].set_xlabel(feature)
        ax[1].set_xlabel(feature)
        ax[0].set_ylabel('Number of Observations')

def plot_categorical(data, sub_data, country, feature):
        """Plots a bar chart of the number of each category in a categorical feature for the entire data and the selected country"""
        fig, ax = plt.subplots(1, 2, figsize = (14, 5))
        sns.countplot(data[feature], ax = ax[0], order = ['No name found', 'Match', 'No match'])
        sns.countplot(sub_data[feature], ax = ax[1], order = ['No name found', 'Match', 'No match'])
        ax[0].set_title(f'Number of {feature} responses for All Data')
        ax[0].set_ylabel('Number of responses')
        ax[1].set_title(f'Number of {feature} responses for {country.upper()} Data')
        ax[1].set_ylabel('Number of responses')
    
def correlations(sub_data):
    """Calculates Pearson correlations with fraud for each feature. One-hot encodes categorical features with pd.get_dummies() -- may not actually be appropriate for calculating correlations"""
    sub_corrs1 = sub_data.corr()['isfraud']
    sub_corrs2 = pd.get_dummies(sub_data[['Primary Phone to Name', 'Email to Name']], dummy_na = True).corrwith(sub_data['isfraud'])
    sub_corrs = pd.concat([sub_corrs1, sub_corrs2], axis = 0)
    print("Feature correlations:")
    print(sub_corrs.sort_values(ascending = False)[1:])

def split_and_plot(data, country):
    """Wrapper function for splitting data into fraudulent and nonfraudulent transactions and plotting the percentages for each feature"""
    sub_data = data[data['input country code'] == country]
    
    #Split continuous data and plot
    for feature in ['Email First Seen Days', 'IP Distance From Address', 'Transaction Risk Score']:
        plot_continuous_splitdata(split_continuous(sub_data, feature, 20), feature, country)
    
    #Split categorical data and plot
    for feature in ['Primary Phone to Name', 'Email to Name']:
        plot_categorical_splitdata(split_categorical(sub_data, feature), feature, country)

def continuous_plotting_label(x):
    """Function to make x-axis labels for continous feature plots"""
    if type(x) == str:
        return x
    else:
        return f"{int(round(x.left))}-{int(round(x.right))}"

def split_continuous(data, feature, bins):
    """Function to split continuous data into fraudulent and nonfraudulent transactions and return a binned dataframe with counts and percentages in each bin."""
    #Add bins for continous features
    data['feature_bin'] = pd.cut(data[feature], bins = bins)
    #Add a bin for missing data specifically
    data['feature_bin'].cat.add_categories(['Missing'], inplace = True)
    #NaN = 'Missing'
    data['feature_bin'].fillna('Missing', inplace = True)
    data[feature].fillna('Missing', inplace = True)
    #Get number of observations in each bin
    count_notfraud = data[data['isfraud'] == 0].groupby('feature_bin')[feature].count()
    #Calculate the percent of observations in each bin  
    percent_notfraud = []
    for num in count_notfraud:
        percent_notfraud.append((num/sum(count_notfraud))*100)
    #Make new dataframe of counts and percentages
    binned_notfraud = pd.concat([pd.DataFrame(count_notfraud).reset_index(), 
                                 pd.DataFrame(percent_notfraud, columns = ['Percent'])], axis = 1)
    #Add column for fraud label
    binned_notfraud['isfraud'] = ['Not Fraud']*len(binned_notfraud)
    #Add column for the max of each bin interval
    binned_notfraud['binmax'] = binned_notfraud['feature_bin'].apply(continuous_plotting_label) #(lambda x: int(round(x.right)))
    
    #Repeat process for fraudulent data
    count_fraud = data[data['isfraud'] == 1].groupby('feature_bin')[feature].count()
    percent_fraud = []
    for num in count_fraud:
        percent_fraud.append((num/sum(count_fraud))*100)
    binned_fraud = pd.concat([pd.DataFrame(count_fraud).reset_index(), 
                              pd.DataFrame(percent_fraud, columns = ['Percent'])], axis = 1)
    binned_fraud['isfraud'] = ['Fraud']*len(binned_fraud)
    binned_fraud['binmax'] = binned_fraud['feature_bin'].apply(continuous_plotting_label) #(lambda x: int(round(x.right)))
    #Merge two dataframes
    df = pd.concat([binned_notfraud, binned_fraud], axis = 0)
    
    return df

def split_categorical(data, feature):
    """Function to split categorical data into fraudulent and nonfraudulent transactions and return a dataframe with counts/percentages for each category."""
    #Replace missing data with a missing label
    data[feature].fillna('Missing', inplace = True)
    #Get number of observations in each category
    count_notfraud = data[data['isfraud'] == 0].groupby(feature)[feature].count()
    #Calculate the percent of observations in each category  
    percent_notfraud = []
    for num in count_notfraud:
        percent_notfraud.append((num/sum(count_notfraud))*100)
    #Make new dataframe of counts and percentages
    binned_notfraud = pd.concat([pd.DataFrame(count_notfraud).rename({f'{feature}': 'counts'}, axis = 1).reset_index(), 
                                 pd.DataFrame(percent_notfraud, columns = ['Percent'])], axis = 1)
    #Add column for fraud label
    binned_notfraud['isfraud'] = ['Not Fraud']*len(binned_notfraud)
    
    #Repeat process for fraudulent data
    count_fraud = data[data['isfraud'] == 1].groupby(feature)[feature].count()
    percent_fraud = []
    for num in count_fraud:
        percent_fraud.append((num/sum(count_fraud))*100)
    binned_fraud = pd.concat([pd.DataFrame(count_fraud).rename({f'{feature}': 'counts'}, axis = 1).reset_index(), 
                              pd.DataFrame(percent_fraud, columns = ['Percent'])], axis = 1)
    binned_fraud['isfraud'] = ['Fraud']*len(binned_fraud)
    #Merge two dataframes
    df = pd.concat([binned_notfraud, binned_fraud], axis = 0)
    
    return df

def plot_continuous_splitdata(split_data, feature, country):
    """Plot continuous data that has been split into fraudulent/nonfraudulent."""
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    sns.barplot(data = split_data, x = 'binmax', y = 'Percent', hue = 'isfraud', ax = ax);
    plt.xticks(rotation=90)
    ax.set_xlabel(f'{feature}')
    ax.set_title(f'Percent of fraudulent/nonfraudulent {feature} in {country.upper()}')
    ax.legend();

def plot_categorical_splitdata(split_data, feature, country):
    """Plot categorical data that has been split into fraudlent/nonfraudlent."""
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    sns.barplot(data = split_data, x = feature, y = 'Percent', hue = 'isfraud', ax = ax);
    ax.set_xlabel(f'{feature}')
    ax.set_title(f'Percent of fraudulent/nonfraudulent {feature} in {country.upper()}')
    ax.legend();
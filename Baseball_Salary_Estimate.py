##########################################
# Salary Estimation with Machine Learning
###########################################

######################
# Business Problem
######################

# Develop a machine learning model to estimate the salaries of baseball players
# whose salary information and career statistics for 1986 are shared.

#########################
# About Dataset
#########################

# Dataset 1988 ASA Graphics Section. It is part of the data used in the Poster Session.
# Salary data originally taken from Sports Illustrated, April 20, 1987.
# 1986 and career statistics, 1987 Baseball Encyclopedia published by Collier Books,
# Macmillan Publishing Company, New York obtained from the update.

# 20 Features 322 Observations 21 KB

# AtBat: Number of hits with a baseball bat in the 1986-1987 season
# Hits: Number of hits in the 1986-1987 season
# HmRun: Most valuable hits in the 1986-1987 season
# Runs: The points (s)he earned for his team in the 1986-1987 season
# RBI: Number of players a hitter had jogged when (s)he hit
# Walks: Number of mistakes made by the opposing player
# Years: Player's playing time in major league (years)
# CAtBat: Number of hits during the player's career
# CHits: Number of hits made by the player throughout his/her career
# CHmRun: The player's most valuable point during his/her career
# CRuns: The number of points the player has earned for his/her team during his career
# CRBI: Number of players the player has run during his/her career
# CWalks: Number of mistakes made by the opposing player during the player's career
# League: A factor with A and N levels showing the league in which the player played until the end of the season
# Division: A factor with E and W levels indicating the position played by the player at the end of 1986
# PutOuts: Helping your teammate during the game
# Assits: The number of assists the player made in the 1986-1987 season
# Errors: The number of errors of the player in the 1986-1987 season
# Salary: Salary of the player in 1986-1987 season (over thousand)
# NewLeague: A factor with A and N levels showing the player's league at the start of the 1987 season

##############################################
# Salary Prediction with Multiple Linear Regression
##############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

################################################
# 1. Exploratory Data Analysis
################################################

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/hitters.csv")

# Overview;


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


# Classification of variable types;


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of 3 lists with return is equal to the total number of variables:
        cat_cols + num_cols + cat_but_car = number of variables

    """


    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]


    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Analysis of categorical variables;


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)


# Analysis of numerical variables;


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)


# Average of numerical variables relative to the dependent variable;


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Salary", col)


# Mean of categorical variables by dependent variable;


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)


# Correlation analysis;


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


correlation_matrix(df, num_cols)


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

# Outlier threshold for variables;


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


#  Outliers check;


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in df.columns:
    print(col, check_outlier(df, num_cols))

# No outliers in defined ranges for numeric variables.


# Missing value and ratio analysis;


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(df, True)

#         n_miss  ratio
# Salary      59  18.32

msno.bar(df)

# Filling missing value with KNN method;

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["Salary"] = dff[["Salary"]]
df.head()

df.isnull().sum()

# Label Encoding;

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    label_encoder(df, col)

df.head()


# Standardization for numeric variables;

scaler = RobustScaler()
num_cols = [col for col in num_cols if 'Salary' not in col]
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()

################################################
# 3. Model
################################################

X = df.drop('Salary', axis=1)

y = df[["Salary"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
reg_model = LinearRegression().fit(X_train, y_train)
reg_model.intercept_
reg_model.coef_

# Train RMSE

y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 272.8647419577628

# Train RKARE

reg_model.score(X_train, y_train)
# 0.5432676471389162

# Test RMSE

y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 343.5094128218927

# Test RKARE

reg_model.score(X_test, y_test)
# 0.5288914529288602


# Cross Validation RMSE

np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
# 306.2601509645383

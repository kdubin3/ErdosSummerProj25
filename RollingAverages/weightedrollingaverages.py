# Packages

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from datetime import datetime, timedelta
from seaborn import set_style
from sklearn.metrics import mean_squared_error

# Choose line here
# Options: '21', '22', '34', '47', '54', '60', '62', '63', '77', '79', '97', '94', '146', '152'

line = '152'

# Import data
df = pd.read_csv('../Data/CTA_Average_Bus_Ridership_1999_2024_cleaned.csv')

# Sort by date
df['DATE'] = pd.to_datetime(df['DATE'])
df.sort_values('DATE', inplace=True)

# Consider ridership averages in thousands
df = df.rename(columns={'Sunday - Holiday': 'SundayHoliday'})
df['Weekday'] = df['Weekday'].div(1000)
df['Saturday'] = df['Saturday'].div(1000)
df['SundayHoliday'] = df['SundayHoliday'].div(1000)

"""
This function takes in a single bus route as a string, retricts the bus
dataframe to this one route. If only the bus route is given, the function
will plot the Weekday, Saturday, and Sunday-Holiday ridership averages
and return the restricted dataframe. If a second False argument is given,
the function will just return the restricted dataframe.
"""

def plot_bus_routes(bus_route, plot=True, save = True):
    # Restrict dataframe to bus route
    bus_route_df = df[df['BUS'] == bus_route]

    # Plot!
    if plot:
        # Create plot
        plt.figure(figsize=(20,8))

        # Plot all day types
        plt.plot(bus_route_df.DATE, bus_route_df.Weekday, 'r', label='Weekday Average')
        plt.plot(bus_route_df.DATE, bus_route_df.Saturday, 'g', label='Saturday Average')
        plt.plot(bus_route_df.DATE, bus_route_df.SundayHoliday, 'b', label='Sunday/Holiday Average')

        # Aesthetics
        plt.title(bus_route + " Bus Ridership", fontsize=24)
        plt.xlabel("Date", fontsize=18)
        plt.ylabel("Ridership on Day Type by Month in Thousands", fontsize=18)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(fontsize=14, loc=3)

        # Save!
        if save:
            plt.savefig(bus_route+"bus.png")

        # Show!
        plt.show()

    # Return dataframe
    return bus_route_df

df_line = plot_bus_routes(line, plot=False, save=False)

# Make traing and test sets
# Only start training 4 months after start of COVID impact

df_line_train = df_line.drop(df_line.tail(12).index, inplace=False)[-40:]
df_line_test = df_line.tail(12)

# Create the Rolling Averages for each day type

# New test/train sets
df_line_train_train = df_line_train.drop(df_line_train.tail(12).index)
df_line_train_test = df_line_train.tail(12)

# Assign weights for weighted averages

weights = np.array([1.0/6, 1.0/6, 2.0/3])
bus_train_Weekday_fit =  df_line_train.Weekday.rolling(3, closed='left').apply(lambda x: np.dot(weights, x), raw=True)
bus_train_Saturday_fit =  df_line_train.Saturday.rolling(3, closed='left').apply(lambda x: np.dot(weights, x), raw=True)
bus_train_SundayHoliday_fit =  df_line_train.SundayHoliday.rolling(3, closed='left').apply(lambda x: np.dot(weights, x), raw=True)

# Make plot
plt.figure(figsize=(20,8))

# Weekday
plt.plot(df_line_train.DATE, df_line_train.Weekday, 'r', label="Weekday Average")
plt.plot(df_line_train.DATE, bus_train_Weekday_fit ,'r--', label="Weekday Weighted Rolling Average")
plt.plot(df_line_test.DATE, bus_train_Weekday_fit.iloc[-1]*np.ones(12) ,'r--o', label="Weekday Weighted Rolling Average Prediction")
plt.plot(df_line_test.DATE, df_line_test.Weekday, 'r')



# Saturday
plt.plot(df_line_train.DATE, df_line_train.Saturday, 'g', label="Saturday Average")
plt.plot(df_line_train.DATE, bus_train_Saturday_fit ,'g--', label="Saturday Weighted Rolling Average")
plt.plot(df_line_test.DATE, bus_train_Saturday_fit.iloc[-1]*np.ones(12) ,'g--o', label="Saturday Weighted Rolling Average Prediction")
plt.plot(df_line_test.DATE, df_line_test.Saturday, 'g')

# Sunday - Holiday

plt.plot(df_line_train.DATE, df_line_train.SundayHoliday, 'b', label="Sunday/Holiday Average")
plt.plot(df_line_train.DATE, bus_train_SundayHoliday_fit ,'b--', label="Sunday/Holiday Weighted Rolling Average")
plt.plot(df_line_test.DATE, bus_train_SundayHoliday_fit.iloc[-1]*np.ones(12) ,'b--o', label="Sunday/Holiday Weighted Rolling Average Prediction")
plt.plot(df_line_test.DATE, df_line_test.SundayHoliday, 'b')
# plt.plot(df_line_train.DATE, df_line_train.SundayHoliday, 'b', label="Sunday/Holiday Average")
# plt.plot(df_line_train.DATE[-16:],  df_line_train_train.SundayHoliday.rolling(window=3, closed = 'left').mean()[-16:], 'b--', label="Sunday/Holiday Model")
# plt.plot(df_line_test.DATE, df_line_test.SundayHoliday, 'b')
# plt.plot(df_line_test.DATE, df_line_test.SundayHoliday.mean()*np.ones(12), 'b--o')

# Aesthetics
plt.title("The "+line+" Bus: Weighted Rolling Averages Ridership Model", fontsize=24)
plt.xlabel("Date", fontsize=18)
plt.ylabel("Ridership on Day Type by Month in Thousands", fontsize=18)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14, loc=2)

# Save!
plt.savefig("wra_"+line+".png")

# Show!
plt.show()
#!/usr/bin/env python
# coding: utf-8

# In[27]:


###########################################################
# course: cmps3500
# CLASS Project
# PYTHON IMPLEMENTATION: BASIC NEURAL NETWORK
# date: 05/08/2025
# Student 1: Sophia Stewart
# Student 2: Dason Baird
# Student 3: Andrew Little
# Description: A basic neural network that trains
# and tests on LA crime data to predict arrest or no arrest
###########################################################


# In[28]:


# Here several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy import stats
import time


# In[29]:


#importing libraries
import numpy as np
import regex
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from tabulate import tabulate


#tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout


# Keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense


# In[30]:


#reading the CSV file into dataframe df
# Data should be located in the same folder as the notebook for this to work
df = pd.read_csv('LA_Crime_Data_2023_to_Present_data.csv') 


# In[31]:


vict_desc = ['unknown', 'Other Asian', 'Black', 'Chinese',
    'Cambodian', 'Filipino', 'Gaumanian', 'Hispanic/Latin/Mexican',
    'American Indian/Alaskan Native', 'Japanese', 'Korean', 'Laotian', 'Other',
    'Pacific Islander', 'Samoan', 'Hawaiian', 'Vietnamese', 'White', 'Asain Indian']

vict_sex = ['Male', 'Female', 'Intersex', 'Unknown']

# Map Target Column
# Mapping dictionary
# Add mapping to time of day
mapping = {
            'IC': 'No Arrest'
            ,'AA': 'Arrest'
            ,'AO': 'No Arrest'
            ,'JO': 'No Arrest'
            ,'JA': 'Arrest'
            ,'CC': 'No Arrest'
}

df['Target'] = df['Status'].map(mapping)


# In[33]:


# Function to evaluate predicted vs test data categorical variables
def plot_prediction_vs_test_categorical(y_test, y_pred, class_labels):
    # Plots the prediction vs test data for categorical variables.

    # Args:
    #     y_test (array-like): True labels of the test data.
    #     y_pred (array-like): Predicted labels of the test data.
    #     class_labels (list): List of class labels.

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Calculates performance of multivariate classification model
def calculate_performance_multiclass(y_true, y_pred):
    # Calculates various performance metrics for multiclass classification.

    # Args:
    #     y_true: The true labels.
    #     y_pred: The predicted labels.

    # Returns:
    #     A dictionary containing the calculated metrics.

    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Precision, Recall, and F1-score (macro-averaged)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')

    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics


# In[34]:


def plot_numerical_column(data_frame, Column_Name, n_bins):
# Plot histogram and box plot for numerical columns
# num_col: Numerical Column i.e. df['Age']
# n_bins : number of bins

   # Get data set
   data = df[Column_Name]
    
   # Plotting histogram
   ####################
   # Plotting a basic histogram
   plt.hist(data, bins=n_bins, color='skyblue', edgecolor='black')
    
   # Adding labels and title
   plt.xlabel(Column_Name)
   plt.ylabel('Frequency')
   plt.title('Basic Histogram')

   plt.show()
 
   # Plotting box plot
   ####################
   plt.boxplot(data)
   plt.ylabel(Column_Name)
   plt.ylim(data.min(), data.max()) 
   plt.title('Basic Boxplot')
   plt.show()

   # Display the plot
   plt.show()

def plot_no_numerical_column(data_frame, Column_Name):
# Plot counts for no numerical cases

   # Get data set
   data = data_frame[Column_Name]

   occupation_count = data.value_counts()
   sns.barplot(x = occupation_count.values, y = occupation_count.index, orient = 'h')
   plt.xlabel('Count')
   plt.ylabel(Column_Name)

   # Display the plot
   plt.show()


# In[35]:


# Check variable using plots:
plot_numerical_column(df, 'Vict Age', 20)


# In[36]:


# Check variable using plots:
plot_no_numerical_column(df, 'Status')


# In[37]:


def spearmenGraph(data_frame, columns, threshold):
    df_corr = df[l_cols_numerical].corr(method='spearman')
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df_corr,
            mask=np.triu(np.ones_like(df_corr, dtype=bool)), 
            cmap=sns.diverging_palette(230, 20, as_cmap=True), 
            vmin=-1.0, vmax=1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

    # Identify correlated feature pairs with their correlation values
    #high_corr_pairs = []
    #for col in df_corr.columns:
    #    for row in df_corr.index:
    #        if abs(df_corr.loc[row, col]) > threshold and row != col:
    #            high_corr_pairs.append((row, col, df_corr.loc[row, col]))

    # Sort by absolute value, descending
    #high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)  
    #print("\nCorrelated pairs with correlation values:")
    #for pair in high_corr_pairs:
    #    print(f"{pair[0]} and {pair[1]} have correlation: {pair[2]:.2f}")

l_cols_numerical = ['Vict Age', 'Crm Cd', 'AREA', 'Part 1-2', 'Premis Cd', 'Weapon Used Cd','TIME OCC', 'Rpt Dist No'] 
spearmenGraph(df, l_cols_numerical, 0.0)


# In[38]:


#axs = pd.plotting.scatter_matrix(df[l_cols_numerical], figsize=(10,10), marker = 'o', hist_kwds = {'bins': 10}, s = 60, alpha = 0.2)

#def wrap(txt, width=8):
#    '''helper function to wrap text for long labels'''
#    import textwrap
#    return '\n'.join(textwrap.wrap(txt, width))

#for ax in axs[:,0]: # the left boundary
#    ax.set_ylabel(wrap(ax.get_ylabel()), size = 8)
#    ax.set_xlim([None, None])
#    ax.set_ylim([None, None])

#for ax in axs[-1,:]: # the lower boundary
#    ax.set_xlabel(wrap(ax.get_xlabel()), size = 8)
#    ax.set_xlim([None, None])
#    ax.set_ylim([None, None])


# In[39]:


df[l_cols_numerical].boxplot(rot=90)


# In[40]:


corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[41]:


################################################################
# 1. Data Loading & Cleaning
################################################################
def cleanFile(file_path, check):
    # Error handling in case of issues with file
    try:
        if check == 0:
            ndf = pd.read_csv('LA_Crime_Data_2023_to_Present_data.csv')
        else:
            ndf = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File does not exist")
    except pd.errors.EmptyDataError:
        print("Error: File is empty")
    except pd.errors.ParserError:
        print("Error: Failed to parse file. Check for invalid formatting")
    except Exception as e:
        print(f"Error: unexpected failure occured: {e}")
    else:
        print("File Read Successful")
    try:
        ndf['AREA NAME'] = ndf['AREA NAME'].astype('string')
        ndf['Crm Cd Desc'] = ndf['Crm Cd Desc'].astype('string')
        ndf['Mocodes'] = ndf['Mocodes'].astype('string')
        ndf['Vict Sex'] = ndf['Vict Sex'].astype('string')
        ndf['Vict Descent'] = ndf['Vict Descent'].astype('string')
        ndf['Premis Desc'] = ndf['Premis Desc'].astype('string')
        ndf['Weapon Desc'] = ndf['Weapon Desc'].astype('string')
        ndf['Status'] = ndf['Status'].astype('string')
        ndf['Status Desc'] = ndf['Status Desc'].astype('string')
        ndf['Target'] = ndf['Status'].map(mapping)
        ndf['Target'] = ndf['Target'].astype('string')
        ndf = ndf.drop(ndf[ndf['Status'] == 'IC'].index)
        ndf = ndf.drop(ndf[ndf['Status'] == 'CC'].index)
        status = 1
    except:
        print("No Status column given")
        status = 0
    try:
        ndf['Vict Sex'] = ndf['Vict Sex'].astype('string')
        ndf['Vict Descent'] = ndf['Vict Descent'].astype('string')
        
        ndf['Mocodes_freq'] = ndf['Mocodes'].map(ndf['Mocodes'].value_counts())
        
        # Print the size of dataset before cleaning
        #print(f"Before cleaning: {size}")
        ndf['Weapon Used Cd'] = ndf['Weapon Used Cd'].fillna(0)
        ndf['Mocodes_freq'] = ndf['Mocodes_freq'].fillna(0)
        ndf['Premis Cd'] = ndf['Premis Cd'].fillna(0)
        ndf['Vict Sex'] = ndf['Vict Sex'].fillna('X')
        ndf['Vict Descent'] = ndf['Vict Descent'].fillna('X')
        ndf['TIME OCC'] = pd.to_datetime(ndf['TIME OCC'], format='%H%M', 
                                         errors='coerce').dt.strftime('%I%M').astype(str)
        ndf['TIME OCC'] = ndf['TIME OCC'].astype('string')

        
        
        print("Checking for Duplicates to Remove...")
        print(f"Duplicates found: {ndf.duplicated().sum()}\n")
        ndf = ndf.drop_duplicates()
        
        
        # Date Column split dates into year, month, and day column
        ndf['DATE OCC'] = pd.to_datetime(ndf['DATE OCC'])
        ndf['Year'] = ndf['DATE OCC'].dt.year
        ndf['Month'] = ndf['DATE OCC'].dt.month
        ndf['Day'] = ndf['DATE OCC'].dt.day
    except SyntaxError as se:
        print("Syntax Error occued")
    except:
        print("Error with updating columns")


    # Drop columns we no longer need
    if status == 1:
        ndf = ndf.drop(columns=['Unnamed: 0', 'DR_NO', 'Date Rptd', 
                                'AREA NAME', 'Crm Cd Desc', 'Premis Desc', 'Weapon Desc', 
                                'Status', 'Status Desc','Mocodes'])
    else:
        ndf = ndf.drop(columns=['Unnamed: 0', 'DR_NO',  'Date Rptd',
                                'AREA NAME', 'Crm Cd Desc', 'Premis Desc', 'Weapon Desc', 'Mocodes'])

    

    # Drop Age if < 1 <- negative vals & zeros causes
    # innacurate outlier calculation and can be considered NaN
    ndf = ndf[ndf['Vict Age'] > 1]

    

    # Reorder so date and time stay together
    
    if status == 1:
        columns_order = ['Year', 'Month', 'Day', 'TIME OCC', 'AREA', 
                         'Rpt Dist No','Part 1-2', 'Crm Cd', 'Vict Age', 
                         'Vict Sex','Vict Descent', 'Mocodes_freq', 
                         'Premis Cd', 'Weapon Used Cd', 'Target']
    else:
        columns_order = ['Year', 'Month', 'Day', 'TIME OCC', 'AREA', 
                         'Rpt Dist No','Part 1-2', 'Crm Cd', 
                         'Vict Age', 'Vict Sex','Vict Descent', 
                         'Mocodes_freq', 'Premis Cd', 'Weapon Used Cd']
        
    ndf = ndf[columns_order]
    


    
    # Remove outliers by calling function
    ndf = remOutliers(ndf, 'Vict Age')

    
    
    # Print new size of clean dataset before returning it
    rows = ndf.shape[0]
    print(f"After cleaning:")
    print(f"Size: {ndf.size}")
    print(f"Shape: {ndf.shape}\n")    
    clean_time = time.localtime()
    clean_time = time.strftime("%Y-%m-%d %H:%M:%S", clean_time)
    print(f"[{clean_time}] Total Rows after cleaning is: {rows}")

    return ndf, status


# In[42]:


# =============================================================================
# Function: remOutliers
# Purpose: Removes outliers from the given DataFrame column using the IQR method.
# Parameters:
#   df  - Pandas DataFrame containing the dataset.
#   col - Column name (string) on which to perform outlier removal.
# Returns:
#   df  - Filtered DataFrame with outliers removed based on the IQR range.
# Notes:
#   - Uses absolute values for bounds.
#   - Terminates the program if the DataFrame is empty.
# =============================================================================
def remOutliers(df, col):
    # Check if data exists before calculating IQR
    if df.empty:
        print("DataFrame is empty")
        quit()
    else:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = abs(Q1 - 1.5 * IQR)
        upper_bound = abs(Q3 + 1.5 * IQR)

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df


# In[43]:


# Set pandas to display all data
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# In[44]:


# =============================================================================
# Function: loading_set
# Purpose: Loads a CSV file, identifies it as training or testing data,
#          and logs various metadata including number of rows/columns and load time.
# Parameters:
#   file_name - Name of the CSV file to be loaded.
# Returns:
#   df        - Pandas DataFrame if the file is successfully loaded.
#             - None if an error occurs while reading the file.
# Notes:
#   - Prints detailed status messages with timestamps for debugging/logging.
# =============================================================================
def loading_set(file_name):
    set_type = "None"
    if file_name == 'LA_Crime_Data_2023_to_Present_data.csv':
        set_type = "Training"
    elif file_name == 'LA_Crime_Data_2023_to_Present_test1.csv':
        set_type = "Testing"

    start_time = time.time()
    print(f"Loading {set_type} set:")
    print("*********************")
        
    ############################################################
    # Log the current script start time in a human-readable format
    ############################################################
    script_time = time.localtime()
    script_time = time.strftime("%Y-%m-%d %H:%M:%S", script_time)
    print(f"[{script_time}] Starting Script")
    ############################################################

    ############################################################
    # Read the csv file
    ############################################################
    try: 
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: File {file_name} does not exists.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: File is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: Could not parse file.")
        return None
    except Exception as e:
        print(f"Unexpected error reading file: {e}")
        return None

    data_time = time.localtime()
    data_time = time.strftime("%Y-%m-%d %H:%M:%S", data_time)
    print(f"[{data_time}] Loading {set_type} data set")
    #df = pd.read_csv(file_name)
    #data_time = time.localtime()
    #data_time = time.strftime("%Y-%m-%d %H:%M:%S", data_time)
    #print(f"[{data_time}] Loading {set_type} data set")
        
    ############################################################
    # Get the number of columns in the dataset
    ############################################################
    column_size = df.shape[1]
    column_size_time = time.localtime()
    column_size_time = time.strftime("%Y-%m-%d %H:%M:%S", column_size_time)
    print(f"[{column_size_time}] Total Columns Read: {column_size}")
        
    ############################################################
    # Get the number of rows in the dataset
    ############################################################
    row_size = df.shape[0]
    row_size_time = time.localtime()
    row_size_time = time.strftime("%Y-%m-%d %H:%M:%S", row_size_time)
    print(f"[{row_size_time}] Total Rows Read: {row_size}")

    ############################################################
    # Record the end time and calculate the total loading time
    ############################################################
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to load is: {elapsed_time:.2f}secs\n")

    return df


# In[45]:


# =============================================================================
# Function: trainNN
# Purpose: Prepares the dataset, encodes features/target, scales continuous features,
#          builds and trains a neural network to predict the target class.
# Parameters:
#   df - Cleaned Pandas DataFrame that includes both categorical and continuous features.
# Returns:
#   model           - Trained Keras Sequential model.
#   encoder         - OneHotEncoder instance used on categorical input features.
#   target_encoder  - OneHotEncoder used for encoding target labels.
#   scaler          - StandardScaler instance used to scale continuous features.
# Notes:
#   - Drops rows with missing target values before training.
#   - Includes early stopping to avoid overfitting.
#   - Includes exception handling for training errors and memory issues.
# =============================================================================
def trainNN(df):
    df = df.dropna(subset=['Target'])
    early_stop = EarlyStopping(
        monitor='val_loss',       # You can also use 'val_accuracy'
        patience=6,               # Wait this many epochs after no improvement
        restore_best_weights=True # Roll back to the best weights seen during training
    )
    
    # Encode input features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    try:
        encoded_features = encoder.fit_transform(df[categorical_features])
    except KeyError as ke:
        print(f"Missing feature during encoding: {ke}")
        return None
    except Exception as e:
        print (f"Unexpected error during encoding: {e}")
        return None

    scaler = StandardScaler()
    scaled_continuous = scaler.fit_transform(df[continuous_features])

    # Encode target
    target_encoder = OneHotEncoder(sparse_output=False)
    encoded_target = target_encoder.fit_transform(df[['Target']])

    # Now define datasets properly
    X = np.concatenate([encoded_features, scaled_continuous], axis=1)
 
    y = encoded_target    

    # MODELING
    ##########
    features_for_model = ['Year', 'Month', 'Day', 'TIME OCC', 'AREA', 'Rpt Dist No',
                         'Part 1-2', 'Crm Cd', 'Vict Age', 'Vict Sex', 'Mocodes_freq',
                         'Vict Descent', 'Premis Cd', 'Weapon Used Cd']
    target_features = ['Arrest', 'No Arrest']

    X_train, X_test, y_train, y_test = train_test_split(X, y , 
                                                        test_size=0.01, 
                                                        random_state=42)

    # Set up the layers
    ###################
    # The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. Hopefully, these representations are meaningful for the problem at hand.
    # Most of deep learning consists of chaining together simple layers. Most layers, such as tf.keras.layers.Dense, have parameters that are learned during training.

    # Create network topology
    model = keras.Sequential()

    # Adding input model --> 15 input layers
    model.add(Dense(15, input_dim = X_train.shape[1], activation = 'relu'))

    # Adding hidden layer 
    model.add(keras.layers.Dense(30, activation="relu"))
    model.add(keras.layers.Dense(60, activation="relu"))
    model.add(keras.layers.Dense(60, activation="relu"))
    model.add(keras.layers.Dense(30, activation="relu"))

    # output layer
    # For classification tasks, we generally tend to add an activation function in the output ("sigmoid" for binary, and "softmax" for multi-class, etc.).
    model.add(keras.layers.Dense(2, activation="softmax"))

    print(model.summary())
    # Compile the Model
    ###################
    model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
    # Train the Model
    #################
    print("Training..\n")
    try:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 7, batch_size = 256)
    except tf.errors.ResourceExhaustedError:
        print("Error: Not enough memory to train the model.")
        return None
    except ValueError as ve:
        print(f"Value error during training: {ve}")
        return None
    except Exception as e:
        print(f"Unexpected error during training: {e}")
        return None
    print("Done Training..\n")
    #Evaluate accuracy
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    test_acc = test_acc * 100
    test_loss = test_loss * 100
    print(f'\nTest accuracy: {test_acc}%')
    print(f'\nLoss: {test_loss}%')
    #df = df.dropna(subset=['Target'])
    

    
    return model, encoder, target_encoder, scaler


# In[48]:


## UI PART OF PROGRAM
####################
user_input = 0
# Using Set to eliminate dupes
user_input_set = {0}
try:
    while (user_input != 7):
        print("(1) Load training data")
        print("(2) Process (Clean) data")
        print("(3) Train NN")
        print("(4) Load testing data")
        print("(5) Generate Predictions")
        print("(6) Print Accuracy (Actual Vs Predicted")
        print("(7) Quit")
        try:
            user_input = input("Enter digit 1 - 7: ")
        except KeyboardInterrupt:
            print("\nUser cancelled. Exiting...")
            user_input = 7
            break
        print("")
        
        # Check for invalid user inputs
        try:
            user_input = int(user_input)
            if (user_input < 1 or user_input > 7):
                print("Invalid input\n")
                continue;
        except (ValueError, KeyboardInterrupt, EOFError):
            print("Invalid input or user cancelled\n")
            continue;
        except:
            print("Invalid input\n")
            continue
     
        if (user_input == 1):
            print("Load training data set:")
            print("***********************")
            
            training_data = 0
            try: 
                training_file_name = input("Input Training Data file name: ").strip()
            except KeyboardInterrupt:
                print("\nUser cancelled. Exiting...")
                break
        
            if os.path.isfile(training_file_name):
                print("\nTraining file found! Proceeding with loading...\n")
                training_data = loading_set(training_file_name)
                user_input_set.add(1)
                continue
            else:
                print("\nFile not found! Check the filename and try again.\n")
          
        elif 1 in user_input_set and user_input == 2:
            print("Process (Clean) data:")
            print("*********************")
            
            #######################################################################
            start_time2 = time.time()
            
            clean_time = time.localtime()
            clean_time = time.strftime("%Y-%m-%d %H:%M:%S", clean_time)
            try:
                print(f"[{clean_time}] Performing Data Clean Up")
                file_path = training_file_name
                training_data, status = cleanFile(training_file_name, 0)
                training_data.to_csv('LA_Crime_Data_2023_to_Present_clean_data.csv',
                                     index=False)
            except:
                print("Error trying to clean file")
            end_time2 = time.time()
            elapsed_time = end_time2 - start_time2
            print(f"Time to Process is: {elapsed_time:.2f}secs")
            #######################################################################
    
            user_input_set.add(2)
            user_input_set.remove(1)
            
        elif 2 in user_input_set and user_input == 3:
            print("\nTrain NN:")
            print("*********")
            target = ['Target']
            continuous_features = ['Year', 'Month', 'Day', 'Part 1-2', 
                                   'Mocodes_freq', 'Vict Age'] 
            categorical_features = ['AREA', 'Rpt Dist No', 'TIME OCC', 'Crm Cd', 
                                    'Premis Cd', 'Weapon Used Cd', 'Vict Sex', 'Vict Descent']
            try:
                model, encoder, target_encoder, scaler = trainNN(training_data)
            
                user_input_set.add(3)
                user_input_set.remove(2)
            except:
                print("Error Training Model")
        elif 3 in user_input_set and user_input == 4:
            print("\nLoading testing data set:")
            print("*************************")  
            loaded = False
            testing_data = 0
            testing_file_name = input("Input file name of testing data: ").strip()
            try:
                if os.path.isfile(testing_file_name):
                    print("\nTesting file found! Proceeding with loading...\n")
                    start_time3 = time.time()
                    testing_data = loading_set(testing_file_name)
                    loaded = True
                else:
                    print("\nFile not found! Check the filename and try again.\n")
                
                if loaded:     
                    test_data, status = cleanFile(testing_file_name, 1)
                    test_data.to_csv('LA_Crime_Data_2023_to_Present_clean_test1.csv',
                                  index=False)
                    end_time3 = time.time()
                    elapsed_time = end_time3 - start_time3
                    print(f"Time to load is: {elapsed_time:.2f}secs")
                    user_input_set.add(4)
                else:
                    print("Error CLeaning Testing data, please try again...")
            except:
                print("Error cleaning file")
    
        elif 4 in user_input_set and user_input == 5:
    
            # ================================
            # Prediction Phase Description
            # ================================
            # This section takes the cleaned and preprocessed testing data and uses the
            # trained neural network model to generate predictions. It includes:
            # - Transforming the target column using the trained target encoder
            # - Scaling continuous features and encoding categorical features
            # - Combining them into a single input array for the model
            # - Running predictions using the trained model
            # - Displaying sample predictions for inspection
            # - Converting prediction indices to one-hot format
            # - Decoding them back to human-readable class labels
            # - Adding step 5 to the user progress tracker if successful
            
            print("\nGenerate Predictions:")
            print("*********************")    
            try:
                if status == 1:
                    y_test = target_encoder.transform(test_data[['Target']])
                print("Target found in Dataframe")
                # First, make sure continuous features are standardized using the same scaler from training
                scaled_continuous_test = scaler.transform(test_data[continuous_features])
                
                encoded_categorical_test = encoder.transform(test_data[categorical_features])
                # Concatenate the encoded categorical features with the scaled continuous features
                X_test = np.concatenate([encoded_categorical_test, scaled_continuous_test], axis=1)
                print("X_test shape:", X_test.shape)
                # Make Predictions
                predictions = model.predict(X_test)
                
                # Here, the model has predicted the label for each image in the testing set.
                print(predictions[0])
                if X_test.shape[0] >= 10:
                    print(predictions[10])
                if X_test.shape[0] >= 100:
                    print(predictions[100])
                if X_test.shape[0] >= 1000:
                    print(predictions[1000])
                if X_test.shape[0] >= 10000:
                    print(predictions[10000])
                if X_test.shape[0] >= 100000:
                    print(predictions[100000])
                # Step 1: Predict the class probabilities
                predictions = model.predict(X_test)
                
                # Step 2: Get the predicted class indices (e.g., class 0, class 1, etc.)
                predicted_classes = np.argmax(predictions, axis=1)
                
                # Step 3: Convert predicted class indices to one-hot encoding
                one_hot_predictions = np.zeros((predicted_classes.size, predicted_classes.max() + 1))
                one_hot_predictions[np.arange(predicted_classes.size), predicted_classes] = 1
                
                # Step 4: Decode the predicted classes back to their original labels using the target encoder
                decoded_predictions = target_encoder.inverse_transform(one_hot_predictions)
                
                #print(decoded_predictions[:15])
                
                # RSME Calculations
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                
                print("RMSE:", rmse)
                
                if status:
                    user_input_set.add(5)
            except:
                print("failed to make predictions")
    
        elif 5 in user_input_set and user_input == 6 and status == 1:
            # ===========================================
            # Model Evaluation and Accuracy Calculation
            # ===========================================
            # This block:
            # - Verifies that a model and predictions exist
            # - Evaluates the model using test data, printing accuracy and loss
            # - Attempts to decode the true labels (y_test) and predicted labels
            # - Displays a sample comparison between actual and predicted values
            # - Visualizes performance using a plot and prints metrics
            # - Handles potential errors in decoding and evaluation
            # - Updates user input state upon successful evaluation
            try: 
                if model is None:
                    print("Error: Model is not trained. Cannot calculate accuracy.")
                    continue
        
                if predictions is None:
                    print("Error: No predictions available to evaluate. Cannot calculate accuracy.")
                    continue
                
                print("Accuracy of prediction is:")
                print("**************************")
                
                # Evaluate
                loss, acc = model.evaluate(X_test, y_test)
               
        
                # getting y_test values
                try: 
                    y_tested = target_encoder.inverse_transform(y_test)
                except ValueError as ve:
                    print(f"Error in inverse transforming Y_test: {ve}")
                    y_tested = ["Invalid"] * len(y_test) # Creates a list of length y_test with all values "Invalid"
                except Exception as e:
                    print (f"Unexpected error during inverse transforming y_test: {e}")
                    y_tested = ["Invalid"] * len(y_test)
        
        
                # getting the value of the predictions
                try: 
                    y_predicted = target_encoder.inverse_transform(predictions)
                except ValueError as ve:
                    print(f"Error converting predictions: {ve}")
                    y_predicted = ["Invalid"] * len(predictions)
        
                # printing the first 25 values of the test and predicted values 
                data = []
                for i in range(15):
                    data.append([y_tested[i], y_predicted[i]])
        
                headers = ["Actual Value", "Predicted Value"]
        
                print(tabulate(data, headers=headers, tablefmt="grid"))
                class_labels=['Arrest', 'No Arrest']
                plot_prediction_vs_test_categorical(y_tested,
                                                    y_predicted, class_labels)
                metrics = calculate_performance_multiclass(y_tested, y_predicted)
                print(calculate_performance_multiclass(y_tested, y_predicted))
             

                metrics_arrest = metrics['confusion_matrix'][0][0]
                metrics_no_arrest = metrics['confusion_matrix'][1][1]

                total_correct = metrics_arrest + metrics_no_arrest
                print("")
                print(f"{total_correct} of correct predicted observations ")
                print(f"{acc*100:.4f}% of correct predicted observations")
                
                user_input_set.add(6)
            except:
                print("Error getting Accuracy")
                
        elif user_input == 7:
            print("\nQuitting Application")
        else:
            try:
                if status == 0:
                    print("Accuracy not available")
                elif 3 in user_input_set:
                    print("Please Start from 1")
                else:
                    print("\nPlease go in order")
            except:
                print("Error with UI")
        print("")
except:
    print("Keyboard Interrupted/n/nQuitting Application")


# In[ ]:





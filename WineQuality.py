# <font color=darkblue>SECTION 1: ANALYSIS AND IMPLEMENTATION OF DATA </font>

#Our selected project is the quality of red wines based on several different factors. Although most alcohols require a lot of care and resources to ensure good quality, red wines in particular are quite strict with what classifies as a "good" or a "bad" wine. In this particular project, a database provided by Paulo Cortez and his associates from the University of Minho in Portugal listed several measureable factors of about 1600 red and white wine samples of the brand "Vinho Verde" and determined whether they qualified as good or bad based on a scale from 3 to 8. Anything above or equal to a score of 6 was considered a good wine and anything below was considered bad for the purposes of our project. A description of all the factors are listed below.

#**Fixed Acidity** - Acidity is one of the most significant factors in wine, as different acidity values affect the flavor in different ways. Fixed acids are ones that do not evaporate readily, namely tartaric acid. Wines that are high in acidity tend to be more tart and ones that are low tend to have a softer, more subtle flavor.

#**Volatile Acidity** - Volatile acids are those that do evaporate readily and contribute to the smell and vinegar flavor in the wine. A primary volatile acid in wine is acetic acid. This factor should ideally be kept lower, as a higher volatile acidity causes the wine to have an unpleasant vinegar taste and reuslt in a bad wine.

#**Citric Acid** - Another acidity to add to the list, the database states that a small amount of citric acid can give the wine a fresher flavor. Based on this, a good wine would have a citric acid value that is more than 0, but this value should not be excessive, as with everything else.

#**Residual Sugar** - The amount of sugar leftover after the wine fermentation process. Yeast, which is used to ferment the wine, uses sugar as "food", so naturally, the more sugar residual sugar in the wine after the yeast finishes its reaction, the sweeter the wine will be. According to the database, a red wine should have a good sweetness to it, but not excessively, as red wine should have a drier taste, especially when compared to white wine.

#**Chlorides** - Chlorides are simply the measurement of the amount of salt in the wine. Of course, wine should not have a salty flavor, so this value should be kept *very* low. Salt does allow for the enhancement of certain flavors and the color of the wine when used in moderation, but too much can cause the red wine, which has high concentration in tannins, to taste bitter and metallic. Therefore, once again, chlorides should stay quite low overall in order to make a good wine.

#**Free Sulfur Dioxide** - Free Sulfur Dioxide (SO2) contributes to the prevention of oxidation and spoilage of wine. This value would be a bit higher than the others due to its benefits to the overall quality of the wine, but too much can mask the overall flavor of the wine and even cause a bitter or metallic flavor. 

#**Total Sulfur Dioxide** - The overall amount of free sulfur dioxide and sulfur dioxide molecules that are bound to other chemicals like sugars and pigments. This, of course, contributes to the flavor of the wine in the same way as free sulfur dioxide.

#**Density** - Self-explanatory, wine density is simply how dense or concentrated the wine is. This value remains very close to the density of water, only slightly changed depending on the sugar and alcohol content, so this should not have too much of an effect on the overall quality of wine on its own.

#**pH** - The overall acidity of the wine. According to the database, most wines fall between 3 and 4. As before, low pH means higher acidity, which would result in a more sour taste in the wine. Higher pH means lower acidity, which would result in a more rounded wine. Red wines in particular should fall closer to the middle of that range, around 3.3-3.6, to be considered a good wine.

#**Sulphates** - An additive that can contribute to the amount of SO2 in the wine. As mentioned before, SO2 prevents oxidation and bacterial growth, which would result in a better wine for a longer period of time. And just like with sulfur dioxide, too high of a concentration of sulphates results in a dulling of the wine's flavor.

#**Alcohol** - A measure of the alocholic content of the wine. Wines are usually lower in alcohol content when compared to other alcoholic drinks, but it still has quite the effect on the flavor. A higher alcohol content will result in a richer wine with a full taste and body while a lower alcohol will be a bit more balanced. Cortez et al found through their research that a higher alcohol content tends to result in a higher quality wine.

#**Quality** - The final score of the wine after accounting for all of the above factors. As stated, this score ranges from 3 to 8, 3 being the lowest quality and 8 being the highest. This project aims to predict the quality of a wine based on given values for the other factors.

#First lets start off with Data Acquisition 
#CSV file was provided in the project folder so
#Let's start with:
#Importing Libraries; I'm just importing basically everything, some might not be used
import numpy as np
import pandas as pd
import statistics
import scipy.stats
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
#Here's where it might get weird, these were in the MultipleLinearRegression lecture notes
#We're clearly using multi-variable linear regression, so I'll stick these on aswell
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Okay now lets move onto getting the data ready to be manipulated...
#aka reading the CSV file and storing it as a dataframe
wineDF = pd.read_csv('redwinequality.csv')
wineDF.head()

#Cool, dataset successfully imported. Let's run descriptive analysis commands on it
wineDF.info()
wineDF.describe()

#Let's check if there are any null values in any of the columns. If there are, that row should just be removed
wineDF.isnull().sum()

#Next off, let's create a pairplot for the variables. This will provide us a visual graph so we can furthee analyze the
#variables

plots = sns.pairplot(wineDF)
plots.fig.set_size_inches(25,25)
plots.fig.suptitle('Wine Quality Pairplot', y=1.05)

#Let's begin describing the data. Now the pairplot is a bit hard to read, but let's go ahead and make some generalizations that we can easily see despite the graphs being slightly hard to see:
#First, rows 1,2,8,9, and 10 appear to have normal graphs (observed on the diagonal starting from the top left to bottom right of the pairplots). Rows 4 and 5 also seem like they could be normal BUT they are very small and hard to read, which is actually a very key feature that tells a good bit about the data. The size of the diagonal graphs are actually key to identifying outliers. Of course we can't actually see the outliers yet, but a good general rule here is the small the graph, the farther out an outlier is. This means we'll want to check for outliers later.
#Lastly, rows 3 , 6, 7 and 11 seem to be skewed to the right.

#Next lets plot a correlation matrix heatmap, this will allow us to gain a quick glance at what variables will affect
#the regression equation more than others
plt.figure(figsize=(12,12))
plt.title('Correlaton of Features')

sns.heatmap(wineDF.corr(), linewidths=0.1, linecolor='white', annot=True, annot_kws={'size':16});

#This heatmap matrix tells us a few key features to expect from the linear regression model:
#1st: Alcohol will most likely have the most influence on the quality, with a Pearson correlation of .48. Sulphates and Citric Acid follow alcohol at second and third most influence with a Pearson correlation of .25 and .23 respectively.
#2nd: Volatile Acidity will have the least affect, and by a pretty wide margin. This is because the Pearson correlation provided a -.39

#Next, let's go variable by variable graphing each indepedently to take a closer look at the data.
#We'll begin with the dependent variable first; Quality
sns.countplot(y='quality',data=wineDF).set(title='Value Counts of Quality')

#This plot of the value counts of the quality column tells us a few things:
#1. The qualities range form 3 to 8
#2. A majority of the wines are classified as 5 or 6 quality
#3. The value count plot is near-normal

#fAcidPlot = plt.figure(figsize = (10,10))
#plt.hist(wineDF['fixed acidity'])

#Ok SO BASICALLY we keep graphing stuff maybe and continue to describe the dataset using words based off those graphs? I got lazy so im stopping and just gonna go alter the data bc its more fun to actually do stuff instead of typing descriptions.

# <font color=darkblue>SECTION 2.0: CREATION OF FROM-SCRATCH MODEL </font>

#To create the model by hand, we must first change the data into standard units
#Alright so we now let's begin modifying the data so we can use it
#Because there are a whopping 11 whole variables I will take the libery to create
#A new dataframe instead of adding 11 new columns to our existing one
stUnits = pd.DataFrame()
#Now let's add a column for each variable converted to standard units
stUnits['fixed acidity(standard units)'] = (wineDF['fixed acidity'] - np.mean(wineDF['fixed acidity']))/(np.std(wineDF['fixed acidity']))
stUnits['volatile acidity(standard units)'] = (wineDF['volatile acidity'] - np.mean(wineDF['volatile acidity']))/(np.std(wineDF['volatile acidity']))
stUnits['citric acid(standard units)'] = (wineDF['citric acid'] - np.mean(wineDF['citric acid']))/(np.std(wineDF['citric acid']))
stUnits['residual sugar(standard units)'] = (wineDF['residual sugar'] - np.mean(wineDF['residual sugar']))/(np.std(wineDF['residual sugar']))
stUnits['chlorides(standard units)'] = (wineDF['chlorides'] - np.mean(wineDF['chlorides']))/(np.std(wineDF['chlorides']))
stUnits['free sulfur dioxide(standard units)'] = (wineDF['free sulfur dioxide'] - np.mean(wineDF['free sulfur dioxide']))/(np.std(wineDF['free sulfur dioxide']))
stUnits['total sulfur dioxide(standard units)'] = (wineDF['total sulfur dioxide'] - np.mean(wineDF['total sulfur dioxide']))/(np.std(wineDF['total sulfur dioxide']))
stUnits['density(standard units)'] = (wineDF['density'] - np.mean(wineDF['density']))/(np.std(wineDF['density']))
stUnits['pH(standard units)'] = (wineDF['pH'] - np.mean(wineDF['pH']))/(np.std(wineDF['pH']))
stUnits['sulphates'] = (wineDF['sulphates'] - np.mean(wineDF['sulphates']))/(np.std(wineDF['sulphates']))
stUnits['alcohol(standard units)'] = (wineDF['alcohol'] - np.mean(wineDF['alcohol']))/(np.std(wineDF['alcohol']))
#Lets add the quality column
stUnits['quality'] = wineDF['quality']
stUnits.head(10)


#Next, there's actually another column that we need to create for this logistic regression
#We use logistic regression due to classifying the wine between two binary options, so we will have to classify our data
#To make this simple we will use the follow rule
# 0 = Bad wine
# 1 = Good wine
#As a reminder QS>=6 is good, QS<=5 is bad

#function to classify each row as bad(0) or good(1)
def qualityScore_function(x):
    if x <=5:
        return 0
    elif x >= 6:
        return 1

#code to add the new column
#Adding this column to both the standardized dataframe, and the original, as we will use the original
#with the data science library
stUnits['quality score']= stUnits['quality'].apply(qualityScore_function)
wineDF['quality score']= wineDF['quality'].apply(qualityScore_function)
wineDF.head()

#this is just a simple check to see if the above apply function worked
stUnits.head()

#Next we randomly shuffle (without replacement) all the rows in the dataframe
#Picking 1599 random rows, the first 66% of these rows will be the training set
#The latter 33% will be used as the test set.

random = stUnits.sample(1599, replace = False)
training_set = random.iloc[0:1066, :]
test_set = random.iloc[1066:, :]

#Next for logistic regression we have to separate the features and response variable
#in the training and test sets

x_train = training_set.iloc[:, 0:11].values
y_train = training_set.iloc[:, 12].values
y_train = np.reshape(y_train, (len(y_train), 1))

x_test = test_set.iloc[:, 0:11].values
y_test = test_set.iloc[:, 12].values
y_test = np.reshape(y_test, (len(y_test), 1))

print("x_train_Shape:", np.shape(x_train))
print("y_train_Shape:", np.shape(y_train))

print("x_test_Shape:", np.shape(x_test))
print("y_test_Shape:", np.shape(y_test))

#Now we need to vectorize the training and test sets by transposing and stacking a row of ones vertically

x_train_trans = np.transpose(x_train)
x_train_Aug = np.vstack((np.ones((1,len(x_train))),x_train_trans))
print("x_train_Aug:", np.shape(x_train_Aug))

x_test_trans = np.transpose(x_test)
x_test_Aug = np.vstack((np.ones((1,len(x_test))),x_test_trans))
print("x_test_Aug:", np.shape(x_test_Aug))



#Next let's define theta as an array of zeroes
theta = np.zeros((12,1))
print("theta:", np.shape(theta))

#Implementing the gradient descent algorithm
no_of_iter = np.arange(1, 30001)
alpha = 0.003
m_train = len(x_train)
m_test = len(x_test)

costfunc = []

#Iteration loop
for i in no_of_iter:
    Z = np.transpose(theta)@x_train_Aug
    p = 1/(1+np.exp(-Z))
    ft = ((np.log10(1/(1+np.exp(-Z))))@y_train)[0,0]  
    st = ((np.log10(1-(1/(1+np.exp(-Z)))))@(1-y_train))[0,0]
    cf = (1/m_train)*(-ft-st) #Cost function
    costfunc.append(cf)
    delthetaj = (1/m_train)*((x_train_Aug)@(np.transpose(p)-y_train)) #Derivative of cost function
    theta = theta - (alpha*delthetaj) #Updating theta values

print(Z)
print(len(costfunc))
print(theta)

#Testing the model on the training set

y_train_pred = np.zeros((m_train,1))
h_Theta = np.transpose(theta)@x_train_Aug
h_Theta_trans = np.transpose(h_Theta)


#Comparing predicted and actual results

for j in range(m_train):
    if  1/(1+np.exp(-h_Theta_trans[j])) >= 0.5: 
        y_train_pred[j] = [1]
    else: 
        y_train_pred[j] = [0]

#Testing the model on the test set

y_test_pred = np.zeros((m_test,1))
h_Theta = np.transpose(theta)@x_test_Aug
h_Theta_trans = np.transpose(h_Theta)


#Comparing predicted and actual results

for j in range(m_test):
    if  1/(1+np.exp(-h_Theta_trans[j])) >= 0.5: 
        y_test_pred[j] = [1]
    else: 
        y_test_pred[j] = [0]

print(x_train_Aug)

#Plotting cost function vs. number of iterations

plt.plot(no_of_iter,costfunc,color='r',linewidth = '3')
plt.xlabel("Number of iterations")
plt.ylabel("Cost function")
plt.title("Cost function vs. number of iterations")

#Model evaluation on the training set

#True positive

count_TP=0
for TP in range(m_train):
    if (y_train_pred[TP] == 1) & (y_train[TP] == 1): 
        count_TP = count_TP+1
print("True_Positives:",count_TP)


#False positive
count_FP=0
for FP in range(m_train):
    if (y_train_pred[FP] == 1) & (y_train[FP] == 0): 
        count_FP = count_FP+1
print("False Positives:",count_FP)


#True negative
count_TN=0
for TN in range(m_train):
    if (y_train_pred[TN] == 0) & (y_train[TN] == 0): 
        count_TN = count_TN+1
print("True Negatives:",count_TN)


#False negative
count_FN=0
for FN in range(m_train):
    if (y_train_pred[FN] == 0) & (y_train[FN] == 1): 
        count_FN = count_FN+1
print("False Negatives:",count_FN)

Accuracy = (count_TP+count_TN)/m_train
print("Accuracy:", Accuracy)

Precision =count_TP/(count_TP+count_FP)
print("Precision:", Precision)

Recall =count_TP/(count_TP+count_FN)
print("Recall:", Recall)

F1_Score = (2*Precision*Recall)/(Precision+Recall)
print("F1_Score:", F1_Score)

#Model evaluation on the test set

#True positive

count_TP=0
for TP in range(m_test):
    if (y_test_pred[TP] == 1) & (y_test[TP] == 1): 
        count_TP = count_TP+1
print("True_Positives:",count_TP)


#False positive
count_FP=0
for FP in range(m_test):
    if (y_test_pred[FP] == 1) & (y_test[FP] == 0): 
        count_FP = count_FP+1
print("False Positives:",count_FP)


#True negative
count_TN=0
for TN in range(m_test):
    if (y_test_pred[TN] == 0) & (y_test[TN] == 0): 
        count_TN = count_TN+1
print("True Negatives:",count_TN)


#False negative
count_FN=0
for FN in range(m_test):
    if (y_test_pred[FN] == 0) & (y_test[FN] == 1): 
        count_FN = count_FN+1
print("False Negatives:",count_FN)

Accuracy = (count_TP+count_TN)/m_test
print("Accuracy:", Accuracy)

Precision =count_TP/(count_TP+count_FP)
print("Precision:", Precision)

Recall =count_TP/(count_TP+count_FN)
print("Recall:", Recall)

F1_Score = (2*Precision*Recall)/(Precision+Recall)
print("F1_Score:", F1_Score)

# <font color=darkblue>SECTION 2.5: CREATION OF DATA SCIENCE LIBRARY MODEL </font>

#First lets separate our features and response variable.

x = wineDF.iloc[:,0:11]
y = wineDF.iloc[:,12]
print(x.head())
print(y.head())

#Feature scaling the 3 feature columns

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x.astype(float))
x

#Splitting the dataset into training set and the test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.333333)

#Training the model based on 'x_train and y_train' and getting the coefficients and the intercept

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=30000, solver = 'lbfgs')
logmodel.fit(x_train,y_train)

print(logmodel.coef_)
print(logmodel.intercept_)

#Testing the model on the training set

y_train_pred = logmodel.predict(x_train)

#Testing the model on the test set

y_test_pred = logmodel.predict(x_test)



#Model evaluation on the training set

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_train,y_train_pred))
print(confusion_matrix(y_train,y_train_pred))

#Model evaluation on the test set

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))

# <font color=darkblue>SECTION 3: PREDICTION USER INTERFACE </font>

#Simple user interface: Prompt for 11 inputs
#Output two predictions: 1st prediction is 'from-scratch', 2nd is from scikit library
facid = float(input('Input the Fixed Acidity: '))
vacid = float(input('Input the Volatile Acidity: '))
cacid = float(input('Input the Citric Acid: '))
sugar = float(input('Input the Residual Sugar: '))
chlorides = float(input('Input the Chlorides: '))
sulfur = float(input('Input the Free Sulfur Dioxide: '))
tsulfur = float(input('Input the Total Sulfur Dioxide: '))
dens = float(input('Input the Density: '))
pH = float(input('Input the pH: '))
sulphates = float(input('Input the Sulphates: '))
alcohol = float(input('Input the Alcohol: '))

#Create a 2d array full of the variables
wineVariables = [[ facid, vacid, cacid, sugar, chlorides, sulfur, tsulfur, dens,pH,sulphates, alcohol]]
#2nd array that can be modified for from scratch, just a copy
wineCopy = wineVariables
#Really long prediction required for from scratch
scratchPrediction = -1
wineCopy_trans = np.transpose(wineCopy)
wineCopy_Aug = np.vstack((np.ones((1,len(wineCopy))), wineCopy_trans))
theta = np.zeros((12,1))
wine_pred = np.zeros((len(wineCopy),1))
h_Theta = np.transpose(theta)@wineCopy_Aug
h_Theta_trans = np.transpose(h_Theta)
if  1/(1+np.exp(-h_Theta_trans[0])) >= 0.5:
    scratchPrediction = 1
else:
    scratchPrediction = 0

if scratchPrediction == 0:
    print('Scratch says: Bad Wine')
elif scratchPrediction == 1:
    print('Scratch says: Good Wine')

#Simple Scikit Prediction
predictedQuality = logmodel.predict(wineVariables)
if predictedQuality == 0:
    print('Scikit says: Bad Wine')
elif predictedQuality == 1:
    print('Scikit says: Good Wine')

# <font color=darkblue>SECTION 4: PREDICTION UPDATING DATABASE </font>

#To update the data, let's create a new dataframe
updateWineDF = pd.read_csv('redwinequality.csv')
#Then we will add the good or bad column
#function to classify each row as bad(0) or good(1)
def qualityScore_function(x):
    if x <=5:
        return 0
    elif x >= 6:
        return 1

#code to add the new column
#Adding this column to both the standardized dataframe, and the original, as we will use the original
#with the data science library

updateWineDF['quality score']= updateWineDF['quality'].apply(qualityScore_function)
updateWineDF.head()

facid = float(input('Input the Fixed Acidity: '))
vacid = float(input('Input the Volatile Acidity: '))
cacid = float(input('Input the Citric Acid: '))
sugar = float(input('Input the Residual Sugar: '))
chlorides = float(input('Input the Chlorides: '))
sulfur = float(input('Input the Free Sulfur Dioxide: '))
tsulfur = float(input('Input the Total Sulfur Dioxide: '))
dens = float(input('Input the Density: '))
pH = float(input('Input the pH: '))
sulphates = float(input('Input the Sulphates: '))
alcohol = float(input('Input the Alcohol: '))

#Create a 2d array full of the variables
wineVariables = [[ facid, vacid, cacid, sugar, chlorides, sulfur, tsulfur, dens,pH,sulphates, alcohol]]
#2nd array that can be modified for from scratch, just a copy
wineCopy = wineVariables
#Really long prediction required for from scratch
scratchPrediction = -1
wineCopy_trans = np.transpose(wineCopy)
wineCopy_Aug = np.vstack((np.ones((1,len(wineCopy))), wineCopy_trans))
theta = np.zeros((12,1))
wine_pred = np.zeros((len(wineCopy),1))
h_Theta = np.transpose(theta)@wineCopy_Aug
h_Theta_trans = np.transpose(h_Theta)
if  1/(1+np.exp(-h_Theta_trans[0])) >= 0.5:
    scratchPrediction = 1
else:
    scratchPrediction = 0

updateWineDF.loc[len(updateWineDF)] = [ facid, vacid, cacid, sugar, chlorides, sulfur, tsulfur, dens,pH,sulphates, alcohol, 'none', scratchPrediction]
print('Database Updated')

updateWineDF.tail()


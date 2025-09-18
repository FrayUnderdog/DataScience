import numpy as np
import pandas as pd
import miceforest as mice


input_dataframe = pd.read_csv("Nutrition.csv")
print(input_dataframe)

#Smooth Noises in Protein Level using Binning
input_dataframe['protein_level'] = pd.qcut(input_dataframe['protein'], q=3)
input_dataframe['protein'] = pd.Series ([interval.mid for interval in input_dataframe['protein_level']])
print(input_dataframe)
del input_dataframe['protein_level']


##Detect Outliers in Calories
q3, q1 = np.percentile(input_dataframe['calories'], [75 ,25], method='inverted_cdf')
fence = 1.5 * (q3 - q1)
upper_bound = q3 + fence
lower_bound = q1 - fence
print( 'q1=',q1,' q3=', q3, ' IQR=', q3-q1, ' upper=', upper_bound, 'lower=',lower_bound)
input_dataframe.loc[(input_dataframe['calories'] < lower_bound) | (input_dataframe['calories'] > upper_bound), 'calories'] = None
print(input_dataframe)


##Encoding Categorical Variable (Manufacturer)
input_dataframe= pd.get_dummies(input_dataframe, dtype='int')
print(input_dataframe)


##MICE Imputation
impute = mice.ImputationKernel(data=input_dataframe)
impute.mice(10)
imputed_dataframe = impute.complete_data(0)
print(imputed_dataframe)


##Normalize data
norm_dataframe = (imputed_dataframe - imputed_dataframe.min()) / (imputed_dataframe.max() - imputed_dataframe.min())  #Min_Max_Norm
# norm_dataframe = (imputed_dataframe - imputed_dataframe.mean()) / imputed_dataframe.std() #Standard_Norm
print(norm_dataframe)

##Save Data
norm_dataframe.to_csv("Nutrition_Cleaned.csv",index=False)

'''PANDAS'''
'''1) Series'''
import numpy as np
import pandas as pd
labels=['a','b','c']
my_data=[10,20,30]
arr=np.array(my_data)
d={'a':10,'b':20,'c':30}
s1=pd.Series(data=my_data)
print(s1)
s2=pd.Series(data=my_data,index=labels)#No need to put data= or index=
print(s2)
s3=pd.Series(my_data,labels)
print(s3)
# Pandas are better than numpy as numpy work with only numbers but pandas is very flexible

import numpy as np
import pandas as pd
ser1=pd.Series([1,2,3,4],['USA','Germany','Japan','USSR'])
print(ser1)
print(ser1['Japan'])
ser2=pd.Series([1,2,5,4],['USA','UK','Italy','USSR'])
print(ser1+ser2)#Where the index are same like USA and Ussr,it'll add up and for others where index are differnent at same location,it'll shoe Nan(NOT A NUMBER)

'''2)DataFrames A'''
import numpy as np
import pandas as pd
from numpy.random import randn 
np.random.seed(101)
df=pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z']) 
print(df)
# Using indexing to select columns
print(df[['W','Z']])
print(type(df['W']))
df['new']=df['W']+df['Y']#if we want to add a new column
print(df.drop('new',axis=1))#drop is to remove the column
print(df)#We can see that the drop wasn't permanent so we'll use inplace=True
print(df.drop('new',axis=1,inplace=True))
print(df)#We can see that the 'new' column is permanently removed
print(df.drop('E',axis=0))
#Selecting rows
print(df.loc['A']) #Method 1(using label)
print(df.iloc[0]) #Method 2(using index)
print(df.loc['B','Y'])
print(df.loc[['A','B'],['X','Y']])


'''3)DataFrames B'''
import numpy as np
import pandas as pd
from numpy.random import randn 
np.random.seed(101)
df=pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
print(df>0)
booldf=df>0
print(df[booldf])
print(df[df>0])
print(df[df['W']>0])#We can see row C is removed because it's element in 'W' column was negative
print(df[df['W']>0]['X'])
print(df[df['W']>0][['Y','X']])
print(df[(df['W']>0) & (df['Y']>1)]) #Use bitwise operator for 'and' or 'or' statements


import numpy as np
import pandas as pd
from numpy.random import randn 
np.random.seed(101)
df=pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
print(df.reset_index())
newind='PU MP OR HP JMU'.split()
df['States']=newind
print(df)
print(df.set_index('States',inplace=True))
print(df)

'''4)DataFrames C'''
import numpy as np
import pandas as pd
from numpy.random import randn
# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
print(hier_index)
df=pd.DataFrame(randn(6,2),hier_index,['A','B'])
print(df)
df.index.names=['Groups','Num']
print(df)
print(df.loc['G2'].loc[2]['B'])
print(df.xs('G1'))
print(df.xs(1,level='Num'))

'''5)Missing Data'''
import numpy as np
import pandas as pd
d={'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}
df=pd.DataFrame(d)
print(df)
print(df.dropna(axis=0))
print(df.dropna(axis=1)) #drop rows or columns with NaN
print(df.dropna(thresh=2))#min no of Nan should be 2 before dropping
print(df.fillna(value='FillVALUE'))
print(df['A'].fillna(df['A'].mean()))

'''6)Groupby'''
import numpy as np
import pandas as pd
# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]} 
df=pd.DataFrame(data)
print(df)
byComp=df.groupby('Company')
# print(byComp) #Will return memory location
print(byComp.sum())
# print(byComp.std()) SOMETHING IS WRNG AS STRINGS ARE ALSO BEING USED IN FUNCTION
df=df.groupby('Company').describe()
print(df)
print(df.transpose())

'''7)Merging,Joining and Concatenating'''
# CONCATENATING
import numpy as np
import pandas as pd
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3]) 
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])

dfa=pd.concat([df1,df2,df3],axis=0)
dfb=pd.concat([df1,df2,df3],axis=1)
print(dfa)
print(dfb)
# MERGING
import pandas as pd
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})  
print(pd.merge(left,right,how='inner',on='key'))
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})
print(pd.merge(left, right, how='outer', on=['key1', 'key2']))
# JOINING
# Joining is a convenient method for combining the columns of two potentially differently-indexed DataFrames
# into a single result DataFrame.
import pandas as pd
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3']) 
print(left.join(right))
print(left.join(right,how='outer'))

'''8)Operations'''








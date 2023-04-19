# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
NAME:NIROSHA.S
ROLL NUMBER:212222230097
~~~.py
DATA.CSV
import pandas as pd
df=pd.read_csv("data.csv")
df

# feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
~~~

# ENCODING.CSV
~~~.py
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

# feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
~~~
# TITANIC.CSV

~~~.py
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
~~~

# OUPUT:
# DATA CSV
# Initial Dataset:

![DS O1](https://user-images.githubusercontent.com/121418437/232954367-3df14350-ff94-40a5-8815-6bde87d1760b.png)

# Binary Encoding:

![DS O2](https://user-images.githubusercontent.com/121418437/232954498-7ec5387a-01d3-4f75-813c-fc6737852ae2.png)

![DS O3](https://user-images.githubusercontent.com/121418437/232954520-6e0fc272-7b95-4a00-b254-c430caf5b945.png)

# Encoded Dataset:

![DS O4](https://user-images.githubusercontent.com/121418437/232954537-2376408b-5bbf-49b6-bc28-2335540af010.png)

# Data Scaling using MinMaxScaler:

![DS O5](https://user-images.githubusercontent.com/121418437/232954576-3d40f663-6e13-47e5-9c8e-53982532ed98.png)

# Data Scaling using StandardScaler:

![DS O6](https://user-images.githubusercontent.com/121418437/232954606-25210ddc-2698-461b-a9ed-b9735f7085c2.png)

# Data Scaling using MaxAbsScaler:

![DS O7](https://user-images.githubusercontent.com/121418437/232954723-92317942-e19b-469b-9e14-a9d3db99234d.png)

# Encoding.csv :
# Initial Dataset:

![DS O8](https://user-images.githubusercontent.com/121418437/232954793-162ff925-a856-4670-82db-9cf63e5ac0fd.png)

# Binary Encoding: 

![DS O9](https://user-images.githubusercontent.com/121418437/232954812-78b561f6-8893-4600-8409-c1691e629a18.png)

![DS O10](https://user-images.githubusercontent.com/121418437/232954899-07de12ce-b068-4e21-8887-356a9ce79a00.png)

# Encoded Dataset:

![DS O11](https://user-images.githubusercontent.com/121418437/232954930-06ec7c84-11b3-4254-a065-05078be17d75.png)

# Data Scaling using MinMaxScaler:

![DS O12](https://user-images.githubusercontent.com/121418437/232955097-d4f62375-3194-43c6-a582-f65ef5dbedf1.png)

# Data Scaling using MaxAbsScaler:

![DS O13](https://user-images.githubusercontent.com/121418437/232955129-080c68c9-2c69-469e-97d7-ca5a60025a64.png)

# Data Scaling using RobustScaler:

![DS O14](https://user-images.githubusercontent.com/121418437/232955172-c758c4ad-2971-4cfc-b2f3-4ab506bb29ec.png)

# Titanic.csv :
# Initial Dataset:

![DS O15](https://user-images.githubusercontent.com/121418437/232955248-6442e459-ceb2-49b9-a5e2-fa2fbfc29921.png)

# Data cleaning before encoding:

![DS O16](https://user-images.githubusercontent.com/121418437/232955285-ec51861e-dc16-4eae-81fb-a0f39304a764.png)

![DS O17](https://user-images.githubusercontent.com/121418437/232955305-bb9cd321-957f-4fbf-8ec3-5704dfe03da7.png)

![DS O18](https://user-images.githubusercontent.com/121418437/232955327-974a6099-8e3d-4eb4-8d59-b82c48d59ea6.png)

# Cleaned Dataset:

![DS O19](https://user-images.githubusercontent.com/121418437/232955350-197f0c98-75f5-4e1f-bd65-949a9e78ce77.png)

# Binary Encoding:

![DS O20](https://user-images.githubusercontent.com/121418437/232955375-1e14a005-bfee-4865-9a10-286469374a8f.png)

# Encoded Dataset:

![DS O21](https://user-images.githubusercontent.com/121418437/232955517-c5fcf53d-9fad-4008-9929-80487643a573.png)

# Data Scaling using MinMaxScaler:

![DS O22](https://user-images.githubusercontent.com/121418437/232955576-4d1cf861-87c1-45cd-b844-eab29e56ac56.png)

# Data Scaling using StandardScaler:

![DS O23](https://user-images.githubusercontent.com/121418437/232955604-2784f614-9450-4e1d-a82b-7009c12e803d.png)

# Data Scaling using MaxAbsScaler:

![DS O24](https://user-images.githubusercontent.com/121418437/232955644-f4c6064d-8126-4b71-9079-1a36abc08183.png)

# Data Scaling using RobustScaler:

![DS O25](https://user-images.githubusercontent.com/121418437/232955671-d0f2ba87-dab5-47e8-9968-515f2a94b16d.png)

# RESULT:
   Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.

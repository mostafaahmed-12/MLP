import pandas as pd

data = pd.read_csv("F:\matrail\deep task2\penguins.csv")
columns = []
for i in range(0, 5):
    if i == 0:
        columns.append("target")
    columns.append("X" + str(i + 1))
data.columns = columns


data["X4"].fillna("male", inplace=True)


def convertx4_to_numerical(x):
    if x == "male":
        return 1
    elif x == "female":
        return 2

data["X4"] = data["X4"].apply(convertx4_to_numerical)

def convert_class(x):
    if x == "Adelie":
        return 1
    elif x == "Chinstrap":
        return 3
    elif x == "Gentoo":
        return 2

data["target"]=data["target"].apply(convert_class)


# split the dataset
traing_data = pd.concat([data.iloc[0:30], data.iloc[50:80],data.iloc[100:130]]).sample(frac=1).reset_index()
testing_data=pd.concat([data.iloc[30:50], data.iloc[80:100],data.iloc[130:150]]).sample(frac=1).reset_index()
traing_data.pop("index")
y_training=traing_data["target"]
x_traing=traing_data.iloc[0:,1:]
testing_data.pop("index")
y_testing=testing_data["target"]
x_testing=testing_data.iloc[0:,1:]



from sklearn.preprocessing import MinMaxScaler
std_slc = MinMaxScaler()
std_slc.fit(x_traing)
std_slc.fit(x_testing)
x_testing=std_slc.transform(x_testing)
x_traing= std_slc.transform(x_traing)






















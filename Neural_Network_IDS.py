import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop,adam

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.svm import SVC, LinearSVC

import matplotlib.pyplot as plt
plt.style.use('bmh')

names = ["duration","protocol","service","flag","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot",
        "num_failed_logins","logged_in","num_compromised",
        "root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count",
        "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
        "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serr_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "attack_type","other"]

train_path = "Dataset/NSL-KDD/KDDTrain+.txt"
test_path = "Dataset/NSL-KDD/KDDTest+.txt"

df_train = pd.read_csv(train_path,names=names,header=None)
df_test = pd.read_csv(test_path,names=names,header=None)

print("Shapes of training and testing are:",df_train.shape,df_test.shape)

full_dataset = pd.concat([df_train,df_test])
full_dataset['label'] = full_dataset['attack_type']

full_dataset.loc[full_dataset.label == 'neptune','label'] = 'DOS'
full_dataset.loc[full_dataset.label == 'back','label'] = 'DOS'
full_dataset.loc[full_dataset.label == 'land','label'] = 'DOS'
full_dataset.loc[full_dataset.label == 'pod','label'] = 'DOS'
full_dataset.loc[full_dataset.label == 'smurf','label'] = 'DOS'
full_dataset.loc[full_dataset.label == 'teardrop','label'] = 'DOS'
full_dataset.loc[full_dataset.label == 'mailbomb','label'] = 'DOS'
full_dataset.loc[full_dataset.label == 'processtable','label'] = 'DOS'
full_dataset.loc[full_dataset.label == 'udpstorm','label'] = 'DOS'
full_dataset.loc[full_dataset.label == 'apache2','label'] = 'DOS'
full_dataset.loc[full_dataset.label == 'worm','label'] = 'DOS'

full_dataset.loc[full_dataset.label == 'buffer_overflow','label'] = 'U2R'
full_dataset.loc[full_dataset.label == 'loadmodule','label'] = 'U2R'
full_dataset.loc[full_dataset.label == 'perl','label'] = 'U2R'
full_dataset.loc[full_dataset.label == 'rootkit','label'] = 'U2R'
full_dataset.loc[full_dataset.label == 'sqlattack','label'] = 'U2R'
full_dataset.loc[full_dataset.label == 'xterm','label'] = 'U2R'
full_dataset.loc[full_dataset.label == 'ps','label'] = 'U2R'

full_dataset.loc[full_dataset.label == 'ftp_write','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'guess_passwd','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'imap','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'multihop','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'phf','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'spy','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'warezclient','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'warezmaster','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'xlock','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'xsnoop','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'snmpgetattack','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'httptunnel','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'snmpguess','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'sendmail','label'] = 'R2L'
full_dataset.loc[full_dataset.label == 'named','label'] = 'R2L'

full_dataset.loc[full_dataset.label == 'satan','label'] = 'Probe'
full_dataset.loc[full_dataset.label == 'ipsweep','label'] = 'Probe'
full_dataset.loc[full_dataset.label == 'nmap','label'] = 'Probe'
full_dataset.loc[full_dataset.label == 'portsweep','label'] = 'Probe'
full_dataset.loc[full_dataset.label == 'saint','label'] = 'Probe'
full_dataset.loc[full_dataset.label == 'mscan','label'] = 'Probe'

full_dataset = full_dataset.drop(['other','attack_type'],axis=1)
print("Unique Labels",full_dataset.label.unique())

#One Hot Encoding
full_dataset = pd.get_dummies(full_dataset,drop_first=False)
#Train test split
features = list(full_dataset.columns[:-5])
y_train = np.array(full_dataset[:df_train.shape[0]][['label_normal','label_DOS','label_Probe','label_R2L','label_U2R']])
X_train = full_dataset[:df_train.shape[0]][features]

y_test = np.array(full_dataset[:df_test.shape[0]][['label_normal','label_DOS','label_Probe','label_R2L','label_U2R']])
X_test = full_dataset[:df_test.shape[0]][features]

#Scaling data
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = np.array(scaler.transform(X_train))
X_test_scaled = np.array(scaler.transform(X_test))

print()
print("--------------------Start of Adversarial Sample Generation---------------------")
print()

def NN_model():
    model = Sequential()
    model.add(Dense(256,activation='relu',input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(0.4))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(y_train.shape[1],activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    return model

Neural_Network_Model = NN_model()
Neural_Network_Model.fit(X_train_scaled,y_train,epochs=5,verbose=1,batch_size=32)
scores = Neural_Network_Model.evaluate(X_test_scaled,y_test)
print("Accuracy : ",scores[1]*100)

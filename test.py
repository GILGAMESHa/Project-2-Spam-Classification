import joblib
import pandas as pd
from train import processing, create_feature


i = 'a'
while i == 'a':
    path = input('Please enter the path of email you want to test: ')
    fea = []
    f = open(path, 'r').read()
    contents = processing(f)
    word_indices = create_feature(contents)
    fea.append(word_indices)
    dataframe = pd.DataFrame(fea)
    # store the DataFrame as a .csv, 'index=False' means line label was unseen,'sep='''means default=True
    dataframe.to_csv("test_sample.csv", index=False, sep=',')
    # read train data set
    df = pd.read_csv("test_sample.csv")  # path to train set
    data_set = df.values
    x_test = data_set[:, 1:len(word_indices)]  # features

    predicts = []
    bnb = joblib.load("BNB_model.m")
    predicts.extend(bnb.predict(x_test))

    svm = joblib.load("SVM_model.m")
    predicts.extend(svm.predict(x_test))

    mlp = joblib.load("MLP_model.m")
    predicts.extend(mlp.predict(x_test))

    dt = joblib.load("DT_model.m")
    predicts.extend(dt.predict(x_test))

    knn = joblib.load("KNN_model.m")
    predicts.extend(knn.predict(x_test))
    # obtain the predicted results by integrating the outputs of the most predicted items from classifiers
    print(predicts)
    result = max(predicts, key=predicts.count)
    if result == 1:
        print("It's a normal email!")
    else:
        print("It's a spam!")
    fea = []
    i = input('Do you want to continue to test email?\na. Yes, b. No\n')


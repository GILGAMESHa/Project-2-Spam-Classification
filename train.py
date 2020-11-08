from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import time
import re
import nltk.stem.porter
import matplotlib.pyplot as plt
import seaborn as sns

vectorizer = CountVectorizer(min_df=1)
stemmer = nltk.stem.porter.PorterStemmer()


def processing(email_contents):
    email_contents = email_contents.lower()  # Lowercase all the words
    email_contents = re.sub('<[^<>_]+>', ' ', email_contents)  # delete the 'html' labels
    email_contents = re.sub('[0-9]+', 'number', email_contents)  # transform numbers into 'number'
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)  # replace the 'url' label
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)  # replace the mail address
    email_contents = re.sub('[$]+', 'dollar', email_contents)  # replace the '$' symbol
    return email_contents


df = pd.read_csv('spam.csv', encoding='latin-1')
data = df.values
contents = []
for i in range(1, 11100):
    x = data[i - 1:i:, 1]
    x = x.tolist()
    x[0] = processing(x[0])
    contents.extend(x)

X = vectorizer.fit_transform(contents)
vocab_list = vectorizer.get_feature_names()  # recognize words in the contents and establish a dictionary
# normalize the words to their prototypes
for i in range(1, len(vocab_list) + 1):
    vocab_list[i - 1] = stemmer.stem(vocab_list[i - 1])
vocab = []
# delete the repeated words in the dictionary
for i in range(1, len(vocab_list) + 1):
    if vocab_list[i - 1] not in vocab:
        vocab.append(vocab_list[i - 1])


def create_feature(email_file):
    word = [0]*(len(vocab)+1)
    content = processing(email_file)
    tokens = re.split('[@$/#.-:&*+=\[\]?!(){\},\'\">_<;% \n\t]', content)
    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token)
        if len(token) < 1:
            continue
        # if the word occur in the dictionary, assign the value of 1, whereas 0
        for i in range(1, len(word)):
            if token == vocab[i - 1]:
                word[i] = 1
    return word


if __name__ == '__main__':
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df.head
    #  draw the comparison figure of ham and spam mails
    fig = sns.countplot(df.v1)
    plt.xlabel('Label')
    plt.title('Number of ham and spam messages')
    plt.show
    rate = df['v1'].value_counts()
    sns.set_style("darkgrid")
    bar_plot = sns.barplot(x=(rate.index), y=(rate.values / sum(rate)), palette="muted")
    plt.xticks(rotation=90)
    plt.show()

    features = []  # establish a list for storing feature vectors

    df = pd.read_csv('spam.csv', encoding='latin-1')
    data = df.values
    y = data[:, 0]
    y = y.tolist()
    for i in range(1, 11100):
        x = data[i - 1:i:, 1]
        x = x.tolist()
        content = processing(x[0])
        word_indices = create_feature(content)
        if y[i - 1] == 'ham':
            word_indices[0] = 1
        features.append(word_indices)

    data_frame = pd.DataFrame(features)
    #  store the DataFrame as a .csv, 'index=False' means line label was unseen,'sep='''means default=True
    data_frame.to_csv("train_set.csv", index=False, sep=',')

    # read train data set
    df = pd.read_csv("train_set.csv")  # path to the train set
    data_set = df.values
    x = data_set[:, 1:len(vocab)+1]  # features
    y = data_set[:, 0]  # labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

    # Naive Bayes(Bernoulli)
    time_start = time.time()
    BNB = BernoulliNB()
    BNB.fit(x_train, y_train)
    time_end = time.time()
    t = time_end-time_start
    # save the trained naive bayes model
    joblib.dump(BNB, "BNB_model.m")
    # accuracy
    predictions = BNB.predict(x_test)
    print("The train of BNB classifier is done. It cost", t, "s\n Here is the report: ")
    print(classification_report(y_test, predictions))

    # Support Vector Machine
    time_start = time.time()
    SVM = SVC(kernel='linear')
    SVM.fit(x_train, y_train)
    time_end = time.time()
    t = time_end - time_start
    # save the trained svm model
    joblib.dump(SVM, "SVM_model.m")
    # accuracy
    predictions = SVM.predict(x_test)
    print("The train of SVM classifier is done. It cost", t, "s\n Here is the report: ")
    print(classification_report(y_test, predictions))

    # MLP
    time_start = time.time()
    MLP = MLPClassifier(activation='identity')
    MLP.fit(x_train, y_train)
    time_end = time.time()
    t = time_end - time_start
    # save the trained mlp model
    joblib.dump(MLP, "MLP_model.m")
    # accuracy
    predictions = MLP.predict(x_test)
    print("The train of MLP classifier is done. It cost", t, "s\n Here is the report: ")
    print(classification_report(y_test, predictions))

    # Decision Tree (DT)
    time_start = time.time()
    DT = DecisionTreeClassifier(criterion='entropy')
    DT.fit(x_train, y_train)
    y_predict = DT.predict(x_test)
    time_end = time.time()
    t = time_end - time_start
    # save the trained model
    joblib.dump(DT, "DT_model.m")
    # accuracy
    predictions = DT.predict(x_test)
    print("The train of DT classifier is done. It cost", t, "s\n Here is the report: ")
    print(classification_report(y_test, predictions))

    # k nearest neighbor (KNN)
    time_start = time.time()
    KNN = KNeighborsClassifier(n_neighbors=11, weights='distance', algorithm='kd_tree')
    KNN.fit(x_train, y_train)
    y_predict = KNN.predict(x_test)
    time_end = time.time()
    t = time_end - time_start
    # save the trained model
    joblib.dump(KNN, "KNN_model.m")
    # accuracy
    predictions = KNN.predict(x_test)
    print("The train of KNN classifier is done. It cost", t, "s\n Here is the report: ")
    print(classification_report(y_test, predictions))








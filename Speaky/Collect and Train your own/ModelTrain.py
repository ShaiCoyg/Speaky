import pandas as pd  # analyze data
import pickle  # save model
from sklearn.model_selection import train_test_split  # train model
from sklearn.pipeline import make_pipeline  # transform the data
from sklearn.preprocessing import StandardScaler  # standarize the data
# training algorithms
from sklearn.linear_model import LogisticRegression, RidgeClassifier
# training algorithms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score  # Accuracy measurement
from sklearn.metrics import classification_report  # report


df = pd.read_csv(
    r'C:\Users\WSM\Desktop\Speaky\Source code\Coordinates\Coordinates___MOOD E___0.14v.csv')
X = df.drop('CLASS', axis=1)  # our features - using our x,y,z coords
y = df['CLASS']  # our targets to identify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1988)  # split train test 70% 30%

pipelines = {  # unique pipe to each algo
    # the algo we use
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    # to compare and measure RFC preformance
    # 'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    # 'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    # 'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}  # send out split data to each pipe line, save all models in array
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

# for algo, model in fit_models.items():  # print mediaPipe accuracy score for each model
#     yhat = model.predict(X_test)
#     print('rf', accuracy_score(y_test, yhat))
#     print(classification_report(y_test, yhat))

# export the RFC model to desktop
with open(r'C:\Users\WSM\Desktop\Speaky\Source code\Models\Model___MOOD___0.17v.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)

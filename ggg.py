import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split #Para dividir o conjunto de teste e aleatoriezar.
from sklearn.metrics import accuracy_score #Para calcular a pontuação de acerto.
from sklearn.metrics import classification_report #Mostra a média das métricas de classificação

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
sns.set_style('whitegrid')


#Imprime os tipos de fantasma e a quantidade total de cada um. 
#O argumento deve ainda conter a feature TYPE.
def printTypes(train):
	print(train['type'].value_counts())

#Imprime os K primeiros exemplos.
def printExamples(train,k):
	print(train.head(k))

#imprime uma tabela descrevendo o argumento passado.
def printDescribe(arg):
	print(arg.describe(include='all'))



train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

sns.factorplot("type", col="color", col_wrap=4, data=train, kind="count", size=2.4, aspect=.8)

#O 'type' é o y que queremos aprender.
y=train['type']
indexes_test = test["id"]

#removendo colunas id, type e color do trainset para que elas não entrem no treinamento dos algoritmos de aprendizado.
train=train.drop(['id','type','color'],axis=1)

#fazendo o mesmo para o testset.
test=test.drop(['id','color'],axis=1)

#converte as categorias das tabelas em um conjunto de variaveis e seus valores.
train=pd.get_dummies(train)
test=pd.get_dummies(test)

#Quebra array de trainamento em 20% dos casos de forma aleatória 
#test_size representa a proporção do conjunto de dados a ser incluída na divisão de teste.
#Esta função embaralha os dados do conjunto de treino e validação por default. 
#O conjunto de traino será dividido em X_train e Y_train para o aprendizado dos algoritmos e X_test, e y_test para a validção.
#(Y_train para o treinamento de )
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=2)


print("K-Neighbors Classifier")
knn = KNeighborsClassifier()
params = {'n_neighbors' : [3, 5, 10, 15, 20], 
          'weights' : ['uniform', 'distance']}
knn_clf = GridSearchCV(knn, param_grid = params,  cv = 5)
knn_clf.fit(X_train, y_train)
print ('Best score : {}'.format(knn_clf.best_score_))
print ('Best parameters : {}'.format(knn_clf.best_params_))


y_pred = knn_clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("Accuracy Score is: ",accuracy_score(y_test, y_pred))
#y_pred = knn_clf.predict(test)





print("\n\n######################################################################")
print("Árvore de Decisão:\n")
dt = DecisionTreeClassifier(random_state = 0)

parameter_grid = {'max_depth': [1, 2, 3, 4, 5],
                  'max_features': [1, 2, 3, 4]}

dt_clf = GridSearchCV(dt, param_grid = parameter_grid)

dt_clf.fit(X_train, y_train)

print("Best Score: {}".format(dt_clf.best_score_))
print("Best params: {}".format(dt_clf.best_params_))

y_pred = dt_clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy Score is: ", accuracy_score(y_test, y_pred))
#y_pred = dt_clf.predict(test)


print("\n\n######################################################################")
print("Regressão Logistica:\n")
lr = LogisticRegression()

parameter_grid = {'C' : [0.005, 0.01, 0.05, 1, 2.5, 5, 10, 100, 1000,10000000]}

lr_clf = GridSearchCV(lr, param_grid=parameter_grid)
lr_clf.fit(X_train, y_train)
print('Best score: {}'.format(lr_clf.best_score_))
print('Best parameters: {}'.format(lr_clf.best_params_))

y_pred= lr_clf.predict(X_test) 

print(classification_report(y_pred,y_test))
print("Accuracy Score is: ",accuracy_score(y_test, y_pred))
#y_pred= lr_clf.predict(test) 


print("\n\n######################################################################")
print("SVM\n")

svc = SVC(probability = True, random_state = 0)
params = {'kernel' : ['linear', 'rbf'], 'C' : [1, 3, 5, 10], 'degree' : [3, 5, 10]}
svc_clf = GridSearchCV(svc, param_grid = params, cv = 5)
svc_clf.fit(X_train, y_train)

print ('Best score : {}'.format(svc_clf.best_score_))
print ('Best parameters : {}'.format(svc_clf.best_params_))


y_pred = svc_clf.predict(X_test)
print(classification_report(y_pred,y_test))
print("Accuracy Score is: ",accuracy_score(y_test, y_pred))




print("\n\n######################################################################")
print("Naive Bayes\n")
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)
print(classification_report(y_pred,y_test))
print("Accuracy Score is: ",accuracy_score(y_test,y_pred))
#y_pred = nb_clf.predict(test)


#Escolhendo a predição do SVM para predizer o teste, pois ele teve o melhor resultado.
y_pred = svc_clf.predict(test)
#Gerando CSV para a submissão.
submission = pd.DataFrame()
submission["id"] = indexes_test
submission["type"] = y_pred
submission.to_csv("submission.csv",index=False)


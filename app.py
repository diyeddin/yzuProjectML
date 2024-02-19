import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class App:
    def __init__(self):
        self.dataset_file = None
        self.dataset = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classifier_name = None
        self.selected_features = None
        self.n_features = None
        self.init_page()


    def run(self):
        # wait until user loads a dataset
        try:
            self.load_dataset()
        except:
            st.write("Please Select a File to Continue...")
            return
        self.preprocess()
        self.train()


    def init_page(self):
        st.title('YZUP Project - Diyeddin Almohamad')
        st.write('## Explore Different Classifiers')

        self.dataset_file = st.sidebar.file_uploader('Choose the Dataset', 'csv')
        self.classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Naive Bayes'))
        self.n_features = st.sidebar.slider("Number of Features (RFE)", 1, 30, value=10)


    def load_dataset(self):
        self.dataset = pd.read_csv(self.dataset_file)
        st.write('### First 10 rows of the dataset')
        st.write(self.dataset.head(10))
        st.write('### Columns')
        st.write(self.dataset.columns.values)

        # clean dataset
        self.dataset.dropna(axis=1, inplace=True)
        self.dataset.drop(['id'], axis=1, inplace=True)

        st.write('### Last 10 rows of the dataset')
        st.write(self.dataset.tail(10))

        self.dataset['diagnosis'] = self.dataset['diagnosis'].map({'M':1, 'B':0})
        self.X = self.dataset.drop(['diagnosis'], axis=1)
        self.y = self.dataset['diagnosis']

        st.pyplot(self.draw_corr().figure)
        plt.clf()
        st.pyplot(self.scatter_plt().figure)


    def draw_corr(self):
        st.write("### Correlation Matrix")

        sns.set_theme(style="white")
        corr = self.dataset.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(15, 15))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        corr_matrix_fig = sns.heatmap(corr, cmap=cmap, mask=mask, vmin=corr.values.min(), vmax=1, center=0, annot=True,
                                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 6.5})

        return corr_matrix_fig
        

    def scatter_plt(self):
        st.write("### Scatter Plot")
        values = list(self.dataset.columns.values)
        scatter_x = st.selectbox('Select X (default: radius_mean)', values, index=values.index('radius_mean'))
        scatter_y = st.selectbox('Select Y (default: texture_mean)', values, index=values.index('texture_mean'))

        plt.figure(figsize=(5,5))
        scatter = sns.scatterplot(self.dataset, x=scatter_x, y=scatter_y, hue='diagnosis', palette={1:'red', 0:'green'}, alpha=.5)
        plt.legend(labels=['Malignant', 'Benign'], loc='upper right')

        return scatter


    def preprocess(self):
        # split data into training and test datasets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)\

        # set up Recursive Feature Elimination to find the best features
        estimator = SVC(kernel="linear")
        selector = RFE(estimator, n_features_to_select=self.n_features, step=1)
        selector = selector.fit(self.X_train, self.y_train)
        self.selected_features = self.X.columns[selector.support_].values

        st.write("### Selected Features (After RFE)")
        st.write(self.selected_features)


    def model_selection(self):
        if self.classifier_name == 'KNN':
            self.model = Pipeline([
                ('scaler', MinMaxScaler()), 
                ('classifier', KNeighborsClassifier())
                ])
        elif self.classifier_name == 'SVM':
            self.model = Pipeline([
                ('scaler', StandardScaler()), 
                ('classifier', SVC(gamma='auto'))
                ])
        else:
            self.model = GaussianNB()


    def gridsearch(self):
        # model parameter grids
        models = {
            'KNN': (KNeighborsClassifier(), {'classifier__n_neighbors': (3, 5, 7, 9), 'classifier__leaf_size': (20,40,1), 'classifier__p': (1,2), 'classifier__weights': ('uniform', 'distance'), 'classifier__metric': ('minkowski', 'chebyshev'),}),
            'SVM': (SVC(), {'classifier__C': (0.1, 1, 10, 15), 'classifier__kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'classifier__coef0': (0.0, 10.0, 1.0), 'classifier__shrinking': (True, False)})
        }
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', models[self.classifier_name][0])
        ])
        param_grid = models[self.classifier_name][1]

        # apply gridsearch
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)

        return grid_search.best_params_


    def evaluate(self):
        y_pred = self.model.predict(self.X_test[self.selected_features])
        scores = {'Accuracy':accuracy_score(self.y_test, y_pred), 
                  'Precision':precision_score(self.y_test, y_pred, average='weighted'), 
                  'Recall':recall_score(self.y_test, y_pred, average='weighted'), 
                  'F1':f1_score(self.y_test, y_pred, average='weighted')}
        cm = confusion_matrix(self.y_test, y_pred)
        heatmap = sns.heatmap(cm, annot=True)

        return scores, heatmap


    def train(self):
        # select model
        self.model_selection()
        st.write(f'### Classifier: {self.classifier_name}')
        if self.classifier_name != 'Naive Bayes':
            best_params = self.gridsearch()
            self.model.set_params(**best_params)
            st.write('#### Best Parameters (GridSearchCV)')
            st.write(best_params)

        # train
        self.model.fit(self.X_train[self.selected_features], self.y_train)

        # show results
        plt.clf()
        scores, heatmap = self.evaluate()

        st.write('#### Test Results')
        st.write(f"Accuracy Score: :green[{scores['Accuracy']}]")
        st.write(f"Precision Score: :green[{scores['Precision']}]")
        st.write(f"Recall Score: :green[{scores['Recall']}]")
        st.write(f"F1 Score: :green[{scores['F1']}]")
        st.write('#### Confusion Matrix')
        st.pyplot(heatmap.figure)


if __name__ == "__main__":
    app = App()
    app.run()

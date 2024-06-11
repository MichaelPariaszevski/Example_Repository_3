import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import pair_confusion_matrix, roc_curve, precision_recall_fscore_support, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.gridspec as gridspec
import scikitplot as skplt
import os 

cwd=os.getcwd() 

print("----------------")
print(cwd)
print("----------------")

def main(): 
    @st.cache_data()
    def load_dataset_and_label_encode(data_path):   
        dataset=pd.read_csv(data_path)
        label=LabelEncoder()
        for column in dataset.columns: 
            dataset[column]=label.fit_transform(dataset[column]) 
        return dataset
    
    @st.cache_data()
    def load_dataset(data_path): 
        dataset=pd.read_csv(data_path) 
        return dataset
    
    data_labeled_2=load_dataset_and_label_encode("D:\Mushrooms_5.csv")

    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App") 
    # st.markdown("🍄 Are Your Mushrooms Edible or Poisonous? 🍄")

    data_unlabeled=load_dataset("D:\Mushrooms_5.csv")

    # st.markdown(data_unlabeled.head)
    st.sidebar.markdown("🍄 Are Your Mushrooms Edible or Poisonous? 🍄")
    # st.sidebar.markdown(data_labeled_2.head)

    if st.sidebar.checkbox("Show Raw Data", False): 
        st.subheader("Mushroom Dataset (Classification)") 
        st.write(data_labeled_2)

    @st.cache_resource() 
    def split(df): 
        y=df.iloc[:, 0]
        x=df.drop(columns=["type"])
        x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=1)
        return x_train, x_test, y_train, y_test
    
    x_train, x_test, y_train, y_test=split(data_labeled_2)
    class_names=["edible", "poisonous"]

    # @st.cache_resource()
    def plot_metrics(metrics_list): 
        # fig, axes=plt.subplots(1, 1, sharex=False)
        if "Confusion Matrix" in metrics_list: 
            fig1, axes1=plt.subplots(1, 1, sharex=False)
            st.subheader("Confusion Matrix") 
            axes1.set_title("Confusion Matrix")
            cm=confusion_matrix(y_test, y_pred)
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(include_values=True, cmap="Blues", ax=axes1)
            # fig=plt.figure(figsize=(6, 4))
            # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False) 
            # plt.xlabel("Predicted Labels") 
            # plt.ylabel("True Labels") 
            # plt.title("Confusion Matrix")
            st.pyplot(fig1) 
        # fig, axes=plt.subplots(1, 1, sharex=False)
        if "ROC Curve" in metrics_list: 
            fig2, axes2=plt.subplots(1, 1, sharex=False)
            st.subheader("ROC Curve") 
            fpr, tpr, _ =roc_curve(y_test, y_pred)
            axes2.set_xlabel("FPR") 
            axes2.set_ylabel("TPR")
            axes2.plot(fpr, tpr)
            # plt.plot(tpr, fpr)
            # fig_2=plt.figure(figsize=(6, 4))
            # fig_2=skplt.metrics.plot_roc_curve(y_test, y_pred)
            st.pyplot(fig2)
        # fig, axes=plt.subplots(1, 1, sharex=False)
        if "Precision-Recall Curve" in metrics_list: 
            fig3, axes3=plt.subplots(1, 1, sharex=False)
            st.subheader("Precision-Recall Curve")
            # ps=precision_score(y_test, y_pred) 
            # rs=recall_score(y_test, y_pred) 
            precisions, recalls, thresholds=precision_recall_curve(y_test, y_pred)
            axes3.set_xlabel("Recalls") 
            axes3.set_ylabel("Precisions")
            axes3.plot(recalls, precisions)
            st.pyplot(fig3)

    st.sidebar.subheader("Choose Classifier") 
    classifier=st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier=="Support Vector Machine (SVM)": 
        st.sidebar.subheader("Model Hyperparameters") 
        C=st.sidebar.number_input("C (Regularization Parameter); Range: 0.01-10", 0.01, 10.0, step=0.01, key="C")
        kernel=st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel") 
        gamma=st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")
        metrics=st.sidebar.multiselect("Metrics to Plot", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if st.sidebar.button("Classify", key="classify"): 
            st.subheader("Support Vector Machine (SVM) Results") 
            model=SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: "+str(round(accuracy, 2))) 
            st.write("Precision: "+str(round(precision_score(y_test, y_pred, labels=class_names), 2))) 
            st.write("Recall Score: "+str(round(recall_score(y_test, y_pred, labels=class_names), 2)))
            plot_metrics(metrics)

    if classifier=="Logistic Regression": 
        st.sidebar.subheader("Model Hyperparameters") 
        C=st.sidebar.number_input("C (Regularization Parameter); Range: 0.01-10", 0.01, 10.0, step=0.01, key="C_LR")
        max_iter=st.sidebar.slider("Maximum Number of Training Iterations; Range 100-500", 100, 500, key="max_iter")
        metrics=st.sidebar.multiselect("Metrics to Plot", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if st.sidebar.button("Classify", key="classify"): 
            st.subheader("Logistic Regression Results") 
            model=LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: "+str(round(accuracy, 2))) 
            st.write("Precision: "+str(round(precision_score(y_test, y_pred, labels=class_names), 2))) 
            st.write("Recall Score: "+str(round(recall_score(y_test, y_pred, labels=class_names), 2)))
            plot_metrics(metrics)
    
    if classifier=="Random Forest": 
        st.sidebar.subheader("Model Hyperparameters") 
        n_estimators=st.sidebar.number_input("The Number of Trees in the Forest", 100, 5000, step=10, key="n_estimators")
        max_depth=st.sidebar.number_input("The Maximum Depth of the Tree", 1, 20, step=1, key="max_depth")
        bootstrap=st.sidebar.radio("Bootstrap Samples when Building Trees", (True, False), key="bootstrap")
        metrics=st.sidebar.multiselect("Metrics to Plot", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if st.sidebar.button("Classify", key="classify"): 
            st.subheader("Random Forest Results") 
            model=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: "+str(round(accuracy, 2))) 
            st.write("Precision: "+str(round(precision_score(y_test, y_pred, labels=class_names), 2))) 
            st.write("Recall Score: "+str(round(recall_score(y_test, y_pred, labels=class_names), 2)))
            plot_metrics(metrics)


    print(data_labeled_2.head)
    print(x_train)
    print(y_train)

main()

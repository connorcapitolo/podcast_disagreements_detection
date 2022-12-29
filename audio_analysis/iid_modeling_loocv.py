import os
import pandas as pd
import time
# import tensorflow as tf
# print(tf.__version__)
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score,roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def iid_modeling_loocv(loaded_parquet_df: dict, scikit_model) -> dict:
    '''Performing a leave-one-out with each of the podcasts, meaning each podcast episode will have the chance to be the test set
    
    Args:
        parquet_list: a dictionary where the key is the name of the podcast episode parquet file, and the values are the dwt dataframe
        scikit_model: the particular scikit-learn model to be used (e.g. LogisticRegression, RandomForest, etc.)
    '''
    parquet_list = list(loaded_parquet_df.keys())


    start_all_models = time.time()

    model_metrics = {}


    # looping through each time to get the podcast saved to be the test set
    for test_podcast in parquet_list:

        start_one_model = time.time()

        train_dfs = []

        # looping through each podcast to either add them to train or test set
        for key, val in loaded_parquet_df.items():
            if key != test_podcast:
                train_dfs.append(val)

        print(f'{test_podcast} will be the test podcast')

        # get the training set
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
        train_set = pd.concat(train_dfs).dropna()
        print(f'Training set size: {train_set.shape}')

        # get the test set
        test_set = loaded_parquet_df[test_podcast].dropna()
        print(f'Test set size: {test_set.shape}')


        X_train, y_train = train_set.drop('y', axis=1), train_set['y']
        X_test, y_test = test_set.drop('y', axis=1), test_set['y']


        # logreg = LogisticRegression(penalty = 'l2', class_weight ='balanced', max_iter = 1e10)
        scikit_model.fit(X_train, y_train)

        y_pred = scikit_model.predict(X_test)
        y_pred_prob = scikit_model.predict_proba(X_test)

        model_metrics[test_podcast] = {}

        model_metrics[test_podcast]['model'] = scikit_model
        model_metrics[test_podcast]['y_pred'] = y_pred
        model_metrics[test_podcast]['classification_report'] = classification_report(y_test, y_pred)
        model_metrics[test_podcast]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        model_metrics[test_podcast]['y_pred_prob'] = scikit_model.predict_proba(X_test)
        model_metrics[test_podcast]['y_pred_prob_log'] = scikit_model.predict_log_proba(X_test)
        model_metrics[test_podcast]['accuracy'] = scikit_model.score(X_test, y_test)
        model_metrics[test_podcast]['precision'] = precision_score(y_test, y_pred)
        model_metrics[test_podcast]['recall'] = recall_score(y_test, y_pred)
        model_metrics[test_podcast]['f1'] = f1_score(y_test, y_pred)
        model_metrics[test_podcast]['logit_roc_auc'] = roc_auc_score(y_test, y_pred)
        model_metrics[test_podcast]['fpr_tpr_thresholds'] = roc_curve(y_test, y_pred_prob[:,1])

        pred_result_df = pd.DataFrame(y_pred_prob)
        pred_result_df['y_pred'] = y_pred
        pred_result_df['y_pred_true_label'] = y_test
        model_metrics[test_podcast]['false_negatives'] = pred_result_df[(pred_result_df['y_pred'] == 0) & (pred_result_df['y_pred_true_label'] == 1)]
        model_metrics[test_podcast]['false_positives'] = pred_result_df[(pred_result_df['y_pred'] == 1) & (pred_result_df['y_pred_true_label'] == 0)]
        model_metrics[test_podcast]['true_positives'] = pred_result_df[(pred_result_df['y_pred'] == 1) & (pred_result_df['y_pred_true_label'] == 1)]

        accuracy = model_metrics[test_podcast]['accuracy']
        precision = model_metrics[test_podcast]['precision']
        recall = model_metrics[test_podcast]['recall']
        f1 = model_metrics[test_podcast]['f1']
        print(f'{test_podcast} scores: accuracy = {accuracy:.4f}, precision = {precision:.4f}, recall = {recall:.4f}, f1_score = {f1:.4f}')
        tn, fp, fn, tp = model_metrics[test_podcast]['confusion_matrix'].ravel()
        print(f'TN={tn}   FP={fp}   FN={fn}   TP={tp}')

        print(f'{test_podcast} completed in {(time.time() - start_one_model):.2f} seconds\n')

    #     if test_podcast == '0TkGYYIPwRqx8xzP0XGvRG.parquet':
    #         break

    print(f'Finished LOOCV in {((time.time() - start_all_models) / 60):.2f} minutes')
    
    tot_acc = []
    tot_prec = []
    tot_rec = []
    tot_f1 = []
    for test_podcast in model_metrics.values():
        tot_acc.append(test_podcast['accuracy'])
        tot_prec.append(test_podcast['precision'])
        tot_rec.append(test_podcast['recall'])
        tot_f1.append(test_podcast['f1'])
        
    avg_acc = np.average(np.array(tot_acc))
    avg_prec = np.average(np.array(tot_prec))
    avg_rec = np.average(np.array(tot_rec))
    avg_f1 = np.average(np.array(tot_f1))
    
    print(f'Average scores: accuracy = {avg_acc:.4f}, precision = {avg_prec:.4f}, recall = {avg_rec:.4f}, f1_score = {avg_f1:.4f}')
   
        

    return model_metrics
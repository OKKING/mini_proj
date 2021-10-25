import pandas as pd
import numpy as np
import os
import warnings
import pickle, time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from sklearn.metrics import log_loss, recall_score, precision_score,roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve,brier_score_loss





def wide_view(df):
    from IPython.core.display import HTML
    display(HTML(df.to_html()))
    
def display_sides(*args):
    from IPython.core.display import display_html
    html_str = ''
    for arg in args:
        html_str += arg.to_html()
    display_html(html_str.replace('table','table style = display:inline'),raw = True)
    
    
def plot_bar(df,feature,target = 'Call_Flag',show_counts = False,show_annotate = True,rotation = 0,title_extra =''):
    fig,ax = plt.subplots(figsize = (15,7))
    for x,h in zip([x - 0.4 for x in list(np.arange(1,df.shape[0]+1))],df[(target, 'mean')].values):
        ax.bar(x,h,color = 'salmon',alpha = 0.45,width = 0.4,align = 'edge',edgecolor = 'black')
        if show_annotate:
            ax.annotate(str(round(100*h,2))+'%',(x+0.15,h),
                       textcoords="offset points",
                       xytext =(0,9),ha = 'left',va = 'center',size = 12)
    ax.set_xticks(list(np.arange(1,df.shape[0]+1)))
    ax.set_xticklabels(list(df.index),size =12,rotation = rotation)
    ax.set_title('Barplot: {} Rate vs {} '.format(target,feature)+title_extra,size = 20)
#     ax.set_ylim([0,0.66])
    ax.set_ylabel('{} Rate'.format(target), size = 16)
    ax.set_xlabel('%s' % feature, size = 16)
    if show_counts:
        ax2 = ax.twinx()
        for x,z in zip([x for x in list(np.arange(1,df.shape[0]+1))],df[(target, 'size')].values/10000):
            ax2.bar(x,z,color = 'lightblue',alpha = 0.25,width = 0.4,align = 'edge',edgecolor = 'black')
            if show_annotate:
                ax2.annotate(str(z),(x+0.15,z),
                           textcoords="offset points",
                           xytext =(0,9),ha = 'left',va = 'center',size = 12)
    #     ax2.set_ylim([0,350])
        ax2.set_ylabel('Number of counts in (10 thousands)',size = 16)
    plt.show()
    return fig,ax

def plot_box(df,colors = ['lightblue', 'salmon'],rot = 0):
    fig,ax = plt.subplots(figsize =(12,6))
    bp = ax.boxplot([df[col] for col in df.columns],labels =df.columns.tolist(),patch_artist = True,widths =0.6,notch ='True')
    ax.set_xticklabels(df.columns.tolist(),size =18,rotation = rot)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for whisker in bp['whiskers']:
        whisker.set(color ='lightgrey',
                    linewidth = 2,
                    linestyle ="-")
    for cap in bp['caps']:
        cap.set(color ='lightgrey',
                linewidth = 2)
    for median in bp['medians']:
        median.set(color ='lightgrey',
                   linewidth = 2)
    plt.show()
    return fig

def plot_heatmap(corr_mat,cmap="Blues"):
    fig,ax = plt.subplots(figsize = (12,8.9))
    mask = np.zeros_like(corr_mat)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_matrix, mask = 1-mask,annot=True,linewidths=.5,cmap=cmap,ax = ax)
    plt.yticks(rotation=0)
    plt.xticks(rotation=25)
    plt.show()
    return fig


def create_accuracy_plots(df_s1,df_s2,show_log_odd = True,num_bins = 10,labels = ['catboost','xgboost'],xlim =[0,1]):

    nrows =1
    ncols =2
    if df_s1.shape[0]>0:
            df_s1['grp'] = pd.qcut(df_s1['score'],q=num_bins,duplicates = 'drop')
            df_scores_grp1 = df_s1.groupby('grp')[['y','score']].agg([np.mean,np.size])
            df_scores_grp1['logodd'] = [np.log(x/(1-x)) for x in df_scores_grp1[('y','mean')]]
            df_scores_grp1['logoddp'] = [np.log(x/(1-x)) for x in df_scores_grp1[('score','mean')]]
    if df_s2.shape[0]>0:
            df_s2['grp'] = pd.qcut(df_s2['score'],q=num_bins,duplicates = 'drop')
            df_scores_grp2 = df_s2.groupby('grp')[['y','score']].agg([np.mean,np.size])
            df_scores_grp2['logodd'] = [np.log(x/(1-x)) for x in df_scores_grp2[('y','mean')]]
            df_scores_grp2['logoddp'] = [np.log(x/(1-x)) for x in df_scores_grp2[('score','mean')]]
    
    fig, ax = plt.subplots(nrows,ncols,figsize = (8*ncols,5*nrows))
    if df_s1.shape[0]>0:
        if df_s1.shape[0]>0:
            g_y  = df_scores_grp1[('y','mean')]
            g_x  = df_scores_grp1[('score','mean')]
            ax[0].scatter(g_x,g_y, marker='o', linewidth=1,color = 'lightgreen'
                       ,alpha = 0.55,label = '{}'.format(labels[0]),s = df_scores_grp1[('score','size')]/10)

        if df_s2.shape[0]>0:
            g2_y  = df_scores_grp2[('y','mean')]
            g2_x  = df_scores_grp2[('score','mean')]
            ax[0].scatter(g2_x,g2_y, marker='o', linewidth=1,color = 'salmon'
                   ,alpha = 0.55,label = '{}'.format(labels[1]),s = df_scores_grp2[('score','size')]/10) 
        
        ax[0].plot([0,1],[0,1],linestyle = '--',color = 'grey',linewidth =5,label = 'ideal')
        ax[0].set_title('{} vs {}'.format(labels[0],labels[1]),size = 24)
        ax[0].set_xlabel('Predicted Good Rate %',size = 16)
        ax[0].set_ylabel('Actual Good Rate %',size = 16)
        ax[0].set_xlim(xlim)
        ax[0].set_ylim(xlim)
        ax[0].legend(loc = 'upper left')
    
    if show_log_odd:
        ax[1].plot([-10,2],[-10,2],color = 'grey',linewidth = 3,label = 'ideal',alpha =0.55,linestyle ='--')
        if df_s1.shape[0]>0:
            ax[1].plot(df_scores_grp1['logoddp'],df_scores_grp1['logodd'], marker='D', linewidth=2, label='{}'.format(labels[0]),color = 'lightgreen',alpha = 0.85)
        if df_s2.shape[0]>0:    
            ax[1].plot(df_scores_grp2['logoddp'],df_scores_grp2['logodd'], marker='D', linewidth=2, label='{}'.format(labels[1]),color = 'salmon',alpha = 0.85)
        ax[1].set_title('{} vs {}'.format(labels[0],labels[1]),size = 24)
        ax[1].set_xlabel('Predicted log odd',size = 18)
        ax[1].set_ylabel('True log odd',size = 18)
        ax[1].legend(loc = 'upper left')

    plt.show() 
    return fig,ax




def roc_func(ground_truth, predictions):
    """Return Roc index. Takes input like y_test, model.predict_proba(X_test)[:, 1]"""
    return roc_auc_score(ground_truth, predictions)


def ks_stat_func(ground_truth, predictions):
    """Return KS stat. Takes input like y_test, model.predict_proba(X_test)[:, 1]"""
    thresholds, pct1, pct2, ks_statistic, max_distance_at, classes = \
        skplt.helpers.binary_ks_curve(np.array(ground_truth), np.array(predictions).ravel())
    return ks_statistic, max_distance_at


def prob_thresh(probas, thresh):
    """Convert probability lists to 0s and 1s based on given threshold value. Strict threshold."""
    return [int(i) for i in (probas > thresh)]

def recall_precision_stat_func(ground_truth,probas,thresh):
    """The thresh is set when KS statistic is at the critical threshold value."""
    return (recall_score(ground_truth,prob_thresh(probas,thresh)),
            precision_score(ground_truth,prob_thresh(probas,thresh)))


def model_metrics_df(y_test, y_preda, y_train, y_preda_train, name = 'index',round_decimals=3):
    
    ks_test,ks_thresh    = ks_stat_func(y_test, y_preda)
    ks_train,ks_thresh_train   = ks_stat_func(y_train, y_preda_train)
    roc_test             = roc_func(y_test, y_preda)
    roc_train            = roc_func(y_train, y_preda_train)
    logloss_test         = log_loss(y_test, y_preda)
    logloss_train        = log_loss(y_train, y_preda_train)    
    brierloss_test       = brier_score_loss(y_test, y_preda)
    brierloss_train      = brier_score_loss(y_train, y_preda_train)        
    recall_test,precision_test    = recall_precision_stat_func(y_test, y_preda, ks_thresh)
    recall_train,precision_train  = recall_precision_stat_func(y_train, y_preda_train, ks_thresh)
    
    df_metrics = pd.DataFrame({f'KS(t={round(ks_thresh, round_decimals)})': [ks_test, ks_train, ks_train - ks_test],
                               'Roc': [roc_test, roc_train, roc_train - roc_test],
                               'Log Loss': [logloss_test, logloss_train, logloss_test - logloss_train],
                               'Brier Loss': [brierloss_test, brierloss_train, brierloss_test - brierloss_train],
                               f'Recall(t={round(ks_thresh, round_decimals)})': [recall_test, recall_train, recall_train - recall_test],
                               f'Precision(t={round(ks_thresh, round_decimals)})': [precision_test, precision_train, precision_train - precision_test]
                               },
                              index=['Test Set', 'Training Set', 'Difference']).round(decimals=round_decimals)
    df_metrics.index.name = name
    return df_metrics
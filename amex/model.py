import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset, DataLoader,RandomSampler


from contextlib import contextmanager
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
from sklearn.metrics import roc_auc_score,mean_squared_error,average_precision_score,log_loss

today = datetime.now().today()
current_date = str(today.date())
# current_time = str(today.time().strftime('%H:%M:%S'))

class LGBM():
    def __init__(self, train, test, param):
        self.train = train
        self.test = test
        self.param = param
        # self.seed = seed
        
        
    def train_val_split(self):
        seed = self.param['seed']
        train = self.train
        target = self.param['target_var']
        split_rate = self.param['split_rate']
        X_train, X_val, y_train, y_val = train_test_split(train.drop(target, axis = 1), train[[target]], test_size=split_rate, random_state=seed)
        
        return X_train, X_val, y_train, y_val
        
    
    def fit(self):
        train = self.train
        folds = self.param['folds']
        params = self.param['params']
        rounds = self.param['rounds']
        verbose = self.param['verbose']
        features = self.param['features']
        label_name = self.param['label_name']
        early_stop = self.param['early_stop']
        id_name = self.param['id_name']
        seed = self.param['seed']
        oof = train[[id_name]]
        oof[label_name] = 0
        feature_importance = []
        
        # StratifiedKFold: val_size: 1/n_splits
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        skf_split = skf.split(train, train[[label_name]])
            
        for fold, (train_idx, val_idx) in enumerate(skf_split):
            evals_result_dic = {}
            # train_cids = train.loc[train_idx,id_name].values

            train_data = lgb.Dataset(train.loc[train_idx,features], label=train.loc[train_idx,label_name])
            val_data = lgb.Dataset(train.loc[val_idx,features], label=train.loc[val_idx,label_name])
            model = lgb.train(params,
                train_set = train_data,
                num_boost_round = rounds,
                valid_sets = [train_data, val_data],
                evals_result = evals_result_dic,
                early_stopping_rounds = early_stop,
                verbose_eval = verbose
            )
            model.save_model(f'{current_date}fold-{fold}.ckpt')
            valid_preds = model.predict(train.loc[val_idx,features], num_iteration=model.best_iteration)
            oof.loc[val_idx,label_name] = valid_preds

            importance_gain = model.feature_importance(importance_type='gain')
            importance_split = model.feature_importance(importance_type='split')
            feature_name = model.feature_name()
            new_feature_importance_df = pd.DataFrame({'feature_name':feature_name,
                                                      'importance_gain':importance_gain,
                                                      'importance_split':importance_split})
            feature_importance.append(new_feature_importance_df)

        feature_importance_df = pd.concat(feature_importance)
        feature_importance_df = feature_importance_df.groupby(['feature_name']).mean().reset_index()
        feature_importance_df = feature_importance_df.sort_values(by=['importance_gain'],ascending=False)
        feature_importance_df.to_csv(f'{current_date}-feature_importance.csv',index=False)

        oof.to_csv(f'{current_date}-oof.csv',index=False)
        
        
    def output_predict(self):
        test = self.test
        folds = self.param['folds']
        features = self.param['features']
        sub = None
        
        for fold in range(folds):
            model = lgb.Booster(model_file=f'{current_date}fold-{fold}.ckpt')
            test_preds = model.predict(test[features], num_iteration=model.best_iteration)
            sub['prediction'] += (test_preds / folds)
        # sub[[id_name,'prediction']].to_csv(output_path + '/submission.csv.zip', compression='zip',index=False)
        return sub
    
    
    
    
class GNN():
    def __init__(self, train, test, param):
        self.train = train
        self.test = test
        self.param = param
        # self.seed = seed
        
        
    def fit(self):
        train = self.train
        feature_name = self.param['feature_name']
        obj_max = self.param['obj_max']
        epochs = self.param['epochs']
        smoothing = self.param['smoothing']
        patience = self.param['patience']
        lr = self.param['lr']
        batch_size = self.param['batch_size']
        folds = self.param['folds']
        seed = self.param['seed']
        id_name = self.param['id_name']
        label_name = self.param['label_name']
        oof = train[[id_name]]
        oof[label_name] = 0
        
        skf = StratifiedKFold(n_splits = folds, shuffle=True, random_state=seed)
        for fold, (trn_index, val_index) in enumerate(skf.split(train_y,train_y[label_name])):
            
            
            
            
            return 
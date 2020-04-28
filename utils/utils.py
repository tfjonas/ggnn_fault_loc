import sys
from tqdm import tqdm
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from utils import errors  


def train_model(model, batch_size, epochs, steps_per_epoch,
                TrainDataset, TestDataset, UnrelatedDataset,
                adj_matrices, dist_matrices):
    
    
    EPOCHS = epochs  
    BATCH_SIZE = batch_size
    VAL_BATCH_SIZE = min(len(TestDataset),50)
    UNR_BATCH_SIZE = min(len(UnrelatedDataset),50)

    STEPS_PER_EPOCH = steps_per_epoch

    # Split train dataset in batches
    train_data = DataLoader(TrainDataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data = DataLoader(TestDataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
    unr_data = DataLoader(UnrelatedDataset, batch_size=UNR_BATCH_SIZE, shuffle=False)
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    
    # Initial evaluations
    # =========================================================================
    
    loss_ls = []
    BBE_min_ls, BBE_mean_ls, BBE_max_ls = [], [], []
    WBE_min_ls, WBE_mean_ls, WBE_max_ls = [], [], []
    uBBE_min_ls, uBBE_mean_ls, uBBE_max_ls = [], [], []
    uWBE_min_ls, uWBE_mean_ls, uWBE_max_ls = [], [], []

    # Test dataset
    # -------------------------------------------------------------------------
    model.eval()
    BBE_min, BBE_mean, BBE_max = 1e1000, 0, 0
    WBE_min, WBE_mean, WBE_max = 1e1000, 0, 0
    for batch in val_data:
        X, T, idx = batch
        A = adj_matrices[idx]
        D = dist_matrices[idx]
        Y = model(X.cuda(), A.cuda())
            
        _min_, _mean_, _max_ = errors.biggest_bet_error(Y.cpu(),T.cpu(),D)
        BBE_min = min(BBE_min, _min_)
        if BBE_mean == 0:
            BBE_mean = _mean_
        else:
            BBE_mean = (BBE_mean + _mean_)/2
        BBE_max = max(BBE_max, _max_)

        _min_, _mean_, _max_ = errors.weighted_bets_error(Y.cpu(),T.cpu(),D)
        WBE_min = min(WBE_min, _min_)
        if WBE_mean == 0:
            WBE_mean = _mean_
        else:
            WBE_mean = (WBE_mean + _mean_)/2
        WBE_max = max(WBE_max, _max_)

        del X, Y, A, D 
    
    BBE_min_ls.append(BBE_min)
    BBE_mean_ls.append(BBE_mean)
    BBE_max_ls.append(BBE_max)
    WBE_min_ls.append(WBE_min)
    WBE_mean_ls.append(WBE_mean)
    WBE_max_ls.append(WBE_max)


    # Unrelated dataset
    # -------------------------------------------------------------------------
    model.eval()
    BBE_min, BBE_mean, BBE_max = 1e1000, 0, 0
    WBE_min, WBE_mean, WBE_max = 1e1000, 0, 0
    for batch in unr_data:
        X, T, idx = batch
        A = adj_matrices[idx]
        D = dist_matrices[idx]
        Y = model(X.cuda(), A.cuda())
            
        _min_, _mean_, _max_ = errors.biggest_bet_error(Y.cpu(),T.cpu(),D)
        BBE_min = min(BBE_min, _min_)
        if BBE_mean == 0:
            BBE_mean = _mean_
        else:
            BBE_mean = (BBE_mean + _mean_)/2
        BBE_max = max(BBE_max, _max_)

        _min_, _mean_, _max_ = errors.weighted_bets_error(Y.cpu(),T.cpu(),D)
        WBE_min = min(WBE_min, _min_)
        if WBE_mean == 0:
            WBE_mean = _mean_
        else:
            WBE_mean = (WBE_mean + _mean_)/2
        WBE_max = max(WBE_max, _max_)

        del X, Y, A, D 
    
    uBBE_min_ls.append(BBE_min)
    uBBE_mean_ls.append(BBE_mean)
    uBBE_max_ls.append(BBE_max)
    uWBE_min_ls.append(WBE_min)
    uWBE_mean_ls.append(WBE_mean)
    uWBE_max_ls.append(WBE_max)
    
    # Print results
    # -------------------------------------------------------------------------

    print('\t      val_biggest_bet_error: %.4f / %.4f / %.4f (Min/Avg/Max) \n\t      val_weighted_bet_error: %.4f / %.4f / %.4f (Min/Avg/Max)' 
        % (BBE_min_ls[-1], BBE_mean_ls[-1], BBE_max_ls[-1], WBE_min_ls[-1], WBE_mean_ls[-1], WBE_max_ls[-1]))    
    print('-----------------------------------------------------------------------------------') 
    print('\t      unr_biggest_bet_error: %.4f / %.4f / %.4f (Min/Avg/Max) \n\t      unr_weighted_bet_error: %.4f / %.4f / %.4f (Min/Avg/Max)' 
        % (uBBE_min_ls[-1], uBBE_mean_ls[-1], uBBE_max_ls[-1], uWBE_min_ls[-1], uWBE_mean_ls[-1], uWBE_max_ls[-1]))    
    print('-----------------------------------------------------------------------------------') 

    # =========================================================================
    
    
    # Training 
    # =========================================================================
    
    for epoch in range(EPOCHS):

        # Break condition
        #if (uWBE_mean_ls[-1] < 0.2):
        #    break
        epoch_loss = []
        with tqdm(total=STEPS_PER_EPOCH, file=sys.stdout) as pbar:
            model.train()
            for step in range(STEPS_PER_EPOCH):
                # Get batch data
                X, T, idx = next(iter(train_data))
                A = adj_matrices[idx]
                # Forward pass: Compute predicted y by passing x to the model
                Y = model(X.cuda(), A.cuda())
                # Compute loss
                loss = criterion(Y, T.cuda())
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description('...step %d/%d - loss: %.4f' % ((step+1), STEPS_PER_EPOCH, loss.item()))
                pbar.update(1)

                epoch_loss.append(loss.item())
                loss_ls.append(loss.item())
                
                
        # =====================================================================
        
        # New evaluations
        # =====================================================================        

        # Test dataset
        # ---------------------------------------------------------------------
        model.eval()
        BBE_min, BBE_mean, BBE_max = 1e1000, 0, 0
        WBE_min, WBE_mean, WBE_max = 1e1000, 0, 0
        for batch in val_data:
            X, T, idx = batch
            A = adj_matrices[idx]
            D = dist_matrices[idx]
            Y = model(X.cuda(), A.cuda())
                
            _min_, _mean_, _max_ = errors.biggest_bet_error(Y.cpu(),T.cpu(),D)
            BBE_min = min(BBE_min, _min_)
            if BBE_mean == 0:
                BBE_mean = _mean_
            else:
                BBE_mean = (BBE_mean + _mean_)/2
            BBE_max = max(BBE_max, _max_)

            _min_, _mean_, _max_ = errors.weighted_bets_error(Y.cpu(),T.cpu(),D)
            WBE_min = min(WBE_min, _min_)
            if WBE_mean == 0:
                WBE_mean = _mean_
            else:
                WBE_mean = (WBE_mean + _mean_)/2
            WBE_max = max(WBE_max, _max_)

            del X, Y, A, D 
        
        BBE_min_ls.append(BBE_min)
        BBE_mean_ls.append(BBE_mean)
        BBE_max_ls.append(BBE_max)
        WBE_min_ls.append(WBE_min)
        WBE_mean_ls.append(WBE_mean)
        WBE_max_ls.append(WBE_max)


        # Unrelated dataset
        # ---------------------------------------------------------------------
        model.eval()
        BBE_min, BBE_mean, BBE_max = 1e1000, 0, 0
        WBE_min, WBE_mean, WBE_max = 1e1000, 0, 0
        for batch in unr_data:
            X, T, idx = batch
            A = adj_matrices[idx]
            D = dist_matrices[idx]
            Y = model(X.cuda(), A.cuda())
                
            _min_, _mean_, _max_ = errors.biggest_bet_error(Y.cpu(),T.cpu(),D)
            BBE_min = min(BBE_min, _min_)
            if BBE_mean == 0:
                BBE_mean = _mean_
            else:
                BBE_mean = (BBE_mean + _mean_)/2
            BBE_max = max(BBE_max, _max_)

            _min_, _mean_, _max_ = errors.weighted_bets_error(Y.cpu(),T.cpu(),D)
            WBE_min = min(WBE_min, _min_)
            if WBE_mean == 0:
                WBE_mean = _mean_
            else:
                WBE_mean = (WBE_mean + _mean_)/2
            WBE_max = max(WBE_max, _max_)

            del X, Y, A, D 
        
        uBBE_min_ls.append(BBE_min)
        uBBE_mean_ls.append(BBE_mean)
        uBBE_max_ls.append(BBE_max)
        uWBE_min_ls.append(WBE_min)
        uWBE_mean_ls.append(WBE_mean)
        uWBE_max_ls.append(WBE_max)

        # Print results
        # ---------------------------------------------------------------------
        print('Epoch %d/%d - train_loss: %.4f / %.4f / %.4f (Min/Avg/Max)'  # - val_loss: %.4f'
            % (epoch+1, EPOCHS, min(epoch_loss), sum(epoch_loss)/len(epoch_loss), max(epoch_loss)))  #, val_loss))
        print('\t      ---------------------------------------------------------------------' )
        print('\t      val_biggest_bet_error: %.4f / %.4f / %.4f (Min/Avg/Max) \n\t      val_weighted_bet_error: %.4f / %.4f / %.4f (Min/Avg/Max)' 
            % (BBE_min_ls[-1], BBE_mean_ls[-1], BBE_max_ls[-1], WBE_min_ls[-1], WBE_mean_ls[-1], WBE_max_ls[-1]))    
        print('\t      ---------------------------------------------------------------------' )
        print('\t      unr_biggest_bet_error: %.4f / %.4f / %.4f (Min/Avg/Max) \n\t      unr_weighted_bet_error: %.4f / %.4f / %.4f (Min/Avg/Max)' 
            % (uBBE_min_ls[-1], uBBE_mean_ls[-1], uBBE_max_ls[-1], uWBE_min_ls[-1], uWBE_mean_ls[-1], uWBE_max_ls[-1]))
        print('-----------------------------------------------------------------------------------') 


    # =========================================================================


    statistics = {'loss_ls': loss_ls,
                  'BBE_ls': [BBE_min_ls, BBE_mean_ls, BBE_max_ls],
                  'WBE_ls': [WBE_min_ls, WBE_mean_ls, WBE_max_ls],
                  'uBBE_ls': [uBBE_min_ls, uBBE_mean_ls, uBBE_max_ls],
                  'uWBE_ls': [uWBE_min_ls, uWBE_mean_ls, uWBE_max_ls]}

    return model, optimizer, statistics

import torch 
from torch.distributions.bernoulli import Bernoulli
import numpy as np


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy

import heuristic_selection as hs

class AdvTraining(nn.Module):
    def __init__(self, args, model_name, model, trigger_generator, device):
        super(AdvTraining, self).__init__()
        self.args = args
        self.model_name = model_name
        self.trigger_generator = trigger_generator
        self.model = model
        self.device = device

    def inject_trigger(self, x, edge_index, edge_weight, idx_inject):
        x, edge_index, edge_weight = x.clone().detach(), edge_index.clone().detach(), edge_weight.clone().detach()
        x, edge_index, edge_weight = self.trigger_generator.inject_trigger(idx_inject,x, edge_index, edge_weight,self.device)
        x, edge_index, edge_weight = x.clone().detach(), edge_index.clone().detach(), edge_weight.clone().detach()
        return x, edge_index, edge_weight
    
    def adversarial_attack(self, x, edge_index, edge_weight, idx_2b_attach):
        # randomly select node for poisoning
        self.trigger_generator.trojan.eval()
        num_inject = len(idx_2b_attach)
        idx_attach = hs.obtain_attach_nodes(self.args,idx_2b_attach,size = num_inject)

        adv_x, adv_edge_index, adv_edge_weight = self.inject_trigger(x, edge_index, edge_weight, idx_attach)
        return adv_x, adv_edge_index, adv_edge_weight

    def forward(self, features, edge_index, edge_weight):
        if(self.model_name == 'GCN'):
            output, x = self.model.forward(features, edge_index, edge_weight)
        elif(self.model_name in ['GAT','GraphSAGE','GIN']):
            output= self.model.forward(features, edge_index, edge_weight)
        else:
            raise NotImplementedError("No implemented model")
        return output

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, finetune=False, attach=None):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs

        """
        # assert 0 < prob_drop < 1, "prob_drop must be between 0 and 1 (exclusive)"
    
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)

        self.adv_x, self.adv_edge_index, self.adv_edge_weight= self.adversarial_attack(self.features, self.edge_index, self.edge_weight, idx_train)
        self.labels = labels.to(self.device)
        # self.prob_drop = prob_drop

        if idx_val is None:
            self._train_without_val_with_Adv(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val_with_Adv(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val_Adv(self, labels, idx_train, train_iters, verbose):
        self.model.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()

            # rs_edge_index, rs_edge_weight = self.sample_noise_all(self.prob_drop, self.edge_index, self.edge_weight, self.device)
            output = self.forward(self.adv_x, self.adv_edge_index, self.adv_edge_weight)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.model.eval()
        output = self.forward(self.adv_x, self.adv_edge_index, self.adv_edge_weight)
        self.output = output

    def _train_with_val_with_Adv(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gnn model ===')
        optimizer = optim.Adam(self.model.parameters(), lr=self.model.lr, weight_decay=self.model.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.model.train()
            self.trigger_generator.trojan.eval()
            optimizer.zero_grad()

            # rs_edge_index, rs_edge_weight = self.sample_noise_all(self.prob_drop, self.edge_index, self.edge_weight, self.device)

            output = self.forward(self.adv_x, self.adv_edge_index, self.adv_edge_weight)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.model.eval()
            # rs_edge_index, rs_edge_weight = self.sample_noise_all(self.prob_drop, self.edge_index, self.edge_weight, self.device)
            output = self.forward(self.adv_x, self.adv_edge_index, self.adv_edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.model.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.model.load_state_dict(weights)

    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        
        self.model.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)

            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)
    
    def attack_evaluation(self, args, x, edge_index, edge_weight, labels, idx_atk, idx_clean_test):
        output = self.forward(x, edge_index, edge_weight)
        train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
        print("ASR: {:.4f}".format(train_attach_rate))
        asr = train_attach_rate
        flip_idx_atk = idx_atk[(labels[idx_atk] != args.target_class).nonzero().flatten()]
        flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
        print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
        ca = self.test(x,edge_index, edge_weight,labels,idx_clean_test)
        print("CA: {:.4f}".format(ca))
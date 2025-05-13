from copy import deepcopy
import json
import logging
import os

import numpy as np
import torch
from torch.nn.functional import pad
#from torch.utils.data import DataLoader, Dataset
#from torchtext.vocab import build_vocab_from_iterator

from Hector.hector import Hector
from Hector.Hectree import HecTree


PROBA_SUM = "SUM_PROBA"
PROBA_COMBINE = "COMBINE_PROBA"
PROBA_MAX = "MAX_PROBA"

def exp_sum(a, b):
    """
    sum of 2 proba with their log proba while circumventing precision issues
    """
    min_logproba, max_logproba = np.sort([a, b])
    small_logproba_normalized = min_logproba - max_logproba
    log_proba_combined = max_logproba + np.log(np.exp(small_logproba_normalized) + 1)
    return log_proba_combined

def exp_combine(a,b,threshold = -7): # combining two proba using the appro
    """
    combining two proba, supposing that the events are independant. Using a threshold to avoid accuracy problems, boil down to sum when proba are too small
    """
    if max(a,b)<= threshold :
        return exp_sum(a,b)
    pa = np.exp(a)
    pb = np.exp(b)

    combined_proba = pa + pb - pa*pb
    
    log_proba_combined =  np.log(combined_proba)
    return log_proba_combined



def exp_max(a,b):
    return max(a,b)


def exp_norm(array):
    min_logproba = np.max(array)
    log_norm_array = array - min_logproba
    norm_array = np.exp(log_norm_array)
    norm_array = norm_array / np.sum(norm_array)
    log_proba = min_logproba + np.log(norm_array)
    return log_proba



class CustomXMLHolder():
    def __init__(self, text_batch, beam_parameter, hector ,proba_operator = PROBA_COMBINE,**kwargs ):
        """
        :param dataset: CustomXMLDataset
        :param beam:
        :param tgt_vocab_size:
        :param config:
        """

        self.hector : Hector = hector
        self.beam_param = beam_parameter

        self.hectree  : HecTree = self.hector.hector_forest[0]

        self.current_index=0
        self.ind = np.arange(len(text_batch))
        self.text_ref = text_batch
        self.ind_to_test = deepcopy(self.ind)

        self.paths =  [ [self.hectree.root] for _ in self.ind  ]
        self.kinder = None
        
        self.level = 0  # start at level one

        self.proba = {ind: -1e9 * np.ones_like(self.hectree._mask_template) for ind in self.ind}  # will store logproba for each label
        

        self.prior = [0. for _ in self.paths]  # will store log prior for beam search
        #self.scores = []

        if proba_operator == PROBA_SUM:
            self.proba_operator = exp_sum
        elif proba_operator==PROBA_MAX :
            self.proba_operator = exp_max
        elif proba_operator == PROBA_COMBINE :
            self.proba_operator = exp_combine
        else :
            raise NameError(f"Invalid proba_operator parameter : {proba_operator}. should be one of: {PROBA_SUM}, {PROBA_MAX}, {PROBA_COMBINE}")


        self._prev_index = None
        self._new_ind = []
        self._new_paths = []
        self._new_prior = []
        self._selected_paths = []


    def _get_data(self):

        txt = [ self.text_ref[i] for i in self.ind_to_test]
        path = self.paths

        lvl_mask = [self.hectree.get_level_mask(self.level+1) for _ in self.ind_to_test]
        

        return txt, path, lvl_mask, None

    def _feed_prediction_partial(self, predictions):
        for ii in range(len(predictions)):

            i = self.current_index
            

            cind = self.ind_to_test[i]
            cpath = self.paths[i]
            cprior = self.prior[i]

            pred = predictions[ii]
            predict_order = np.argsort(pred)[::-1]

            if self._prev_index is None:
                self._prev_index = cind
            elif cind != self._prev_index:
                for npath, nprior in sorted(self._selected_paths, key=lambda x: -x[1])[:self.beam_param]:
                    predicted_property = npath[-1]

                    # Combine the two proba while taking into account float precision
                    oprior = self.proba[self._prev_index][predicted_property]
                    log_proba_combined = self.proba_operator(a=oprior, b=nprior)
                    self.proba[self._prev_index][predicted_property] = log_proba_combined

                    

                    if predicted_property not in self.hectree.children_dict : # it is a leaf, let's stop there
                        continue
                    else:
                        
                        self._new_ind.append(self._prev_index)
                        self._new_paths.append(npath)
                        self._new_prior.append(nprior)
                        
                self._prev_index = cind
                self._selected_paths = []

            for predicted_property_idx in predict_order[:self.beam_param]:
                predicted_property = predicted_property_idx
                score = pred[predicted_property_idx]
                npath = cpath + [predicted_property]
                if score >0 : nprior = cprior + np.log(min(score,0.99))  # 0.99 to ensure that children are always slightly less likely than their parents and avoid strange behavior
                else : nprior = cprior + -1e9
                self._selected_paths.append((npath, nprior))

            self.current_index+=1

        


    def _digest_predictions(self):
        
        #for the last index !!!
        for npath, nprior in sorted(self._selected_paths, key=lambda x: -x[1])[:self.beam_param]:
                    predicted_property = npath[-1]

                    # Combine the two proba while taking into account float precision
                    oprior = self.proba[self._prev_index][predicted_property]
                    log_proba_combined = self.proba_operator(a=oprior, b=nprior)
                    self.proba[self._prev_index][predicted_property] = log_proba_combined

                    if predicted_property not in self.hectree.children_dict : # it is a leaf, let's stop there
                        continue
                    else:
                        
                        self._new_ind.append(self._prev_index)
                        self._new_paths.append(npath)
                        self._new_prior.append(nprior)

        
        #saving To-do for next level
        self.ind_to_test = self._new_ind
        self.paths = self._new_paths
        self.prior = self._new_prior

        #resetting temporaty storage

        self._prev_index = None
        self._new_ind = []
        self._new_paths = []
        self._new_prior = []
        self._selected_paths = []
        self.current_index = 0

        self.level += 1

    def _finalize_prediction(self):
        for index_document in self.ind:
            self.proba[index_document][self.hectree.root] = -1e9
        for index_label in  range(len(self.hectree._mask_template)):
            if index_label not in self.hectree._label_to_level :
                self.proba[index_document][index_label] = -1e9
            else : 
                self.proba[index_document][index_label] = max(self.proba[index_document][index_label], -1e8)


        ## need to go back to original labels

        
    def _get_final_prediction(self):

        self._finalize_prediction()

        predictions = []
        scores = []
        indices = sorted(self.proba.keys())

        for ind in indices:
            pred = self.proba[ind]
            predict_order = np.argsort(pred)[::-1]
            predicted_properties = predict_order#[:self.hector._max_padding_tgt]
            predicted_scores = pred[predict_order]#[:self.hector._max_padding_tgt]

            #predicted_labels = [ self.hector._tgt_vocab.get_itos()[i]  for i in predicted_properties]
            predicted_labels = predicted_properties

            predictions.append( predicted_labels.tolist())
            scores.append(np.exp(predicted_scores))

        return predictions, scores



    def run_predictions(self, batch_size = 15):

        while self.level +1 < self.hectree._max_level_tree :
            txt, path, lvl_mask, _ = self._get_data()

            if len(txt)==0 : # all selected paths reached dead ends, no need to continue
                break

            #predictions = []

            for i in range(0, len(txt),batch_size):
                ctxt = txt[i:i+batch_size]
                cpath = path[i:i+batch_size]
                clvl_mask = lvl_mask[i:i+batch_size]
                out = self.hector.test_batch(documents_tokens=ctxt, paths=cpath,lvl_mask=clvl_mask).detach().cpu().numpy()

                #predictions.append(out)
                self._feed_prediction_partial(out)
            
            #predictions =np.concatenate(predictions,axis=0) #(torch.cat(predictions, dim=0)).tolist()

            self._digest_predictions()

        result = self._get_final_prediction()
        return result


if __name__ == "__main__":



    print(exp_sum(0,0))
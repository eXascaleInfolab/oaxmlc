import pickle
import os
from Hectree import HecTree
import torch
from hector import Hector

from prediction_handler import CustomXMLHolder
import time


PATH_TO_GLOVE =  os.path.join("dataTest","glove.840B.300d.gensim")

if __name__ == "__main__":

    #with open(os.path.join("dataTest","taxonomies.pickle"),"rb") as file :
    #    taxos = pickle.load(file)

    
    taxos = torch.load(os.path.join("dataTest","taxonomies.pt"))



    tgt_vocab = torch.load(os.path.join("dataTest","trg_vocab.pt"))
    src_vocab = torch.load(os.path.join("dataTest","src_vocab.pt"))


    abstract_dict = torch.load(os.path.join("dataTest","abstract_dict.pt"))


    #data= torch.load(os.path.join("dataTest","data.pt"))
    data= torch.load(os.path.join("dataTest","batch1.pt"))

    if 1 : 
        h = Hector(src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            path_to_glove=PATH_TO_GLOVE,
            abstract_dict=abstract_dict,
            taxonomies=taxos,
            gpu_target=0
            )

        torch.save(h, "save_hector.pt")

    else :

        h : Hector = torch.load("save_hector.pt")


    for epoch in range(5):

        for task_id in  range(len(data)):
            t= time.time()
            
        
        
            batch = data[task_id]
            #batch= torch.load(os.path.join("dataTest","batch1.pt"))
            documents_tokens, labels_tokens = batch
            
            
            h.train()

            for _ in range(2):
                # Nb of accumulation steps
                for _ in range(20):
                    loss = h.train_on_batch(documents_tokens=documents_tokens, labels_tokens=labels_tokens, task_id=task_id)

            h.eval()

            loss, prec, mass = h.eval_batch(documents_tokens=documents_tokens, labels_tokens=labels_tokens, task_id=task_id)
                
            print("Epoch {} Task {} :  loss {:.3f} mass {:.3f} prec {:.3f} ".format(epoch, task_id, loss.detach().cpu(),mass,prec[1]))
            print("Time spent: {:.2f}".format(time.time()-t))

        print("==========================================")


            

    if 1 :

        xml = CustomXMLHolder(text_batch=data[0][0],task_id=0,beam_parameter=10,hector=h)
        # DF with 2 cols (list of ranked labels predict, and associated probs)
        res = xml.run_predictions(batch_size=15)

        print(res)
        print(data[0][1])


    
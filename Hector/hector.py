from Hector.Hectree import HecTree
from Hector.model import EncoderDecoder, make_model

from Hector.preprocess import build_init_embedding
from Hector.optimizer import rate, DenseSparseAdam, GradientClipper, arate

import torch
from torch.optim.lr_scheduler import ExponentialLR

from Hector.loss import MLabelSmoothing, SimpleLossCompute, CustomPrecisionLoss
import Hector.config as Cf



### TODO :
###     implement eval function



class Hector() : 

    def __init__(self, src_vocab, tgt_vocab, path_to_glove, abstract_dict, taxonomies,  
                 gpu_target  = 0,
                 #adaptive_patience,
                 Number_src_blocs = 6, Number_tgt_blocs = 6, dim_src_embedding = 300,
                  dim_tgt_embedding = 600, dim_feed_forward = 2048, number_of_heads = 12, dropout = 0.1, 
                  learning_rate = 1e-4, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8, weight_decay=0.01, gamma=.99998,
                  accum_iter = 20, loss_smoothing = 0.01,
                  max_padding_document = 128, max_number_of_labels = 20,
                  with_bias = False,
                  **kwargs
                ) -> None:


        self._model :EncoderDecoder= None

        
        print("Building Taxonomy")
        assert len(taxonomies)==1, "Invalid input, taxonomies should be a list of size 1"
        all_label_tokens = sorted(list(tgt_vocab.get_stoi().values()))


        self.hector_forest = []

        widths_collection = None

        with taxonomies[0] as tax:
            root, children_dict = tax
            h = HecTree(root_label=root,children_dict=children_dict, all_tokens=all_label_tokens)
            self.hector_forest.append(h)
            max_level = h.get_max_level()
            h.set_max_level(max_level)



        self._max_level = max_level



        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab

        
            

        

        print("Initializing embeddings")

        if not Cf.QUICK_DEBUG :
            emb_src_init, emb_tgt_init = build_init_embedding(path_to_glove, vocab_src=src_vocab, vocab_tgt=tgt_vocab,
                                                           d_src=dim_src_embedding,
                                                           d_tgt=dim_tgt_embedding, abstract_dict=abstract_dict)
        
        else : 
            emb_src_init = None
            emb_tgt_init = None

        print("Building Model")

        self._init_model(src_vocab, tgt_vocab, N_src = Number_src_blocs, N_tgt = Number_tgt_blocs, d_src=dim_src_embedding,
                          d_tgt = dim_tgt_embedding, d_ff = dim_feed_forward, h = number_of_heads, dropout = dropout,
                            emb_src_init = emb_src_init, emb_tgt_init = emb_tgt_init, with_bias=with_bias)
        

        self._gpu_target = gpu_target
        
        if gpu_target is not None : 
            print("Moving to gpu")
            torch.cuda.set_device(gpu_target)
            #self._model.cuda(gpu_target)
            self._model.to_cuda()


        self._pad_tgt =  tgt_vocab[Cf.padding_token]
        self._pad_src = src_vocab[Cf.padding_token]
        self._max_padding_src = max_padding_document
        self._max_padding_tgt = max_number_of_labels



        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.gamma = gamma

        self.get_optimizer()
        
        
        self._clippy = GradientClipper(start_value=1. * accum_iter, weights = None)


        self.criterion = MLabelSmoothing(
            vocab_size=len(tgt_vocab),
            padding_idx=self._pad_tgt,
            smoothing=loss_smoothing
            )

        if gpu_target is not None : 
            self.criterion.cuda(gpu_target)

        self.loss_function = SimpleLossCompute(self.criterion,widths = None, weights = None)

        self._metric = CustomPrecisionLoss(index_pady=self._pad_tgt, nprec=[1,2,3,4,5,10,20])

        self._training_steps = 0
        self._accum_iter = accum_iter


        print("Ready")

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def get_optimizer(self):
        self._optimizer = DenseSparseAdam(
            self._model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1,self.beta2),
            eps=self.epsilon,
            weight_decay=self.weight_decay,
        )

        self._lr_scheduler = ExponentialLR(
            optimizer=self._optimizer,
            gamma=self.gamma
        )
         


    def _init_model(self,src_vocab, tgt_vocab, N_src, N_tgt, d_src, d_tgt, d_ff, h, dropout, emb_src_init=None, emb_tgt_init=None,
                     with_bias = False,**kwargs):

        self._model = make_model(src_vocab, tgt_vocab, N_src, N_tgt, d_src, d_tgt, d_ff, h, dropout, emb_src_init,
                                  emb_tgt_init, with_bias=with_bias)

    
    def _forward(self,  src, tgt, src_mask, tgt_mask, child_mask, **kwargs):

        return self._model.forward(src, tgt, src_mask, tgt_mask, child_mask)
    



    def _prepare_all_paths(self, labels):
        labs = [l for l in labels if l!= self._pad_tgt]
        output = self.hector_forest[0].labels_to_paths(labs)

        processed_output = []

        for (path, children, mask_level) in output :
            new_path = torch.LongTensor(path+([self._pad_tgt]*(self._max_level-len(path)))).unsqueeze(0)
            new_children = torch.LongTensor(children+([self._pad_tgt]*(self._max_padding_tgt-len(children)))).unsqueeze(0)
            new_mask_level = torch.FloatTensor(mask_level).unsqueeze(0)
            processed_output.append((new_path,new_children,new_mask_level))
        
        return processed_output







    def _prepare_batch(self, documents_tokens, labels_tokens ):  # 0 = <blank>
        """
        :input documents_tokens : 2D tensor of shape (batch_size, max_padding_src) with document tokens ids
                    each row starts with 1 (<s> token id) and ends with 2 (</s> token id)
        :input labels_tokens : 2D tensor of shape (batch_size, max_padding_tgt) with labels ids for each documents


        :variable src: 2D tensor of shape (new_batch_size, max_padding_src) with document tokens ids
                    each row starts with 1 (<s> token id) and ends with 2 (</s> token id)
        :variable path: 2D tensor of shape (new_batch_size, max_padding_tgt) with paths up to current level
        :variable kinder: 2D tensor of shape (new_batch_size, max_padding_tgt) with children of path
        :variable cmask: 2D tensor of shape (new_batch_size, tgt_vocab_size) where all labels except for children of
                    the current path are masked with 0s
        
        """

        src = []
        tgt = []
        kinder = []
        masks = []

        for index_document in range(len(documents_tokens)):
            paths = self._prepare_all_paths(labels_tokens[index_document]) #path_number pairs of (max_padding_tgt,level)
            for (path,children,mask) in paths : 
                tgt.append(path)
                kinder.append(children)
                masks.append(mask)
                
            src.append( documents_tokens[index_document].repeat(len(paths),1) )

 
        
        src = torch.cat(src,dim=0)
        tgt = torch.cat(tgt,dim=0)
        kinder = torch.cat(kinder,dim=0)
        masks = torch.cat(masks,dim=0)


        

        
        src_mask = (src != self._pad_src).unsqueeze(-2)  # padding mask (non-masked positions: True, masked: False)
        tgt_mask = (tgt != self._pad_tgt).unsqueeze(-2)  # padding mask (non-masked positions: True, masked: False)
        
        

        if kinder is not None:
            ntokens = (kinder != self._pad_tgt).data.sum()
        else:
            ntokens = None

        if self._gpu_target is None : 
            return src,tgt,  kinder, masks, src_mask, tgt_mask, ntokens
        
        return src.cuda(),tgt.cuda(),  kinder.cuda(), masks.cuda(), src_mask.cuda(), tgt_mask.cuda(), ntokens.cuda()


    

    def _prepare_test_batch(self, documents_tokens, paths, lvl_mask ):  # 0 = <blank>
        """
        :input documents_tokens : 2D tensor of shape (batch_size, max_padding_src) with document tokens ids
                    each row starts with 1 (<s> token id) and ends with 2 (</s> token id)
        :input labels_tokens : 2D tensor of shape (batch_size, max_padding_tgt) with labels ids for each documents


        :variable src: 2D tensor of shape (new_batch_size, max_padding_src) with document tokens ids
                    each row starts with 1 (<s> token id) and ends with 2 (</s> token id)
        :variable path: 2D tensor of shape (new_batch_size, max_padding_tgt) with paths up to current level
        :variable kinder: 2D tensor of shape (new_batch_size, max_padding_tgt) with children of path
        :variable cmask: 2D tensor of shape (new_batch_size, tgt_vocab_size) where all labels except for children of
                    the current path are masked with 0s
        
        """

        

        src = []
        tgt = []
        masks = []


        for index_document in range(len(documents_tokens)):

            doc = documents_tokens[index_document].unsqueeze(0)
            path = paths[index_document]
            mask = lvl_mask[index_document]    

            new_path = torch.LongTensor(path+([self._pad_tgt]*(self._max_level-len(path)))).unsqueeze(0)
            new_mask_level = torch.FloatTensor(mask).unsqueeze(0)
            



            src.append( doc )
            tgt.append(new_path)
            masks.append(new_mask_level)

 
        
        src = torch.cat(src,dim=0)
        tgt = torch.cat(tgt,dim=0)

        masks = torch.cat(masks,dim=0)


        

        
        src_mask = (src != self._pad_src).unsqueeze(-2)  # padding mask (non-masked positions: True, masked: False)
        tgt_mask = (tgt != self._pad_tgt).unsqueeze(-2)  # padding mask (non-masked positions: True, masked: False)
        
        if self._gpu_target is None : 
            return src,tgt,   masks, src_mask, tgt_mask

        return src.cuda(),tgt.cuda(),  masks.cuda(), src_mask.cuda(), tgt_mask.cuda()





    

    def train_on_batch(self, documents_tokens, labels_tokens):

        src,tgt,  kinder, masks, src_mask, tgt_mask, ntokens = self._prepare_batch(documents_tokens=documents_tokens,labels_tokens=labels_tokens)

        out = self._forward(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask, child_mask=masks)
        loss, loss_node = self.loss_function(x=out, y=kinder, norm=ntokens,level_mask=masks)
        loss_node.backward()

        self._training_steps+=1 
        if self._training_steps % self._accum_iter == 0 :
            norm = self._clippy.clip_gradient(self._model)
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
            self._lr_scheduler.step()

        
        return loss


    




    def eval_batch(self, documents_tokens, labels_tokens):

        src,tgt,  kinder, masks, src_mask, tgt_mask, ntokens = self._prepare_batch(documents_tokens=documents_tokens,labels_tokens=labels_tokens)

        out = self._forward(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask, child_mask=masks)
        loss, _ = self.loss_function(x=out, y=kinder, norm=ntokens,level_mask=masks)
        prec, mass = self._metric(x=out, y=kinder,x_mask=masks)

        return loss, prec, mass
        
        
    
    def test_batch(self, documents_tokens, paths, lvl_mask):
        src,tgt,   masks, src_mask, tgt_mask = self._prepare_test_batch(documents_tokens, paths, lvl_mask)

        out = self._forward(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask, child_mask=masks)

        return out
        



    def freeze(self):
        """
        freeze all layers except the task specifics
        """
        self._model.freeze()


    def get_generator_weights(self):
        weight = self._model.get_generator_weights()
        return weight
   



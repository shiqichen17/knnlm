import torch
import faiss
import math
import numpy as np
import time
import pdb

class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.index = self.setup_faiss(args)

    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int16')

            self.keys = np.memmap(args.dstore_filename + '_keys.npy', dtype=np.float16, mode='r',
                                  shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename + '_vals.npy', dtype=np.int16, mode='r',
                                  shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + '_keys.npy', dtype=np.float32, mode='r',
                                      shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename + '_vals.npy', dtype=np.int, mode='r',
                                  shape=(self.dstore_size, 1))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename + '_keys.npy', dtype=np.float32, mode='r',
                                                  shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension),
                                     dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename + '_vals.npy', dtype=np.int, mode='r',
                                              shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int16 if args.dstore_fp16 else np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int)
            print('Loading to memory took {} s'.format(time.time() - start))

        return index

    def get_knns(self, queries):
        start = time.time()
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns

    def get_knn_log_prob(self, queries, config):
        pdb.set_trace()
        

        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    start = time.time()
                    knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                    if self.half:
                        knns_vecs = knns_vecs.half()
                    query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=2)
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # queries  are TxBxC
        # reshape: (TxB)xC
        qshape = queries.shape  # ([3060, 6, 1024])
        queries = queries.view(-1, qshape[-1])  # ([18360, 1024])
        # tgt = tgt.contiguous().view(-1) #([18360])
        dists, knns = self.get_knns(queries)  # (18360, 32) queries[tgt!=pad_idx] ([2775, 1024])
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        dists = dist_func(dists, knns, queries, function=self.sim_func)  # ([18360, 32])
        probs = torch.log_softmax(dists, dim=-1)  # [18360, 32])

        vocab_size=config.vocab_size #50257
        '''
        a=np.array([-10000 for _ in range(vocab_size)] for _ in range(queries.shape[0]))
        for i in range(queries.shape[0]):
            for j in range(probs.size()[1]):
                a[i][knns[i][j]]=probs[i][j]
        probs=torch.form_numpy(a)
        '''
        a=torch.zeros((queries.shape[0],vocab_size)).fill_(-10000)
        a=a.scatter(1,knns,probs)


        #index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float() #self.vals(103225485, 1)

        '''
        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float() #self.vals(103225485, 1)
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        a=torch.zeros((queries.shape[0],vocab_size)).fill_(-10000)
        a=a.scatter(1,knns,probs)


        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone() #([2775])
        full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000).cuda() #([18360])
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob
        '''

        # TxBx1
        return a #（T*B)*Vocabulary_size


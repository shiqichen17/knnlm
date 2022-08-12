from transformers import GPT2LMHeadModel2,GPT2TokenizerFast
import faiss
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
import time
import scipy
import argparse
import sys
import torch
import faiss
import math
import numpy as np
import time


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

    def get_knn_log_prob(self, queries, pad_idx):
        pdb.set_trace()
        pad_idx=50256

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

        vocab_size=50257
        a=np.array([-10000 for _ in range(vocab_size)] for _ in range(queries.shape[0]))
        for i in range(queries.shape[0]):
            for j in range(probs.size()[1]):
                a[i][knns[i][j]]=probs[i][j]
        probs=torch.form_numpy(a)


        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float() #self.vals(103225485, 1)

        '''

        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float() #self.vals(103225485, 1)
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0


        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone() #([2775])
        full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000).cuda() #([18360])
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob
        '''

        # TxBx1
        return probs


def get_parser():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--probe', default=8, type=int,
                        help='for FAISS, the number of lists to query')
    parser.add_argument('--k', default=32, type=int,
                        help='number of nearest neighbors to retrieve')
    parser.add_argument('--dstore_size', default=103227021, type=int,
                        help='number of items in the knnlm datastore')
    parser.add_argument('--dstore_filename', type=str, default=None,
                        help='File where the knnlm datastore is saved')
    parser.add_argument('--indexfile', type=str, default=None,
                        help='File containing the index built using faiss for knn')
    parser.add_argument('--lmbda', default=0.0, type=float,
                        help='controls interpolation with knn, 0.0 = no knn')
    parser.add_argument('--knn_sim_func', default=None, type=str,
                        help='similarity function to use for knns')
    parser.add_argument('--faiss_metric_type', default='l2', type=str,
                        help='the distance metric for faiss')
    parser.add_argument('--no_load_keys', default=False, action='store_true',
                        help='do not load keys')
    parser.add_argument('--dstore_fp16', default=True, action='store_true',
                        help='if true, datastore items are saved in fp16 and int16')
    parser.add_argument('--move_dstore_to-mem', default=False, action='store_true',
                        help='move the keys and values for knn to memory')
    parser.add_argument('--decoder_embed_dim', default=768, action='store_true',
                        help='move the keys and values for knn to memory')
    parser.add_argument('--gen_subset', default='test', metavar='SPLIT',
                       help='data subset to generate (train, valid, test)')
    parser.add_argument('--output_word_probs', action='store_true',
                       help='if set, outputs words and their predicted log probabilities to standard output')
    parser.add_argument('--output_word_stats', action='store_true',
                       help='if set, outputs word statistics such as word count, average probability, etc')
    parser.add_argument('--context_window', default=0, type=int, metavar='N',
                       help='ensures that every evaluated token has access to a context of at least this size,'
                            ' if possible')
    parser.add_argument('--softmax_batch', default=sys.maxsize, type=int, metavar='N',
                       help='if BxT is more than this, will batch the softmax over vocab to this amount of tokens'
                            ' in order to fit into GPU memory')
    parser.add_argument('--lm_eval', default=True, action='store_true',
                       help='helpful for certain ops that are only used during eval')
    parser.add_argument('--knnlm', action='store_true',
                       help='use the k-nearest neighbors language model')
    parser.add_argument('--save_knnlm_dstore', action='store_true',
                       help='save keys for the knnlm datastore')
    parser.add_argument('--dstore_mmap', default=None, type=str,
                       help='If saving knnlm dstore, save keys and values to this file')
    parser.add_argument('--fp16',default=True, action='store_true',
                        help='save keys for the knnlm datastore')
    parser.add_argument('--labda', default=0.25, action='store_true',
                        help='save keys for the knnlm datastore')
    args = parser.parse_args()
    return  args

args=get_parser()
subset=args.gen_subset
dstore=args.save_knnlm_dstore
knnlm=args.knnlm
device = "cuda"
model_id = "gpt2"
#model = GPT2LMHeadModel2.from_pretrained(model_id).to(device)
model = GPT2LMHeadModel2.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
device = "cuda"
subset= load_dataset("wikitext", "wikitext-103-v1", split=str(subset))
encodings = tokenizer("\n\n".join(subset["text"]), return_tensors="pt")

max_length = model.config.n_positions
stride = 512
nlls = []

if dstore:
    dstore_keys = np.memmap(args.dstore_mmap + '_keys.npy', dtype=np.float16, mode='w+',
                            shape=(args.dstore_size, args.decoder_embed_dim))
    dstore_vals = np.memmap(args.dstore_mmap + '_vals.npy', dtype=np.int16, mode='w+', shape=(args.dstore_size, 1))

for i in tqdm(range(0, min(encodings.input_ids.size(1), args.dstore_size), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1), args.dstore_size)
    trg_len = end_loc - i  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        if knnlm:
            knnlm_dstore=KNN_Dstore(args)
            outputs=model(input_ids,knnlm_dstore,fp16=args.fp16,labda=args.labda)
        else:
            outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs[0] * trg_len

    nlls.append(neg_log_likelihood)

    if dstore:

    #dstore_keys[begin_loc:end_loc] = outputs.logits[0].view( -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)
        dstore_keys[begin_loc:end_loc] = outputs.what_i_need.view( -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)

        dstore_vals[begin_loc:end_loc] = target_ids.view(
            -1, 1).cpu().numpy().astype(np.int16)
ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(ppl)
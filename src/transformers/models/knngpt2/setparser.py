import argparse
import sys
def get_parser():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--probe', default=8, type=int,
                        help='for FAISS, the number of lists to query')
    parser.add_argument('--k', default=32, type=int,
                        help='number of nearest neighbors to retrieve')
    parser.add_argument('--dstore_size', default=119601966, type=int,
                        help='number of items in the knnlm datastore')
    parser.add_argument('--dstore_filename', type=str, default=None,
                        help='File where the knnlm datastore is saved')
    parser.add_argument('--indexfile', type=str, default=None,
                        help='File containing the index built using faiss for knn')
    parser.add_argument('--labda', default=0.0, type=float,
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
    parser.add_argument('--vocab_size', default=50257,
                        help='vocab_size')
    args = parser.parse_args()
    return  args





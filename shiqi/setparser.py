import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--knn-keytype', type=str, default=None,
                        help='for knnlm WT103 results, use last_ffn_input')
    parser.add_argument('--probe', default=8, type=int,
                        help='for FAISS, the number of lists to query')
    parser.add_argument('--k', default=1024, type=int,
                        help='number of nearest neighbors to retrieve')
    parser.add_argument('--dstore-size', default=103227021, type=int,
                        help='number of items in the knnlm datastore')
    parser.add_argument('--dstore-filename', type=str, default=None,
                        help='File where the knnlm datastore is saved')
    parser.add_argument('--indexfile', type=str, default=None,
                        help='File containing the index built using faiss for knn')
    parser.add_argument('--lmbda', default=0.0, type=float,
                        help='controls interpolation with knn, 0.0 = no knn')
    parser.add_argument('--knn-sim-func', default=None, type=str,
                        help='similarity function to use for knns')
    parser.add_argument('--faiss-metric-type', default='l2', type=str,
                        help='the distance metric for faiss')
    parser.add_argument('--no-load-keys', default=False, action='store_true',
                        help='do not load keys')
    parser.add_argument('--dstore-fp16', default=True, action='store_true',
                        help='if true, datastore items are saved in fp16 and int16')
    parser.add_argument('--move-dstore-to-mem', default=False, action='store_true',
                        help='move the keys and values for knn to memory')
    parser.add_argument('--decoder_embed_dim', default=False, action='store_true',
                        help='move the keys and values for knn to memory')

    args = parser.parse_args()
    return args




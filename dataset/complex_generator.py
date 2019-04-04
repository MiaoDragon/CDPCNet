from complex_util import generate_collision, generate_no_collision
import argparse
import os
import numpy as np
from tqdm import tqdm
# generate equal number of collision data and collision-free data
def main(args):
    """
        write in minibatches
        write in the format:
            save_path + cP1_[minibatch_idx]
            save_path + cP2_[minibatch_idx]
            save_path + P1_[minibatch_idx]
            save_path + P2_[minibatch_idx]
    """
    # check if fileexist
    if os.path.exists(args.save_path+'P1_0.npy'):
        return
    minibatch = args.minibatch
    for minibatch_i in tqdm(range(args.data_size // 2 // minibatch)):
        #cP1, cP2 = generate_collision(minibatch, args.num_point, pert_ratio=0.05)
        P1, P2 = generate_no_collision(minibatch, args.num_point)
        #np.save(args.save_path+'cP1_%d' % (minibatch_i), cP1)
        #np.save(args.save_path+'cP2_%d' % (minibatch_i), cP2)
        np.save(args.save_path+'P1_%d' % (minibatch_i), P1)
        np.save(args.save_path+'P2_%d' % (minibatch_i), P2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--data_size', type=int, default=20000,help='total dataset size')
    parser.add_argument('--minibatch', type=int, default=1000,help='minibatch for loading')
    parser.add_argument('--num_point', type=int, default=2800, help='number of points in the cloud')
    parser.add_argument('--save_path', type=str, default='./geometry/', help='save dir')

    args = parser.parse_args()
    main(args)

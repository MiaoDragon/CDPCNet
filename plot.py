from plot_util import *
from save_util import *
import argparse
def main(args):
    file = args.path
    train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = load_eval(args.path)
    print(train_loss)
    plot(loss=train_loss, acc=train_acc, name='train')
    
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help="loc")
args = parser.parse_args()
print(args)
main(args)

from plot_util import *
from save_util import *
import argparse
def main(args):
    file = args.path
    train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = load_eval(args.path)
    plot(loss=train_loss, acc=train_acc, name='train')
    plot(loss=val_loss, acc=val_acc, name='val')
    plot(loss=test_loss, acc=test_acc, name='test')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help="loc")
args = parser.parse_args()
print(args)
main(args)

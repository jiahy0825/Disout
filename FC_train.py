import argparse
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from models.disoutCNN import CnnDisout

from utils import get_training_dataloader, get_test_dataloader, WarmUpLR

parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default='~/data/cifar10/')
parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--dist_prob', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=5)
args, unparsed = parser.parse_known_args()


def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None):
    lr, decay_rate = args.lr, 0.1
    if epoch >= 200:
        lr *= decay_rate ** 1
    elif epoch >= 300:
        lr *= decay_rate ** 2
    elif epoch >= 400:
        lr *= decay_rate ** 3

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(epoch):
    train_loss = 0.0
    correct = 0.0

    net.train()
    lr = adjust_learning_rate(optimizer, epoch, args)

    for batch_index, (images, labels) in enumerate(training_loader):
        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Train set Epoch: {epoch}  Average_loss: {loss:.4f}, Accuracy: {acc:.4f}, lr: {lr: .6f}'.
          format(epoch = epoch, loss = train_loss / len(training_loader.dataset),
                 acc=correct.float() / len(training_loader.dataset), lr = lr))


def eval():
    net.eval()

    test_loss = 0.0
    correct = 0.0

    for (images, labels) in cifar10_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average_loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar10_test_loader.dataset),
        correct.float() / len(cifar10_test_loader.dataset)
    ))

    return correct.float() / len(cifar10_test_loader.dataset)


if __name__ == '__main__':

    training_loader = get_training_dataloader(args.data_root,
                                              (0.4914, 0.4822, 0.4465),
                                              (0.2023, 0.1994, 0.2010),
                                              num_workers=args.w,
                                              batch_size=args.b,
                                              shuffle=args.s
                                              )

    cifar10_test_loader = get_test_dataloader(args.data_root,
                                              (0.4914, 0.4822, 0.4465),
                                              (0.2023, 0.1994, 0.2010),
                                              num_workers=args.w,
                                              batch_size=args.b,
                                              shuffle=args.s
                                              )

    net = CnnDisout(num_class=10, dist_prob=args.dist_prob,
                    alpha=args.alpha, nr_steps=len(training_loader) * args.epochs).cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        acc = eval()
        if best_acc < acc:
            best_acc = acc
    print('best_acc:', best_acc)

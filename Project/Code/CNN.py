import torch
import torch.nn as nn
import torch.nn.functional as F
# import sys
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from collections import OrderedDict

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder5 = UNet._block(features * 8, features * 8, name="enc4")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv5 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder5 = UNet._block((features * 8) * 2, features * 16, name="dec4")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        bottleneck = self.bottleneck(self.pool5(enc5))

        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    tmp = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx * len(data) // 100 > tmp:
            print('Train Epoch: {} [{:3d}/{} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            tmp += 1


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    err = 0
    h, w = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            output = model(data)
            h, w = output.shape[2:]
            pred = output.round()
            pred -= target
            pred = torch.abs(pred)
            err += torch.sum(pred)
            # sum up batch loss
            test_loss += F.mse_loss(output, target)

    test_loss /= len(test_loader.dataset)
    correct = len(test_loader.dataset) * h * w - err

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset) * h * w,
        100. * correct / (len(test_loader.dataset) * h * w)))
    return correct, (len(test_loader.dataset) * h * w)


def train_together():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='marker', help='mask or marker')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--batch', type=int, default=9, help='batch size')
    parser.add_argument('--epo', type=int, default=200, help='max training epochs')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    trainset = np.load('processed_data/1-2-Imgs.npy')
    model_dir = 'models/'
    if args.net in ['mask', 'm']:
        model_dir += 'mask/'
        target = np.load('processed_data/1-2-Masks.npy')
    elif args.net in ['ero', 'e', 'marker']:
        # model_dir += 'marker/'
        # target = np.load('processed_data/1-2-Markers.npy')
        model_dir += 'marker/'
        target = np.load('processed_data/1-2-EroMasks.npy')
    # print(trainset.shape, target.shape)
    X_train, X_test, y_train, y_test = train_test_split(trainset, target, test_size=0.2)
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    # Fetch and load the training data
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    # Fetch and load the test data
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch)

    net = UNet().to(device)

    if list(net.parameters()):
        optimizer = torch.optim.Adam(net.parameters(), eps=0.000001, lr=args.lr,
                                     betas=(0.9, 0.999), weight_decay=0.0001)
        best = 0
        for epoch in range(1, args.epo + 1):
            train(args, net, device, train_loader, optimizer, epoch)
            correct, total = test(args, net, device, test_loader)
            # save the best model
            if correct / total > best:
                torch.save(net, model_dir + f'{str(epoch+10000)[1:]}.pkl')
                best = correct / total


if __name__ == '__main__':
    train_together()

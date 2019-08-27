import torch
import torch.nn as nn
from hamiltonianNN.discrete_models import  HamilNet
from hamiltonianNN.models import HODENet

def accuracy(test_loader,model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (
        100 * correct / total))


def norm(dim):
    return nn.GroupNorm(int(dim / 2), dim)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, num_filters, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.num_filters = num_filters
        self.conv1 = nn.Conv2d(inplanes, self.num_filters,
                               kernel_size=3, stride=stride, padding=1)
        self.norm2 = norm(num_filters)
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters,
                               kernel_size=3, padding=1)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ResNet(nn.Module):
    """Creates an ODEBlock with a convolutional ODEFunc followed by a Linear
    layer.

    Parameters
    ----------
    device : torch.device

    img_size : tuple of ints
        Tuple of (channels, height, width).

    num_filters : int
        Number of convolutional filters.

    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.

    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'

    tol : float
        Error tolerance.

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """

    def __init__(self, device, img_size, num_filters, num_layers=6, output_dim=1, non_linearity='relu'):
        super(ResNet, self).__init__()
        self.device = device
        self.img_size = img_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.channels, self.height, self.width = img_size
        downsampling_layers = [
            nn.Conv2d(1, num_filters, 3, 1),
            ResBlock(num_filters, num_filters, stride=2, downsample=conv1x1(num_filters, num_filters, 2)),
            ResBlock(num_filters, num_filters, stride=2, downsample=conv1x1(num_filters, num_filters, 2)),
        ]
        feature_layers = [ResBlock(num_filters, num_filters) for _ in range(num_layers)]
        fc_layers = [norm(num_filters), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten()]
        self.net = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
        self.linear_layer = nn.Linear(num_filters, self.output_dim)

    def forward(self, x, return_features=False):
        features = self.net(x)
        pred = self.linear_layer(features)
        if return_features:
            return features, pred
        return pred


class ConvHamilNet(nn.Module):
    """Creates an ODEBlock with a convolutional ODEFunc followed by a Linear
    layer.

    Parameters
    ----------
    device : torch.device

    img_size : tuple of ints
        Tuple of (channels, height, width).

    num_filters : int
        Number of convolutional filters.

    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.

    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'

    tol : float
        Error tolerance.

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """

    def __init__(self, device, img_size, num_filters, hidden_dim, output_dim=1,
                 augment_dim=0, non_linearity='relu',
                 tol=1e-3, adjoint=False, final_time=1, level=7, method='leapfrog', discret=False, num_layers=20):
        super(ConvHamilNet, self).__init__()
        self.device = device
        self.img_size = img_size
        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.tol = tol
        if discret:
            self.discret = discret
            self.num_layers = num_layers
            self.hamilnet = HamilNet(device, num_filters, hidden_dim, num_layers=num_layers, output_dim=output_dim,
                                     augment_dim=augment_dim, final_time=final_time, activation=non_linearity)
        else:
            self.hamilnet = HODENet(device, num_filters, hidden_dim, output_dim=output_dim, level=level,
                                    augment_dim=augment_dim,adjoint=adjoint,
                                    non_linearity=non_linearity, final_time=final_time, method=method)
            self.odeblock = self.hamilnet.odeblock
        self.channels, self.height, self.width = img_size
        downsampling_layers = [
            nn.Conv2d(self.channels, num_filters, 3, 1),
            norm(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 4, 2, 1),
            norm(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 4, 2, 1),
        ]
        fc_layers = [norm(num_filters), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten()]
        feature_layers = [self.hamilnet]

        self.net = nn.Sequential(*downsampling_layers, *fc_layers, *feature_layers).to(device)

    def forward(self, x):
        pred = self.net(x)
        return pred

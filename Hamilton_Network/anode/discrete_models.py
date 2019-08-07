import torch.nn as nn
import torch
import numpy

class ResidualBlock(nn.Module):
    """Following the structure of the one implemented in
    https://arxiv.org/pdf/1806.10909.pdf
    """
    def __init__(self, data_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, data_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        return x + self.mlp(x)


class ResNet(nn.Module):
    """ResNet which maps data_dim dimensional points to an output_dim
    dimensional output.
    """
    def __init__(self, data_dim, hidden_dim, num_layers, output_dim=1,
                 is_img=False):
        super(ResNet, self).__init__()
        residual_blocks = \
            [ResidualBlock(data_dim, hidden_dim) for _ in range(num_layers)]
        self.residual_blocks = nn.Sequential(*residual_blocks)
        self.linear_layer = nn.Linear(data_dim, output_dim)
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.is_img = is_img

    def forward(self, x, return_features=False):
        if self.is_img:
            # Flatten image, i.e. (batch_size, channels, height, width) to
            # (batch_size, channels * height * width)
            features = self.residual_blocks(x.view(x.size(0), -1))
        else:
            features = self.residual_blocks(x)
        pred = self.linear_layer(features)
        if return_features:
            return features, pred
        return pred

    @property
    def hidden_dim(self):
        return self.residual_blocks.hidden_dim


class HamilBlock(nn.Module):

    def __init__(self,device, input_dim,hidden_dim, num_layers, final_time=1, non_linearity='relu'):
        super(HamilBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.num_layers = num_layers
        self.final_time = final_time
        if non_linearity=='tanh':
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh()
            )
        elif non_linearity=='relu':
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(True)
            )
        self.vts = nn.Linear(input_dim, input_dim,bias=False)

    def forward(self, x):
        x_ = x[-1]
        z = x_[:, self.input_dim:self.input_dim*2]
        y = x_[:, 0:self.input_dim]
        dz = - self.mlp(y)
        weight =self.mlp[0].weight
        dz = torch.mm(dz, weight)
        z = z+(self.final_time/self.num_layers)*dz
        dy = self.vts(z)
        weight = self.vts.weight
        dy = torch.mm(dy, weight)
        out = torch.cat((dy, dz), 1).to(self.device)
        x.append(x_ + (self.final_time / self.num_layers) * out)
        return x


class HamilNet(nn.Module):
    """An HamilBlock followed by a Linear layer.

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.


    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.

    augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'

    tol : float
        Error tolerance.
    """

    def __init__(self, device, data_dim,hidden_dim, num_layers, augment_dim=0, output_dim=1,final_time=1,activation='relu',
                 is_img=False):
        super(HamilNet, self).__init__()
        hamil_blocks = \
            [HamilBlock(device,data_dim+augment_dim,hidden_dim,num_layers,final_time,non_linearity=activation) for _ in range(num_layers)]
        self.hamil_blocks = nn.Sequential(*hamil_blocks)
        self.linear_layer = nn.Linear(data_dim*2+augment_dim*2, output_dim)
        self.num_layers = num_layers
        self.activation = activation
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.augment_dim = augment_dim
        self.is_img = is_img
        self.device = device

    def forward(self, x, return_features=False):
        if self.augment_dim > 0:
            if self.is_img:
                # Add augmentation
                batch_size, channels, height, width = x.shape
                aug = torch.zeros(batch_size, self.augment_dim,
                                  height, width).to(self.device)
                # Shape (batch_size, channels + augment_dim, height, width)
                x_aug = torch.cat([x, aug], 1)
            else:
                # Add augmentation
                aug = torch.zeros(x.shape[0], self.augment_dim).to(self.device)
                # Shape (batch_size, data_dim + augment_dim)
                x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        aug = torch.zeros(x_aug.shape[0], x_aug.shape[1]).to(self.device)
        x_aug = [torch.cat([x_aug, aug], 1)]

        features = self.hamil_blocks(x_aug)
        pred = self.linear_layer(features[-1])
        if return_features:
            return features, pred
        return pred


class MLPNet(nn.Module):
    """
    """
    def __init__(self, data_dim, hidden_dim):
        super(MLPNet, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x)


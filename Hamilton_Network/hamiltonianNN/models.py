import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
import sys
import random


MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver


class ODEFuncH(nn.Module):
    """ the derivative of Hamiltonian with classical dynamic.

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.

    hidden_dim : int
        Dimension of hidden layers.

    augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.


    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0, non_linearity='relu'):
        super(ODEFuncH, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations


        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim, bias=False)

        if non_linearity == 'relu':
            self.non_linearity1 = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity1 = nn.Softplus()
        elif non_linearity == 'tanh':
            self.non_linearity1 = nn.Tanh()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time. Shape (1,).

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        # Forward pass of model corresponds to one function evaluation, so
        # increment counter
        z=x[:,self.input_dim:self.input_dim*2]
        x = x[:, 0:self.input_dim]

        self.nfe += 1
        out = self.fc1(x)
        outz = - self.non_linearity1(out)
        outx = self.fc2(z)
        weight = self.fc2.weight
        outx = torch.mm(outx, weight)
        out = torch.cat((outx, outz), 1).to(self.device)
        return out





class ODEFuncH2(nn.Module):
    """the derivative of Hamiltonian with quadratic Potential.

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.

    hidden_dim : int
        Dimension of hidden layers.

    augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                  non_linearity='tanh'):
        super(ODEFuncH2, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations

        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim, bias=False)

        if non_linearity == 'relu':
            self.non_linearity1 = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity1 = nn.Softplus()
        elif non_linearity == 'tanh':
            self.non_linearity1 = nn.Tanh()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time. Shape (1,).

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        # Forward pass of model corresponds to one function evaluation, so
        # increment counter
        z = x[:, self.input_dim:self.input_dim*2]
        x = x[:, 0:self.input_dim]

        self.nfe += 1
        out = self.fc1(x)
        weight = self.fc1.weight
        bias = self.fc1.bias
        out=1/2*(torch.mm(x,weight)+bias)+1/2*out
        bias=bias.view((x.shape[1],1))

        activation1=sum(torch.mm(x, weight.t()).mul(x).t())/2

        activation2 =torch.mm(x, bias).reshape(-1)

        activation =activation1+activation2
        activation=activation.view((x.shape[0],1))
        ac=self.non_linearity1(activation)
        outz = - torch.mul(out, ac)
        outx = self.fc2(z)
        weight = self.fc2.weight
        outx = torch.mm(outx, weight)
        out = torch.cat((outx, outz), 1).to(self.device)
        return out




class ODEFuncH3(nn.Module):
    """the derivative of Hamiltonian with general linear Potential.

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.

    hidden_dim : int
        Dimension of hidden layers.

    augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.


    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                  non_linearity='relu'):
        super(ODEFuncH3, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.alpha = nn.Parameter(torch.Tensor([random.random() for i in range(self.input_dim)]))

        if non_linearity == 'relu':
            self.non_linearity1 = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity1 = nn.Softplus()
        elif non_linearity == 'tanh':
            self.non_linearity1 = nn.Tanh()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time. Shape (1,).

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        # Forward pass of model corresponds to one function evaluation, so
        # increment counter
        z = x[:, self.input_dim:self.input_dim*2]
        x = x[:, 0:self.input_dim]

        self.nfe += 1
        out = self.fc1(x)
        outz = - self.non_linearity1(out)
        weight = self.fc1.weight
        outz = torch.mm(outz, weight)
        alpha=torch.diag(self.alpha)
        outz = torch.mm(outz, alpha)
        outx = self.fc2(z)
        weight = self.fc2.weight
        outx = torch.mm(outx, weight)
        out = torch.cat((outx, outz), 1).to(self.device)
        return out



class ODEFuncH4(nn.Module):
    """the derivative of Hamiltonian with quadratic Potential(2nd form)

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.

    hidden_dim : int
        Dimension of hidden layers.

    augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.


    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                 non_linearity='tanh'):
        super(ODEFuncH4, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim, bias=False)

        if non_linearity == 'relu':
            self.non_linearity1 = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity1 = nn.Softplus()
        elif non_linearity == 'tanh':
            self.non_linearity1 = nn.Tanh()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time. Shape (1,).

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        # Forward pass of model corresponds to one function evaluation, so
        # increment counter
        z = x[:, self.input_dim:self.input_dim*2]
        x = x[:, 0:self.input_dim]

        self.nfe += 1
        out = self.fc1(x)
        weight = self.fc1.weight
        bias = self.fc1.bias
        bias=bias.reshape((x.shape[1],1))

        activation1=sum(torch.mm(x, weight.t()).mul(x).t())/2

        activation2 =torch.mm(x, bias).reshape(-1)

        activation =activation1+activation2
        activation=activation.reshape((x.shape[0],1))
        ac=self.non_linearity1(activation)
        outz = - torch.mul(out, ac)
        outx = self.fc2(z)
        weight = self.fc2.weight
        outx = torch.mm(outx, weight)
        out = torch.cat((outx, outz), 1).to(self.device)
        return out



class HODEFunc_inspired(nn.Module):
    """the derivative of Hamiltonian Inspired ODE

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.

    hidden_dim : int
        Dimension of hidden layers.

     augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.



    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim,augment_dim=-1,
                  non_linearity='relu'):
        super(HODEFunc_inspired, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.input_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim=augment_dim
        self.nfe = 0  # Number of function evaluations
        self.bias_aug=nn.Parameter(0*torch.Tensor( 1 , self.hidden_dim))

        self.fc1 = nn.Linear(self.hidden_dim, self.input_dim)

        if non_linearity == 'relu':
            self.non_linearity1 = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity1 = nn.Softplus()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time. Shape (1,).

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        # Forward pass of model corresponds to one function evaluation, so
        # increment counter

        z=x[:,self.input_dim:self.input_dim+self.hidden_dim]
        x = x[:, 0:self.input_dim]
        outy =  self.fc1(z)
        outz = torch.mm(x, self.fc1.weight) \
                   + self.bias_aug
        outz=-1*self.non_linearity1( outz)
        outy=self.non_linearity1(outy)
        out = torch.cat((outy, outz), 1)
        return out


class HODEFunc_inspired2(nn.Module):
    """the derivative of Hamiltonian Inspired ODE with level 2 ODE function

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.

    hidden_dim : int
        Dimension of hidden layers.

     augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.


    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim,augment_dim=-1,
                  non_linearity='relu'):
        super(HODEFunc_inspired2, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.input_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim=augment_dim
        self.nfe = 0  # Number of function evaluations
        self.bias_aug=nn.Parameter(0*torch.Tensor( 1 , self.hidden_dim))
        self.bias_aug2 = nn.Parameter(0 * torch.Tensor(1, self.hidden_dim))

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, self.input_dim)

        if non_linearity == 'relu':
            self.non_linearity1 = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity1 = nn.Softplus()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time. Shape (1,).

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        # Forward pass of model corresponds to one function evaluation, so
        # increment counter

        z=x[:,self.input_dim:self.input_dim+self.hidden_dim]
        x = x[:, 0:self.input_dim]
        outy =  self.fc1(z)
        outz = torch.mm(x , self.fc2.weight) \
                   + self.bias_aug
        outy=self.non_linearity1(outy)
        outz=self.non_linearity1(outz)

        outy=self.fc2(outy)
        outz=torch.mm(outz, -1 * self.fc1.weight ) \
                   + self.bias_aug2
        out = torch.cat((outy,  outz ), 1)
        return out


class HODEBlock(nn.Module):
    """Solves ODE defined by odefunc.

    Parameters
    ----------
    device : torch.device

    odefunc : ODEFunc instance or anode.conv_models.ConvODEFunc instance
        Function defining dynamics of system.

    is_conv : bool
        If True, treats odefunc as a convolutional model.

    tol : float
        Error tolerance.

    level : int
        the level of ODE function

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.

    method : function
        integration method
    """
    def __init__(self, device, odefunc, is_conv=False, tol=1e-3, adjoint=False, final_time=1, level=7, method='leapfrog'):
        super(HODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.final_time = final_time
        self.tol = tol
        self.level= level
        self.method=method

    def forward(self, x, eval_times=None):
        """Solves ODE starting from x.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)
ODEBlock
        eval_times : None or torch.Tensor
            If None, returns solution of ODE at final time t=1. If torch.Tensor
            then returns full ODE trajectory evaluated at points in eval_times.
        """
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, self.final_time]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)


        if self.odefunc.augment_dim > 0:
            if self.is_conv:
                # Add augmentation
                batch_size, channels, height, width = x.shape
                aug = torch.zeros(batch_size, self.odefunc.augment_dim,
                                  height, width).to(self.device)
                # Shape (batch_size, channels + augment_dim, height, width)
                x_aug = torch.cat([x, aug], 1)
            else:
                # Add augmentation
                aug = torch.zeros(x.shape[0], self.odefunc.augment_dim).to(self.device)
                # Shape (batch_size, data_dim + augment_dim)
                x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        if self.level >4:
            aug = torch.zeros(x_aug.shape[0], x_aug.shape[1]).to(self.device)
            x_aug = torch.cat([x_aug, aug], 1)
        elif self.level == 3:
            aug = torch.zeros(x.shape[0], self.odefunc.hidden_dim).to(self.device)
            x_aug = torch.cat([x, aug], 1)
        elif self.level == 4:
            aug = torch.zeros(x.shape[0], self.odefunc.hidden_dim).to(self.device)
            x_aug = torch.cat([x, aug], 1)

        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x_aug, integration_time,
                                 rtol=self.tol, atol=self.tol, method=self.method,
                                 options={'max_num_steps': MAX_NUM_STEPS})
        else:
            out = odeint(self.odefunc, x_aug, integration_time,
                         rtol=self.tol, atol=self.tol, method=self.method,
                         options={'max_num_steps': MAX_NUM_STEPS})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        """Returns ODE trajectory.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)

        timesteps : int
            Number of timesteps in trajectory.
        """
        integration_time = torch.linspace(0., float(self.final_time), timesteps)
        return self.forward(x, eval_times=integration_time)


class hamil_ODENet(nn.Module):
    """An HODEBlock followed by a Linear layer.

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.

    hidden_dim : int
        Dimension of hidden layers.

    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.

    augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.

    non_linearity : string
        One of 'relu' and 'softplus' 'tanh'

    tol : float
        Error tolerance.

    level : int
        the level of ODE function

    final_time : float
        the final time for the ODE Solver

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.

    method : function
        integration method
    """
    def __init__(self, device, data_dim, hidden_dim, output_dim=1,
                 augment_dim=0, non_linearity='relu',
                 tol=1e-3, adjoint=False, level=7, final_time=1., method='leapfrog'):
        super(hamil_ODENet, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.final_time = final_time
        self.tol = tol
        if level == 5:
            odefunc = ODEFuncH(device, data_dim, hidden_dim, augment_dim, non_linearity)
        elif level == 6:
            odefunc = ODEFuncH2(device, data_dim, hidden_dim, augment_dim, non_linearity)
        elif level == 7:
            odefunc = ODEFuncH3(device, data_dim, hidden_dim, augment_dim, non_linearity)
        elif level == 8:
            odefunc = ODEFuncH4(device, data_dim, hidden_dim, augment_dim, non_linearity)
        elif level == 3:
            odefunc = HODEFunc_inspired(device, data_dim, hidden_dim, augment_dim, non_linearity)
        elif level == 4:
            odefunc = HODEFunc_inspired2(device, data_dim, hidden_dim, augment_dim, non_linearity)
        else:
            sys.stderr.write('need level between 3 to 8 but get %d' % level)

        self.odeblock = HODEBlock(device, odefunc, tol=tol, adjoint=adjoint, final_time=final_time,level=level,method=method)

        if level > 4:
            self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim * 2,
                                          self.output_dim)
        elif level == 3:

            self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim + self.hidden_dim,
                                          self.output_dim)
        elif level == 4:
            self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim + self.hidden_dim,
                                          self.output_dim)
        else :
            self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim ,
                                          self.output_dim)

    def forward(self, x, return_features=False):
        features = self.odeblock(x)
        pred = self.linear_layer(features)
        if return_features:
            return features, pred
        return pred




import torch
import torch.nn as nn
from math import pi
from torchdiffeq import odeint, odeint_adjoint
import numpy

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver


class ODEFunc1(nn.Module):
    """MLP modeling the derivative of ODE system with ODE lv1.

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

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0, time_dependent=False, non_linearity='relu'):
        super(ODEFunc1, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        if time_dependent:
            self.fc1 = nn.Linear(self.input_dim + 1, self.input_dim)
        else:
            self.fc1 = nn.Linear(self.input_dim, self.input_dim)

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
        self.nfe += 1

        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(x)
        out = self.non_linearity1(out)

        return out


class ODEFunc2(nn.Module):
    """MLP modeling the derivative of ODE system lv2.

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

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                 time_dependent=False, non_linearity='relu'):
        super(ODEFunc2, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        if time_dependent:
            self.fc1 = nn.Linear(self.input_dim + 1, hidden_dim)
        else:
            self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.input_dim)

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
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(x)
        out = self.non_linearity1(out)
        out = self.fc2(out)
        return out


class ODEFuncH(nn.Module):
    """MLP modeling the derivative of ODE system lv2.

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

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                 time_dependent=False, non_linearity='relu'):
        super(ODEFuncH, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        if time_dependent:
            self.fc1 = nn.Linear(self.input_dim + 1, self.input_dim)
        else:
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
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(x)
        outz = - self.non_linearity1(out)
        outx = self.fc2(z)
        weight = self.fc2.weight
        outx = torch.mm(outx, weight)
        out = torch.cat((outx, outz), 1).to(self.device)
        return out





class ODEFuncH2(nn.Module):
    """MLP modeling the derivative of ODE system lv2.

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

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                 time_dependent=False, non_linearity='tanh'):
        super(ODEFuncH2, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        if time_dependent:
            self.fc1 = nn.Linear(self.input_dim + 1, self.input_dim)
        else:
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
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
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
    """MLP modeling the derivative of ODE system lv2.

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

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                 time_dependent=False, non_linearity='tanh'):
        super(ODEFuncH3, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        if time_dependent:
            self.fc1 = nn.Linear(self.input_dim + 1, self.hidden_dim)
        else:
            self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
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
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(x)
        outz = - self.non_linearity1(out)
        weight = self.fc1.weight
        outz = torch.mm(outz, weight)
        outx = self.fc2(z)
        weight = self.fc2.weight
        outx = torch.mm(outx, weight)
        out = torch.cat((outx, outz), 1).to(self.device)
        return out



class ODEFuncH4(nn.Module):
    """MLP modeling the derivative of ODE system lv2.

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

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                 time_dependent=False, non_linearity='tanh'):
        super(ODEFuncH4, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        if time_dependent:
            self.fc1 = nn.Linear(self.input_dim + 1, self.input_dim)
        else:
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
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
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



class ODEFunc3(nn.Module):
    """MLP modeling the derivative of ODE system.

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

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim,augment_dim=-1,
                 time_dependent=False, non_linearity='relu'):
        super(ODEFunc3, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.input_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim=augment_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent
        self.bias_aug=nn.Parameter(0*torch.Tensor( 1 , self.hidden_dim))

        if time_dependent:
            self.fc1 = nn.Linear(self.hidden_dim + 1, self.input_dim)
        else:
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
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            t_and_z = torch.cat([t_vec, z], 1)
            # Shape (batch_size, hidden_dim)
            outy = self.fc1(t_and_z)
            outz = torch.mm( t_and_z , - self.fc1   )+ self.fc1.bias
        else:
            outy =  self.fc1(z)
            outz = torch.mm(x, self.fc1.weight) \
                   + self.bias_aug
        outz=-1*self.non_linearity1( outz)
        outy=self.non_linearity1(outy)
        out = torch.cat((outy, outz), 1)
        return out


class ODEFunc4(nn.Module):
    """MLP modeling the derivative of ODE system.

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

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim,augment_dim=-1,
                 time_dependent=False, non_linearity='relu'):
        super(ODEFunc4, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.input_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim=augment_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent
        self.bias_aug=nn.Parameter(0*torch.Tensor( 1 , self.hidden_dim))
        self.bias_aug2 = nn.Parameter(0 * torch.Tensor(1, self.hidden_dim))

        if time_dependent:
            self.fc1 = nn.Linear(self.hidden_dim + 1, self.hidden_dim)
        else:
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
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            t_and_z = torch.cat([t_vec, z], 1)
            # Shape (batch_size, hidden_dim)
            outy = self.fc1(t_and_z)
            outz = numpy.dot( t_and_z.detach().numpy(), - self.fc1.weight.detach().numpy()  )+ self.fc1.bias.detach().numpy()
        else:
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



class ODEFunc(nn.Module):
    """MLP modeling the derivative of ODE system.

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

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                 time_dependent=False, non_linearity='relu'):
        super(ODEFunc, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        if time_dependent:
            self.fc1 = nn.Linear(self.input_dim + 1, hidden_dim)
        else:
            self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.input_dim)

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
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(x)
        out = self.non_linearity1(out)
        out = self.fc2(out)
        out = self.non_linearity1(out)
        out = self.fc3(out)
        return out


class ODEBlock(nn.Module):
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

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """
    def __init__(self, device, odefunc, is_conv=False, tol=1e-3, adjoint=False, eval_time=1, level=2, method='dopri5'):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.eval_time = eval_time
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
            integration_time = torch.tensor([0, self.eval_time]).float().type_as(x)
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
            x_aug=  torch.cat([x, aug], 1)
        elif self.level == 4:
            aug = torch.zeros(x.shape[0], self.odefunc.hidden_dim).to(self.device)
            x_aug=  torch.cat([x, aug], 1)


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
        integration_time = torch.linspace(0., float(self.eval_time), timesteps)
        return self.forward(x, eval_times=integration_time)


class ODENet(nn.Module):
    """An ODEBlock followed by a Linear layer.

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
    def __init__(self, device, data_dim, hidden_dim, output_dim=1,
                 augment_dim=0, time_dependent=False, non_linearity='relu',
                 tol=1e-3, adjoint=False, level=0, eval_time=1,method='dopri5'):
        super(ODENet, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.time_dependent = time_dependent
        self.eval_time = eval_time
        self.tol = tol
        if level == 1:
            odefunc = ODEFunc1(device, data_dim, hidden_dim, augment_dim, time_dependent, non_linearity)
        elif level == 2:
            odefunc = ODEFunc2(device, data_dim, hidden_dim, augment_dim, time_dependent, non_linearity)
        elif level == 3:
            odefunc = ODEFunc3(device, data_dim, hidden_dim, augment_dim, time_dependent, non_linearity)
        elif level == 4:
            odefunc = ODEFunc4(device, data_dim, hidden_dim, augment_dim, time_dependent, non_linearity)
        elif level == 5:
            odefunc = ODEFuncH(device, data_dim, hidden_dim, augment_dim, time_dependent, non_linearity)
        elif level == 6:
            odefunc = ODEFuncH2(device, data_dim, hidden_dim, augment_dim, time_dependent, non_linearity)
        elif level == 7:
            odefunc = ODEFuncH3(device, data_dim, hidden_dim, augment_dim, time_dependent, non_linearity)
        elif level == 8:
            odefunc = ODEFuncH4(device, data_dim, hidden_dim, augment_dim, time_dependent, non_linearity)
        else:
            odefunc = ODEFunc(device, data_dim, hidden_dim, augment_dim, time_dependent, non_linearity)

        self.odeblock = ODEBlock(device, odefunc, tol=tol, adjoint=adjoint, eval_time=eval_time,level=level,method=method)

        if level == 3:

            self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim+self.hidden_dim,
                                      self.output_dim)
        elif level == 4:
            self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim + self.hidden_dim,
                                          self.output_dim)
        elif level > 4:
            self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim * 2,
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




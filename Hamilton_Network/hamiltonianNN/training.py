import json
import torch.nn as nn
from numpy import mean
import torch

class Trainer():
    """Class used to train ODENets, ConvODENets and ResNets.
    Parameters
    ----------
    model : one of models.ODENet, conv_models.ConvODENet, discrete_models.ResNet
    optimizer : torch.optim.Optimizer instance
    device : torch.device
    classification : bool
        If True, trains a classification model with cross entropy loss,
        otherwise trains a regression model with Huber loss.
    print_freq : int
        Frequency with which to print information (loss, nfes etc).
    record_freq : int
        Frequency with which to record information (loss, nfes etc).
    verbose : bool
        If True prints information (loss, nfes etc) during training.
    save_dir : None or tuple of string and string
        If not None, saves losses and nfes (for ode models) to directory
        specified by the first string with id specified by the second string.
        This is useful for training models when underflow in the time step or
        excessively large NFEs may occur.
    """
    def __init__(self, model, optimizer, device, classification=False,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.classification = classification
        self.device = device
        if self.classification:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.SmoothL1Loss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose

        self.histories = {'loss_history': [], 'test_loss_history': [], 'nfe_history': [],
                          'bnfe_history': [], 'total_nfe_history': [],
                          'epoch_loss_history': [], 'epoch_nfe_history': [],
                          'epoch_bnfe_history': [], 'epoch_total_nfe_history': []}
        self.buffer = {'loss': [], 'test_loss': [], 'nfe': [], 'bnfe': [], 'total_nfe': []}

        # Only resnets have a number of layers attribute
        self.is_resnet = hasattr(self.model, 'num_layers')
        
        
    def valid_accuracy(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images=images.to(self.device)
                labels =labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total
    
    
    def train(self, data_loader, test_input ,test_target ,
              num_epochs,Rrate=0.001,Arate=0.001,R='No',valid=False,valid_loader=None):
        """Trains model on data in data_loader for num_epochs.
        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        num_epochs : int
        """
        acc=0
        for epoch in range(num_epochs):
            if epoch % 51 == 50:
                for param_group in self.optimizer.param_groups:
                    param_group['lr']=param_group['lr']/10
            avg_loss = self._train_epoch(data_loader, test_input ,test_target , Rrate=Rrate, Arate=Arate, R=R)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))
            if valid:
                if self.valid_accuracy(valid_loader) > acc :
                    torch.save(self.model,'./model/check_point')
                    acc=self.valid_accuracy(valid_loader)
         
        if valid:
            self.model = torch.load('./model/check_point')

    def _train_epoch(self, data_loader, test_input , test_target , Rrate=0.001 ,Arate=0.001, R=''):
        """Trains model for an epoch.
        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        epoch_loss = 0.
        epoch_nfes = 0
        epoch_backward_nfes = 0
        test_input=test_input.to(self.device)
        test_target=test_target.to(self.device)
        for i, (x_batch, y_batch) in enumerate(data_loader):
            self.optimizer.zero_grad()

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            y_pred = self.model(x_batch)

            y_pred_test= self.model(test_input )

            # ResNets do not have an NFE attribute
            if not self.is_resnet:
                iteration_nfes = self._get_and_reset_nfes()
                epoch_nfes += iteration_nfes
            R_loss = torch.tensor(0.).to(self.device)
            A_loss = torch.tensor(0.).to(self.device)
            if R=='L1':
                for name, param in self.model.named_parameters():
                    if  'bias' not in name:
                        R_loss += torch.sum(torch.abs(param))
            elif R=='L2':
                for name, param in self.model.named_parameters():
                    if  'bias' not in name:
                        R_loss += torch.sum(torch.pow(param, 2))
            elif R == 'Mix':
                for name, param in self.model.named_parameters():
                    if  'bias' not in name:
                        if 'alpha' not in name:
                            R_loss += torch.sum(torch.pow(param, 2))
                        if 'alpha' in name:
                            A_loss += torch.sum(torch.abs(param))

            loss_test= self.loss_func(y_pred_test, test_target)
            loss = self.loss_func(y_pred, y_batch) + Rrate *R_loss/x_batch.shape[0]
            if R == 'Mix':
                loss = loss +Arate *A_loss/x_batch.shape[0]
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if not self.is_resnet:
                iteration_backward_nfes = self._get_and_reset_nfes()
                epoch_backward_nfes += iteration_backward_nfes

            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nIteration {}/{}".format(i, len(data_loader)))
                    print("Loss: {:.3f}".format(loss.item()))
                    if not self.is_resnet:
                        print("NFE: {}".format(iteration_nfes))
                        print("BNFE: {}".format(iteration_backward_nfes))
                        print("Total NFE: {}".format(iteration_nfes + iteration_backward_nfes))

            # Record information in buffer at every iteration
            self.buffer['loss'].append(loss.item())
            self.buffer['test_loss'].append(loss_test.item())
            if not self.is_resnet:
                self.buffer['nfe'].append(iteration_nfes)
                self.buffer['bnfe'].append(iteration_backward_nfes)
                self.buffer['total_nfe'].append(iteration_nfes + iteration_backward_nfes)

            # At every record_freq iteration, record mean loss, nfes, bnfes and
            # so on and clear buffer
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))
                self.histories['test_loss_history'].append(mean(self.buffer['test_loss']))

                if not self.is_resnet:
                    self.histories['nfe_history'].append(mean(self.buffer['nfe']))
                    self.histories['bnfe_history'].append(mean(self.buffer['bnfe']))
                    self.histories['total_nfe_history'].append(mean(self.buffer['total_nfe']))

                # Clear buffer
                self.buffer['loss'] = []
                self.buffer['test_loss'] = []
                self.buffer['nfe'] = []
                self.buffer['bnfe'] = []
                self.buffer['total_nfe'] = []

                # Save information in directory
                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)
                        json.dump(self.histories['test_loss_history'], f)
                    if not self.is_resnet:
                        with open('{}/nfes{}.json'.format(dir, id), 'w') as f:
                            json.dump(self.histories['nfe_history'], f)
                        with open('{}/bnfes{}.json'.format(dir, id), 'w') as f:
                            json.dump(self.histories['bnfe_history'], f)
                        with open('{}/total_nfes{}.json'.format(dir, id), 'w') as f:
                            json.dump(self.histories['total_nfe_history'], f)

            self.steps += 1

        # Record epoch mean information
        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))
        if not self.is_resnet:
            self.histories['epoch_nfe_history'].append(float(epoch_nfes) / len(data_loader))
            self.histories['epoch_bnfe_history'].append(float(epoch_backward_nfes) / len(data_loader))
            self.histories['epoch_total_nfe_history'].append(float(epoch_backward_nfes + epoch_nfes) / len(data_loader))

        return epoch_loss / len(data_loader)

    def _get_and_reset_nfes(self):
        """Returns and resets the number of function evaluations for model."""
        if hasattr(self.model, 'odeblock'):  # If we are using ODENet
            iteration_nfes = self.model.odeblock.odefunc.nfe
            # Set nfe count to 0 before backward pass, so we can
            # also measure backwards nfes
            self.model.odeblock.odefunc.nfe = 0
        else:  # If we are using ODEBlock
            iteration_nfes = self.model.odefunc.nfe
            self.model.odefunc.nfe = 0
        return iteration_nfes
    
    
    
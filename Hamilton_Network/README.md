# Hamiltonian Network

## Usage


```python
# ... Load some data ...

import torch
from hamiltonianNN.models import hamil_ODENet
from hamiltonianNN.training import Trainer

# Instantiate a model
# For regular data...
hnode = hamil_ODENet(device, data_dim=2, hidden_dim=10, time_dependent=False,level=7,augment_dim=1,
               non_linearity='relu',eval_time=1,method='leapfrog')

# Instantiate an optimizer and a trainer
optimizer = torch.optim.Adam(hnode.parameters(), lr=1e-3)
trainer = Trainer(hnode, optimizer, device)

# Train model on your dataloader
trainer.train(dataloader,test_inputs,test_targets, num_epochs=10)
```

More detailed examples and tutorials can be found in the `Hamiltonian.ipynb` and `discrete_Hamiltonian.ipynb` notebooks.


## Demos

We provide some demo notebooks that show how to reproduce some of the results and figures.

### Hamiltonian ODE 

The `Hamiltonian.ipynb` notebook contains a demo and tutorial for reproducing the experiments comparing Neural ODEs, Augmented Neural ODEs and Hamiltonian ODEs on simple 2D functions.

### Hamiltonian Net

The `discrete_Hamiltonian.ipynb` notebook contains a demo and tutorial for reproducing the experiments comparing Hamiltonian Net and ResNet on simple 2D functions.

### Hamiltonian Conv

The `Conv.ipynb` notebook contains a demo and tutorial for reproducing the experiments comparing Hamiltonian Net and ResNet on `Mnist`.

```python
# ... Load some data ...

from hamiltonianNN.conv_models import ConvHamilNet,ResNet
from hamiltonianNN.training import Trainer

# Instantiate a discret Hamitonian model
model = ConvHamilNet(device, img_size, num_filters=64,hidden_dim=128, output_dim=10,
                                   augment_dim=0,,discret=True,num_layers=100,final_time=5)

# Instantiate an ODE Hamitonian model
model = ConvHamilNet(device, img_size, num_filters=64,hidden_dim=128, output_dim=10,augment_dim=0,
                                   adjoint=True,level=7, method='leapfrog', discret=False)

# Instantiate a ResNet convolution model
model = ResNet(device, img_size, num_filters=64, output_dim=10,num_layers=6)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0.0)

# Instantiate an optimizer and a trainer
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(model, optimizer, device)

# Train model on your dataloader
trainer.train(dataloader,test_inputs,test_targets, num_epochs=10)
```


### Other experiments
`changeODE.ipynb` and `changeANODE.ipynb` compare different level ODE functions.   

`regularisation.ipynb` tests the regularizations.

`final_time.ipynb` tests the ODE nets with different final time.

`More class.ipynb` notebook contains a demo and tutorial for reproducing the experiments comparing Neural ODEs, Augmented Neural ODEs and Hamiltonian Inspired ODEs on multiple classes 2D functions.


## Citing

These codes are based on the codes of Anode, on implementing the package `torchdiffeq` from Neural ODE.

[Node](https://github.com/rtqichen/torchdiffeq).

[Anode](https://github.com/EmilienDupont/augmented-neural-odes).

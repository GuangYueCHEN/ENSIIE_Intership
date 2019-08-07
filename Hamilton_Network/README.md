# Hamiltonian Network

## Usage


```python
# ... Load some data ...

import torch
from hamiltonianNN.models import HODENet
from hamiltonianNN.training import Trainer

# Instantiate a model
# For regular data...
hnode = HODENet(device, data_dim=2, hidden_dim=16, augment_dim=1)

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

### Other experiments
`changeODE.ipynb` and `changeANODE.ipynb` compare different level ODE functions.   

`regularisation.ipynb` tests the regularizations.

`final_time.ipynb` tests the ODE nets with different final time.

`More class.ipynb` notebook contains a demo and tutorial for reproducing the experiments comparing Neural ODEs, Augmented Neural ODEs and Hamiltonian Inspired ODEs on multiple classes 2D functions.

## Data

The MNIST and CIFAR10 datasets can be directly downloaded using `torchvision` (this will happen automatically if you run the code, unless you already have those datasets downloaded). To run experiments on ImageNet, you will need to download the data from the [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) website.

## Citing

These codes are based on the codes of Anode, on implementing the package `torchdiffeq` from Neural ODE.
[Node](https://github.com/rtqichen/torchdiffeq).
[Anode](https://github.com/EmilienDupont/augmented-neural-odes).

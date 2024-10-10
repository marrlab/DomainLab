# Model Specification

Domainlab is built to easily extend to other models. There exist two options. One is to implement a new model from scratch and add it to the existing models; the other is to extend the custom model abstract class. This guide outlines the necessary steps to add a new model to the domain generalization framework for both approaches mentioned above.

## Option 1: Extend the Custom Model Class
Create a new model by extending `AModelCustom`.


Because `AModelCustom` extends `AModelClassif`, the only function a custom model needs to extend is `dict_net_module_na2arg_na`, which returns a dictionary with the key being the Pytorch module name and value being the command-line argument name. In addition, it is necessary to specify a function called `get_node_na` in the same Python file, which returns the custom algorithm builder as shown [here](../examples/models/demo_custom_model.py).  

To run the custom model, follow the examples [here](./doc_examples.md) under 'Custom algorithm defined in external python file'. It also shows an example of using additional command line arguments. 


## Option 2: Add alongside existing models
If the repository is cloned and it is possible to add files to the source code, one can extend one of the other base classes: `AModel` or `AModelClassif`.

### Step 1: Implement Required Abstract Methods
Implement all abstract methods from the base class. For `AModel`, it is required to implement the following methods:

`cal_task_loss(self, tensor_x, tensor_y)`: Computes the loss for the primary task, which for classification could be cross-entropy. <br>
`_cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None)`: Calculates the task independent regularization loss. 

### Step 2: Add Additional Arguments
If additional arguments for the model are needed,  it is possible to specify a Python file with functionality for the argument parsing, as done [here](../domainlab/models/args_vae.py). The specified function needs to be applied in the  [arg_parser.py](../domainlab/arg_parser.py) in the domainlab root directory in order to add the parameters to the argument dictionary. 

### Step 3: Create a Builder
After specifying the model and retrieving the correct parameters, we can create functionality to create the model. To do so, we need to create a NodeAlgoBuilder. For that, we create a class that inherits from `NodeAlgoBuilder` and extend the `init_business(exp)` method. We must create the trainer, model, observer, and device in this method. See [here](../domainlab/algos/builder_dann.py) for an example. 

After that, we can add the builder into the chain that creates all specified models [here](../domainlab/algos/zoo_algos.py). 
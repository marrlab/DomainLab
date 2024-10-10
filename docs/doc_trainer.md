# Model Specification

When developing new trainers, you can extend the `TrainerBasic` class. This allows you to build upon established training routines while introducing specialized behaviors tailored to your model.

### Steps to Extend `TrainerBasic`

1. **Extend the class**: Begin by inheriting from [`TrainerBasic`](../domainlab/algos/trainers/train_basic.py).

2. **Customize Key Methods**: You have several methods that you can override to customize the trainer's behavior. Here's a brief overview of what they do:

    - **`before_tr(self)`**: Set up necessary configurations or states before training begins. Useful for initial logging or setting model to train mode.
    
    - **`tr_epoch(self, epoch)`**: Define the training logic for each epoch. This is where the bulk of your model's training will be implemented.
    
    - **`before_epoch(self)`**: Prepare anything specific at the start of each epoch, like resetting counters or updating learning rate schedules.
    
    - **`after_epoch(self, epoch)`**: Typically used for logging and validation checks after each epoch.
    
    - **`tr_batch(self, tensor_x, tensor_y, tensor_d, others, ind_batch, epoch)`**: Handle the processing of each batch. This includes forward and backward propagation.
    
    - **`before_batch(self, epoch, ind_batch)`** and **`after_batch(self, epoch, ind_batch)`**: Perform actions right before and after processing a batch, respectively. Useful for implementing behaviors like batch-wise logging or applying gradients.

3. **Register Your Trainer**: Make sure the framework can utilize the new trainer. For that, it is necessary to register it in the [zoo_trainers.py](../domainlab/algos/trainers/zoo_trainer.py).

### Example Implementation

Here is a simple example of a custom trainer that logs additional details at the start of each training:

```python
class MyCustomTrainer(TrainerBasic):
    def before_tr(self):
        super().before_tr()  # Ensure to call the base method if needed
        print("Starting training session.")

    def tr_epoch(self, epoch):
        # Custom training logic for each epoch
        for ind_batch, data in enumerate(self.loader_tr):
            self.tr_batch(data, ind_batch, epoch)
        print(f"Completed epoch {epoch}")

    def tr_batch(self, data, ind_batch, epoch):
        # Process each batch
        super().tr_batch(data, ind_batch, epoch)  # Optionally call the base method
        print(f"Batch {ind_batch} of epoch {epoch} processed.")

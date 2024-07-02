# DomainLab Usage Guide

Given the repository and the dependencies are set up, here is how can use DomainLab: 


## Essential Commands

To run DomainLab, the minimum necessary parameters are:

1. **Task Specification (```--tpath``` or ```--task```):** This is to specify a task. You can eiter give a path to as Python file which specifies the task, or use a predfined set. You can find more about specifying tasks [here](./doc_tasks.md). 
2. **Test Domain (```--te_d```):** Specifies the domain(s) used for testing. Can be a single domain or multiple domains.
3. **Model Choice (```--model```):** Chooses the algorithm or model for the training (e.g., erm, diva). This also includes hyperparameters for the model, e.g., ```--gamma_d``` and ```--gamma_y``` for diva.
4. **Neural Network (```--nname``` or ```--npath```):** Specifies which neural network is used for feature extraction, either through a path or predefined options. 

### Example Command
```python3 main_out.py --te_d 0 1 2  --task=mnistcolor10 --model=diva --nname=conv_bn_pool_2  --gamma_y=7e5 --gamma_d=1e5```


## Optional Commands

### Advanced Configuration

- **Learning Rate (`--lr`):** Set the training learning rate.
- **Regularization (`--gamma_reg`):** Sets the weight of the regularization
 loss. This parameter can be configured either as
 a single value applied to individual classes,
 or using a dictionary to specify different
 weights for different models and trainers.

  - **Command Line Usage:**
    - For a single value: `python script.py --gamma_reg=0.1`
    - For multiple values: `python script.py --gamma_reg='default=0.1,dann=0.05,jigen=0.2'`

  - **YAML Configuration:**
    - For a single value:

      ```yaml
      gamma_reg: 0.1
      ```

    - For different values:

      ```yaml
      gamma_reg:
        dann: 0.05
        jigen: 0.2
        default: 0.1 # value for every other instance
      ```  

- **Early Stopping (`--es`):** Steps for early stopping.
- **Random Seed (`--seed`):** Seed for reproducibility.
- **CUDA Options (`--nocu`, `--device`):** Configure CUDA usage and device settings.
- **Generated Images (`--gen`):** Option to save generated images.
- **Model Preservation (`--keep_model`):** Choose not to delete the model at the end of training.
- **Epochs (`--epos`, `--epos_min`):** Configure maximum and minimum numbers of epochs.
- **Test Interval (`--epo_te`):** Set intervals for testing performance.
- **Hyperparameter Warm-Up (`-w` or `--warmup`):** Epochs for hyperparameter warm-up.
- **Debugging (`--debug`):** Enable debug mode for verbose output.
- **Memory Debugging (`--dmem`):** Memory usage debugging.
- **Output Suppression (`--no_dump`):** Suppress saving the confusion matrix.
- **Trainer Selection (`--trainer`):** Specify which trainer to use.
- **Output Directory (`--out`):** Directory to store outputs.
- **Data Path (`--dpath`):** Path for storing the downloaded dataset.
- **Additional Neural Network Options:**
    - Custom Argument Values (`--npath_argna2val`, `--nname_argna2val`)
    - Domain Feature Extraction Network (`--npath_dom`, `--nname_dom`)
    - Custom Algorithm Path (`--apath`)
- **Experiment and Aggregation Tags (`--exptag`, `--aggtag`):** Tags for experiment tracking and result aggregation.
- **Benchmarking and Plotting:**
    - Partial Benchmark Aggregation (`--agg_partial_bm`)
    - Plot Generation (`--gen_plots`, `--outp_dir`, `--param_idx`)

### Task-Specific Arguments

- **Batch Size (`--bs`):** Loader batch size for mixed domains.
- **Training-Validation Split (`--split`):** Proportion of training, a value between 0 and 1.
- **Training Domain (`--tr_d`):** Specify training domain names.
- **Sanity Check (`--san_check`):** Save images from the dataset as a sanity check.
- **Sanity Check Image Count (`--san_num`):** Number of images for the sanity check.
- **Logging Level (`--loglevel`):** Set the log level of the logger.
- **Shuffling (`--shuffling_off`):** Disable shuffling of the training dataloader for the dataset.

### Model-Specific Hyperparameters

#### VAE Model Parameters

- **Latent Space Dimensions (`--zd_dim`, `--zx_dim`, `--zy_dim`):** Set the size of latent spaces for domain, unobserved, and class features.
- **Topic Dimension (`--topic_dim`):** Number of topics for HDUVA.
- **Networks for HDUVA Model:**
    - Image to Topic Distribution (`--nname_encoder_x2topic_h`, `--npath_encoder_x2topic_h`)
    - Image and Topic to ZD (`--nname_encoder_sandwich_x2h4zd`, `--npath_encoder_sandwich_x2h4zd`)
- **Hyperparameters for DIVA and HDUVA (`--gamma_y`, `--gamma_d`, `--beta_t`, `--beta_d`, `--beta_x`, `--beta_y`):** Multipliers for various classifiers and loss components.

#### MatchDG Parameters

- **Cosine Similarity Factor (`--tau`):** Magnify cosine similarity.
- **Match Tensor Update Frequency (`--epos_per_match_update`):** Epochs before updating the match tensor.
- **Epochs for CTR (`--epochs_ctr`):** Total epochs for CTR.

#### Jigen Parameters

- **Permutation Settings (`--nperm`, `--pperm`, `--jigen_ppath`):** Configure image tile permutations.
- **Grid Length (`--grid_len`):** Length of image in tile units.

#### DIAL Parameters

- **Adversarial Image Generation (`--dial_steps_perturb`, `--dial_noise_scale`, `--dial_lr`, `--dial_epsilon`):** Configure parameters for generating adversarial images.


For a comprehensive understanding of all available commands, use:
```shell
python main_out.py --help
```

## Example
DomainLab comes with some minimal toy-dataset to test its basis functionality, see [a minimal subsample of the VLCS dataset](./zdata/vlcs_mini) and [an example configuration file for vlcs_mini](../examples/tasks/task_vlcs.py).

To train a domain invariant model on the vlcs_mini task:

```shell
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --config=examples/yaml/demo_config_single_run_diva.yaml
```
where `--tpath` specifies the path of a user specified python file which defines the domain generalization task [see here](../examples/tasks/task_vlcs.py), `--te_d` specifies the test domain name (or index starting from 0), `--config` specifies the configurations of the domain generalization algorithms, [see here](../examples/yaml/demo_config_single_run_diva.yaml)

In more detail, in a verbose mode without using the algorithm configuration file:

```shell
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --model=diva --gamma_y=7e5 --gamma_d=1e5 --nname=alexnet --nname_dom=conv_bn_pool_2
```

where `--model` specifies which algorithm to use, `--bs` specifies the batch size, `--debug` restrain only running for 2 epochs and save results with prefix 'debug'. For DIVA, the hyper-parameters include `--gamma_y=7e5` which is the relative weight of ERM loss compared to ELBO loss, and `--gamma_d=1e5`, which is the relative weight of domain classification loss compared to ELBO loss.
`--nname` is to specify which neural network to use for feature extraction for classification, `--nname_dom` is to specify which neural network to use for feature extraction of domains.

See [more examples](./doc_examples.md).

## Further Resources

- **Custom Neural Networks:** [Guide on specifying custom neural networks](./doc_custom_nn.md)
- **Output and Performance:** [Understanding the output structure and performance measures](./doc_output.md)
- **Extending DomainLab:** [Guide for extending or contributing to DomainLab](./doc_extend_contribute.md)

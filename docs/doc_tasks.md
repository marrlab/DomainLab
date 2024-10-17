# Task Specification

The package offers various ways to specify a domain generalization task (where to find the data, which domain to use as training, which to test) according to user's need.

For all thress ways covered below, the user has to prepare a python file to feed via argument `--tpath` (means task path) into DomainLab.  We provide example python files in our repository [see all examples here for specifying domain generalization task](https://github.com/marrlab/DomainLab/tree/master/examples/tasks) so that the user could follow the example to create their own domain generalization task specification.  We provide inline comment to explain what each line is doing, as well as below in this documentation.

## Possibility 1: Specify train and test domain dataset directly
The most straightforward way to specify a domain generalization task is, if you have already a [PyTorch Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) class for each domain: you could define a dictionary with the key being name for domain, and the value being the PyTorch Dataset you created corresponding to that domain (train and validation or only training)
[See an example python file here](https://github.com/marrlab/DomainLab/blob/master/examples/tasks/task_dset_custom.py)

To train a ERM (Emperical Risk Minimization) network on this task:
```shell
python main_out.py --te_d domain1 --tr_d domain2 domain3 --tpath=examples/tasks/task_dset_custom.py --bs=32 --model=erm --nname=conv_bn_pool_2
```
here in the example, 3 domains are defined to be named "domain1", "domain2", "domain3", `--tpath` specifies where the python file specifying the task is stored, `--bs=32` use 32 as batchsize, `erm` is our nickname for ERM, `--nname` specifies a hard coded neural network in DomainLab, to use custom neural network, see related documentation.

## Possibility 2: TaskPathList
The user need to provide a python file indicating the root directories where images (data) from  each domain are located, and for each domain, the user shall provide a plain text file indicating the file names.

An example python file can be found in the [examples directory of the repository](https://github.com/marrlab/DomainLab/blob/master/examples/tasks/demo_task_path_list_small.py) with the corresponding data be found at [the data directory of the repository](https://github.com/marrlab/DomainLab/tree/master/domainlab/zdata), where the text file indicating which file correspond to which class label (starting from 1) can be found at [example text file](https://github.com/marrlab/DomainLab/blob/master/domainlab/zdata/pacs_split/art_painting_10.txt).
The user should specify the location of the python file via command line arguments `--tpath`

```shell
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=2 --npath=examples/nets/resnet.py --model=diva --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5
```

## Possibility 3: TaskFolder
In this mode,  we assume there are structured folders where each folder contains all data from one domain, and each domain folder contains subfolders corresponding to different classes. See examples below.

### Data organization
To give an example, suppose we have a classification task to classify between car, dog, human, chair and bird and there are 3 data sources (domains) with folder name "folder_a", "folder_b" and "folder_c" respectively as shown below.

In each folder, the images are organized in sub-folders by their class. For example, "/path/to/3rd_domain/folder_c/dog" folder contains all the images of class "dog" from the 3rd domain.

It might be the case that the same classes are named differently across the different data sources. For example, in the 1st data source, the class dog is stored in a sub-folder named "hund", in the 2nd data source, the dog is stored in a sub-folder named "husky", and in the 3rd data source, the dog is stored in a sub-folder called "dog".

In that case, we can use `dict_domain_folder_name2class` to map the corresponding subfolders to the correct domain. The `dict_domain_folder_name2class` parameter is a dictionary where:

- Key: The name of a subfolder in a specific domain.
- Value: The corresponding user-defined class name.

This is also shown in the example below, where the dictionary maps the different subfolder names to 'car' and 'chair', which are the user defined classes. A more complex example of such a mapping can be found [here](../examples/tasks/task_blood3.py).

It might also be the case that some classes exist in one data source but do not exist in another data source. For example, folder "/path/to/2nd_domain/folder_b" does not have a sub-folder for the class "human". In that case, we don't add the entry into the `dict_domain_folder_name2class` mapping.

Folder structure of the 1st domain:
```text
    ── /path/to/1st_domain/folder_a
       ├── auto
       ├── hund
       ├── mensch
       ├── stuhl
       └── vogel

```
Folder structure of the 2nd domain:

```
    ── /path/to/2nd_domain/folder_b
       ├── bird
       ├── drive
       ├── sit
       └── husky
```
Folder structure of the 3rd domain:

```
    ── /path/to/3rd_domain/folder_c
        ├── dog
        ├── flying
        ├── sapiens
        ├── sofa
        └── vehicle
```

### Specify the task with domainlab API
The user is expected to implement something similar to the following code in a separate python file with a function with signature `get_task(na=None)`.
```python
import os
from torchvision import transforms

from domainlab.tasks import mk_task_folder, ImSize

# specify torchvision transformations for training
trans_tr = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# specify torchvision transformations at test/inference time
trans_te = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


chain = mk_task_folder(extensions={"domain1": "jpg", "domain2": "jpg", "domain3": "jpg"},
                       list_str_y=["chair", "car"],
                       dict_domain_folder_name2class={
                           "domain1": {"auto": "car", "stuhl": "chair"},
                           "domain2": {"vehicle": "car", "sofa": "chair"},
                           "domain3": {"drive": "car", "sit": "chair"}
                       },
                       dict_domain_img_trans={
                           "domain1": trans_tr,
                           "domain2": trans_tr,
                           "domain3": trans_tr,
                       },
                       img_trans_te=trans_te,
                       isize=ImSize(3, 224, 224),
                       dict_domain2imgroot={
                           "domain1": os.path.join("/path/to/1st_domain", "folder_a"),
                           "domain2": os.path.join("/path/to/2nd_domain", "folder_b"),
                           "domain3": os.path.join("/path/to/3rd_domain", "folder_c")},
                       taskna="task_demo")

def get_task(na=None):  # the package will call this function to get the task
    return chain
```
The domainlab function to create task in this example is `domainlab.tasks.mk_task_folder`
```python
from domainlab.tasks import mk_task_folder
print(mk_task_folder.__doc__)
```

`extensions`: a python dictionary with key as the domain name
and value as the file extensions of the image.

`list_str_y`: a python list with user defined class names where
the order of the list matters.

`dict_domain_folder_name2class`: a python dictionary, with key
as the user specified domain name, value as a dictionary to map the
sub-folder name of each domain's class folder into the user defined
common class name.

`dict_domain_img_trans`: a python dictionary with keys as the user
specified domain name, value as a user defined torchvision transform.
This feature allows carrying out different transformation (composition) to different
domains at training time.

`img_trans_te`: at test or inference time, we do not have knowledge
of domain information so only a unique transform (composition) is allowed.

`isize`: domainlab.tasks.ImSize(image channel, image height, image width)

`dict_domain2imgroot`: a python dictionary with keys as user specified domain names and values
as the absolute path to each domain's data.

`taskna`: user defined task name



## Examples

## User defined tasks

### Folder Task
#### Folder Task with implemented neural network
```shell
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --nname=alexnet --model=diva --nname_dom=alexnet --gamma_y=7e5 --gamma_d=1e5
```

#### Folder Task with externally user defined neural network
```shell
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --npath=examples/nets/resnet.py --model=diva --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5
```

### ImagePath Task
#### ImagePath Task with implemented algorithm
```shell
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=2 --nname=alexnet --model=diva --nname_dom=alexnet --gamma_y=7e5 --gamma_d=1e5
```

#### ImagePath Task with externally user defined neural network
```shell
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=2 --npath=examples/nets/resnet.py --model=diva --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5
```

## Custom algorithm defined in external python file
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=3 --apath=examples/models/demo_custom_model.py --model=custom --nname_argna2val my_custom_arg_name --nname_argna2val alexnet
```

```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=3 --apath=examples/models/demo_custom_model.py --model=custom --npath_argna2val my_custom_arg_name --npath_argna2val examples/nets/resnet.py
```


### make a sanity check for the dataset using 8 instances from each domain and from each class
```shell
python main_out.py --te_d=0 --task=mini_vlcs --debug --bs=2 --model=diva --nname=alexnet --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5 --san_check --san_num=4
```
### sanity check on only 2 train domains and 2 test domain2
```shell
python main_out.py --te_d 0 1 --tr_d 3 5 --task=mnistcolor10 --debug --bs=2 --model=erm --nname=conv_bn_pool_2 --san_check --san_num=4
```

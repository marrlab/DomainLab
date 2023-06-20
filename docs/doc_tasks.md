# Task Specification

The package offers various ways to specify the domain generalization task according to user's need.

## Specify train and test domain directly
[See example here](https://github.com/marrlab/DomainLab/blob/master/examples/tasks/task_dset_custom.py)

## TaskPathList
The user need to provide a python file indicating the root directories where images (data) from  each domain are located, and for each domain, the user shall provide a plain text file indicating the file names.

An example python file can be found in the [examples directory of the repository](https://github.com/marrlab/DomainLab/blob/master/examples/tasks/demo_task_path_list_small.py) with the corresponding data be found at [the data directory of the repository](https://github.com/marrlab/DomainLab/tree/master/data), where the text file indicating which file correspond to which class label (starting from 1) can be found at [example text file](https://github.com/marrlab/DomainLab/blob/master/data/pacs_split/art_painting_10.txt).
The user should specify the location of the python file via command line arguments `--tpath`

## TaskFolder
In this mode,  we assume there are structured folders where each folder contains all data from one domain, and each domain folder contains subfolders corresponding to different classes. See examples below.

### Data organization
To give an example, suppose we have a classification task to classify between car, dog, human, chair and bird and there are 3 data sources (domains) with folder name "folder_a", "folder_b" and "folder_c" respectively as shown below. 

In each folder, the images are organized in sub-folders by their class. For example, "/path/to/3rd_domain/folder_c/dog" folder contains all the images of class "dog" from the 3rd domain.

It might be the case that across the different data sources the same class is named differently. For example, in the 1st data source, the class dog is stored in sub-folder named 
"hund", in the 2nd data source, the dog is stored in sub-folder named "husky" and in the 3rd data source, the dog is stored in sub-folder named "dog".

It might also be the case that some classes exist in one data source but does not exist in another data source. For example, folder "/path/to/2nd_domain/folder_b" does not have a sub-folder for class "human".

Folder structure of the 1st domain:
```
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
```
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
```
from domainlab.tasks import mk_task_folder
print(mk_task_folder.__doc__)

extensions: a python dictionary with key as the domain name
and value as the file extensions of the image.

list_str_y: a python list with user defined class names where
the order of the list matters.

dict_domain_folder_name2class: a python dictionary, with key
as the user specified domain name, value as a dictionary to map the
sub-folder name of each domain's class folder into the user defined
common class name.

dict_domain_img_trans: a python dictionary with keys as the user
specified domain name, value as a user defined torchvision transform.
This feature allows carrying out different transformation (composition) to different
domains at training time.

img_trans_te: at test or inference time, we do not have knowledge
of domain information so only a unique transform (composition) is allowed.

isize: domainlab.tasks.ImSize(image channel, image height, image width)

dict_domain2imgroot: a python dictionary with keys as user specified domain names and values 
as the absolute path to each domain's data.

taskna: user defined task name
```

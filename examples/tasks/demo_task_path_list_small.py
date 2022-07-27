from torchvision import transforms
from domainlab.tasks.task_pathlist import mk_node_task_path_list
from domainlab.tasks.utils_task import ImSize


def get_task(na=None):
    node = mk_node_task_path_list(
        # specify image size, must be consistent with the transformation
        isize=ImSize(3, 224, 224),
        # specify the names for all classes to classify
        list_str_y=["dog", "elephant", "giraffe", "guitar",
                    "horse", "house", "person"],
        # give an index to each class to classify
        dict_class_label2name={"1": "dog",
                               "2": "elephant",
                               "3": "giraffe",
                               "4": "guitar",
                               "5": "horse",
                               "6": "house",
                               "7": "person"},

        #####################################################################
        # specify where to find the text file for each domain containing the
        # relative path to corresponding images

        # folder data is with respect to the current working directory

        # # 1. specify the list of images used for training set
        dict_d2filepath_list_img={
            "art_painting": "data/pacs_split/art_painting_10.txt",
            "cartoon": "data/pacs_split/cartoon_10.txt",
            "photo": "data/pacs_split/photo_10.txt",
            "sketch": "data/pacs_split/sketch_10.txt"},

        # # 2. specify the list of images used for test set
        dict_d2filepath_list_img_te={
            "art_painting": "data/pacs_split/art_painting_10.txt",
            "cartoon": "data/pacs_split/cartoon_10.txt",
            "photo": "data/pacs_split/photo_10.txt",
            "sketch": "data/pacs_split/sketch_10.txt"},

        # # 3. specify the list of images used for validation set
        dict_d2filepath_list_img_val={
            "art_painting": "data/pacs_split/art_painting_10.txt",
            "cartoon": "data/pacs_split/cartoon_10.txt",
            "photo": "data/pacs_split/photo_10.txt",
            "sketch": "data/pacs_split/sketch_10.txt"},
        #####################################################################

        # specify the (absolute or relative with respect to working directory
        # of python) root folder storing the images of each domain:
        # replace with the path where each domain of your pacs images are
        # stored.
        dict_domain2imgroot={
            'art_painting': "data/pacs_mini_10",
            'cartoon': "data/pacs_mini_10",
            'photo': "data/pacs_mini_10",
            'sketch': "data/pacs_mini_10"},
        # specify the pytorch transformation you want to apply to the image
        trans4all=transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor()]))
    return node

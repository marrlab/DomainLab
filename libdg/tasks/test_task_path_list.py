from torchvision import transforms
from libdg.tasks.task_pathlist import mk_node_task_path_list


def test_fun():
    """
    """
    from libdg.arg_parser import mk_parser_main
    from libdg.tasks.utils_task import img_loader2dir, ImSize

    node = mk_node_task_path_list(
        isize=ImSize(3, 224, 224),
        list_str_y=["dog", "elephant", "giraffe", "guitar",
                    "horse", "house", "person"],
        dict_class_label2name={"1": "dog",
                               "2": "elephant",
                               "3": "giraffe",
                               "4": "guitar",
                               "5": "horse",
                               "6": "house",
                               "7": "person"},
        dict_d2filepath_list_img={
            "art_painting": "~/Documents/datasets/pacs_split/art_painting_train_kfold.txt",
            "cartoon": "~/Documents/datasets/pacs_split/cartoon_train_kfold.txt",
            "photo": "~/Documents/datasets/pacs_split/photo_train_kfold.txt",
            "sketch": "~/Documents/datasets/pacs_split/sketch_train_kfold.txt"},

        dict_d2filepath_list_img_te={
            "art_painting": "~/Documents/datasets/pacs_split/art_painting_train_kfold.txt",
            "cartoon": "~/Documents/datasets/pacs_split/cartoon_train_kfold.txt",
            "photo": "~/Documents/datasets/pacs_split/photo_train_kfold.txt",
            "sketch": "~/Documents/datasets/pacs_split/sketch_train_kfold.txt"},

        dict_d2filepath_list_img_val={
            "art_painting": "~/Documents/datasets/pacs_split/art_painting_crossval_kfold.txt",
            "cartoon": "~/Documents/datasets/pacs_split/cartoon_train_kfold.txt",
            "photo": "~/Documents/datasets/pacs_split/photo_train_kfold.txt",
            "sketch": "~/Documents/datasets/pacs_split/sketch_train_kfold.txt"},

        dict_domain2imgroot={
            'art_painting': "~/Documents/datasets/pacs/raw",
            'cartoon': "~/Documents/datasets/pacs/raw",
            'photo': "~/Documents/datasets/pacs/raw",
            'sketch': "~/Documents/datasets/pacs/raw"},
        trans4all=transforms.ToTensor())

    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "1"])
    node.init_business(args)

    img_loader2dir(node._loader_te, list_domain_na=node.get_list_domains(),
                   list_class_na=node.list_str_y, folder="zout/test_loader/pacs", batches=10)
    img_loader2dir(node._loader_tr, list_domain_na=node.get_list_domains(),
                   list_class_na=node.list_str_y, folder="zout/test_loader/pacs", batches=10)

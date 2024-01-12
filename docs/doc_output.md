# Output structure

By default, this package generates outputs into a folder `zoutput` relative to the current working directory, which the user can alter by specifying the corresponding command line argument.

The output structure is something similar to below. ([] means the folder might or might not exist, texts inside () are comments)

```text
zoutput/
├── aggrsts (aggregation of results)
│   ├── task1_test_domain1_tagName.csv
│   ├── task2_test_domain3_tagName.csv
│   
│  
├── [gen] (counterfactual image generation, only exist for generative models with "--gen" specified)
│   ├── [task1_test_domain1]
│   
└── saved_models (persisted pytorch model)
    ├── task1_algo1_git-commit-hashtag1_seed_1_instance_wise_predictions.txt (instance wise prediction of the model)
    ├── [task1_algo1_git-commit-hashtag1_seed_1.model]  (only exist if with command line argument "--keep_model")
    ...
```

`aggrst` folder is aggregating results from several runs corresponding to the same train test split of the identical task, that means after each run, an extra line will be appended to the file corresponding to the same train test split of the identical task, so that it is convenient to compare between different algorithms and configurations upon the same train test split of the identical task.

`saved_models` folder contains instance wise prediction in a text file for each run, and potentially the persisted model on the hard disk (by default, these models will be deleted after run is completed).

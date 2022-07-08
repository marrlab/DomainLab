### Output structure
By default, libDG generate outputs to a folder called "zoutput" relative to the current working directory. The structure is something similar as follows. ([] means the folder might or might not exist)

```
zoutput/
├── aggrsts (aggregation of results)
│   ├── task1_test_domain1_tagName.csv
│   ├── task2_test_domain3_tagName.csv
│   
│  
├── [gen] (counterfactual image generation)
│   ├── task1_test_domain1
│   
└── saved_models (persisted pytorch model)
    ├── task1_algo1_git-commit-hashtag1_seed_1.model
    ├── task1_algo1_git-commit-hashtag1_seed_1.model_oracle
```

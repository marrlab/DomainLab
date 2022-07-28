# Design
![Design Diagram](libDG.svg)


# Extend or contribute with a custom domain generalization algorithm

## External extension by implementing your custom algorithm in a python file inheriting the interface of  LibDG
Look at this dummy example:
```
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --apath=examples/algos/demo_custom_algo_builder.py --aname=custom
```
where the template file corresponding to "--apath" defines a class which inherit specified interfaces.

## Internal extension by integrating an algorithm into LibDG
- implement domainlab/algos/builder_your-algorithm-name.py
- add your algorithm into domainlab/algos/zoo_algos.py by adding `chain = NodeAlgoBuilder[your-algorithm-name](chain)`
- note that all algorithms will be converted to lower case!
- make a pull request

# Credits
Contact: Xudong Sun (smilesun.east@gmail.com)

Please cite our paper if you use this code in your research:
```
@inproceedings{sun2021hierarchical,
  title={Hierarchical Variational Auto-Encoding for Unsupervised Domain Generalization},
  author={Sun, Xudong and Buettner, Florian},
  booktitle={ICLR 2021 RobustML workshop, https://arxiv.org/pdf/2101.09436.pdf},
  year={2021}
}
```

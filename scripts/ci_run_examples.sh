#!/bin/bash -x -v
set -e  # exit upon first error
# >> append content
# > erase original content


files=("docs/docDIAL.md" "docs/docIRM.md" "docs/doc_examples.md" "docs/docHDUVA.md")

for file in "${files[@]}"
do
echo "Processing $file"
# no need to remove sh_temp_algo.sh since the following line overwrite it each time
echo "#!/bin/bash -x -v" > sh_temp_algo.sh  
# remove code marker ```
# we use >> here to append to keep the header #!/bin/bash -x -v
sed -n '/```shell/,/```/ p' $file | sed '/^```/ d' >> ./sh_temp_algo.sh
cat sh_temp_algo.sh
bash -x -v -e sh_temp_algo.sh
# Add your commands to process each file here
echo "finished with $file"
done



echo "#!/bin/bash -x -v" > sh_temp_mnist.sh
sed -n '/```shell/,/```/ p' docs/doc_MNIST_classification.md | sed '/^```/ d' >> ./sh_temp_mnist.sh
bash -x -v -e sh_temp_mnist.sh
echo "mnist example done"

echo "#!/bin/bash -x -v" > sh_temp_nn.sh
sed -n '/```shell/,/```/ p' docs/doc_custom_nn.md | sed '/^```/ d' >> ./sh_temp_nn.sh
bash -x -v -e sh_temp_nn.sh
echo "arbitrary nn done"

echo "#!/bin/bash -x -v" > sh_temp_task.sh
sed -n '/```shell/,/```/ p' docs/doc_tasks.md | sed '/^```/ d' >> ./sh_temp_task.sh
bash -x -v -e sh_temp_task.sh
echo "task done"

echo "#!/bin/bash -x -v" > sh_temp_readme.sh
sed -n '/```shell/,/```/ p' README.md | sed '/^```/ d' >> ./sh_temp_readme.sh
bash -x -v -e sh_temp_readme.sh
echo "read me done"

echo "#!/bin/bash -x -v" > sh_temp_extend.sh
sed -n '/```shell/,/```/ p' docs/doc_extend_contribute.md | sed '/^```/ d' >> ./sh_temp_extend.sh
bash -x -v -e sh_temp_extend.sh

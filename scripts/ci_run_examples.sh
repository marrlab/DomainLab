#!/bin/bash -x -v
set -e  # exit upon first error
# >> append content
# > erase original content


files=("docs/docDIAL.md" "docs/docFishr.md")

for file in "${files[@]}"
do
echo "Processing $file"
echo "#!/bin/bash -x -v" > sh_temp_algo.sh  
sed -n '/```shell/,/```/ p' $file | sed '/^```/ d' >> ./sh_temp_algo.sh
bash -x -v -e sh_temp_also.sh
# Add your commands to process each file here
done



# echo "#!/bin/bash -x -v" > sh_temp_example.sh
sed -n '/```shell/,/```/ p' docs/doc_examples.md | sed '/^```/ d' >> ./sh_temp_example.sh
split -l 5 sh_temp_example.sh sh_example_split
for file in sh_example_split*;
do (echo "#!/bin/bash -x -v" > "$file"_exe && cat "$file" >> "$file"_exe && bash -x -v "$file"_exe && rm -r zoutput);
done
# bash -x -v -e sh_temp_example.sh
echo "general examples done"

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

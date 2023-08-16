sed -n '/```shell/,/```/ p' docs/doc_benchmark.md | sed '/^```/ d' > ./sh_temp_benchmark.sh
bash -x -v -e sh_temp_benchmark.sh
echo "benchmark  done"

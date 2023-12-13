#!/bin/bash
input=$1
path=$2

# test string array works
sentence="this is a story"
stringarray=($sentence)
echo "${stringarray[0]}"

while IFS= read -r line
do
  echo "$line"
  stringarray=($line)
  echo "${stringarray[0]}"
  relativepath=${stringarray[0]}
  fullpath="$path/$relativepath"
  echo "fullpath is $fullpath"
  newpath="$3/$relativepath"
  echo "newpath is $newpath"
  newpath2=`dirname "$newpath"`
  echo "newpath2 is $newpath2"
  mkdir -p "$newpath2"
  cp "$fullpath" "$newpath2"
  # cp "$fullpath" --parent "$3"

done < "$input"

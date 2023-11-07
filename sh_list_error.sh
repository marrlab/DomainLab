# find $1 -type f -print0 | xargs -0 grep -li error 
grep -B 20 -wnr "error" --group-separator="========================" $1

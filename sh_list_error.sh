# find $1 -type f -print0 | xargs -0 grep -li error 
grep -wnr "error" $1

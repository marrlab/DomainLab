# find $1 -type f -print0 | xargs -0 grep -li error 
# B means before, A means after, some erros have long stack exception message so we need at least 
# 100 lines before the error, the last line usually indicate the root cause of error
grep -B 100 -wnr "error" --group-separator="=========begin_slurm_error===============" $1 > slurm_errors.txt
cat slurm_errors.txt

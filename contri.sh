#!/bin/bash
printf "added lines %s removed lines %s net  %s\n"
git log --no-merges --pretty=format:%an --numstat | awk '/./ && !author { author = $0; next } author { ins[author] += $1; del[author] += $2 } /^$/ { author = ""; next } END { for (a in ins) { printf "%10d %10d %10d %s\n", ins[a], del[a], ins[a] - del[a], a } }' | sort -rn

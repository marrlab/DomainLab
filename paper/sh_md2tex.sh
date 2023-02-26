cp ../paper.md .
# \ need 2 times escape
# \1 means the first group (.*) between [@ and ]
sed -i -E "s/\[@(.*)\]/\\\cite\{\1\}/g" paper.md
cat paper.md
pandoc -s ./paper.md -o md_jmlr_mloss_domainlab.tex
# -E: extended regular expression
# -i: in place
cat md_jmlr_mloss_domainlab.tex

#!/bin/bash -x -v
# sudo apt-get install librsvg2-bin
java -jar ~/opt/plantuml.jar domainlab/uml/libDG.uml -tsvg
rsvg-convert -f pdf -o DomainLab.pdf domainlab/uml/libDG.svg

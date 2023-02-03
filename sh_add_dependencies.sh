poetry add $1
poetry update
poetry export -f requirements.txt --output requirements.txt --without-hashes

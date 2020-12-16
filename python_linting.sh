isort --force-single-line-imports .
autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables .
black .
isort -m 3 --trailing-comma .
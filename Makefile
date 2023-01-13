SRC := src

style:
	poetry run isort $(SRC)
	poetry run black $(SRC) --line-length 79
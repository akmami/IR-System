main: install
	python3 main.py

setup: install download train

install:
	pip3 install -r requirements.txt

download:
	python3 download.py

train:
	python3 train.py

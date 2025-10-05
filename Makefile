.PHONY: all data features mpt train bench backtest report dashboard

all: data features mpt train bench backtest report

data:
	python -m src.data

features:
	python -m src.features

mpt:
	python -m src.mpt

train:
	python -m src.train_rl

bench:
	python -m src.benchmark

backtest:
	python -m src.backtest

report:
	python -m src.evaluate

dashboard:
	streamlit run src/dashboard.py

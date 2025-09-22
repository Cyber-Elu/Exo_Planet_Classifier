PY=python
PIP=pip

install:
	$(PIP) install -r requirements.txt

fetch:
	$(PY) src/recup_data.py

train:
	$(PY) src/ml_exoplanete.py

api:
	$(PY) src/web.py

api-gunicorn:
	gunicorn -w 2 -b 0.0.0.0:5000 src.web:app

test:
	pytest -q

lint:
	python -m compileall -q src

all: install fetch train api

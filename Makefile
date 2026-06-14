.PHONY: install test train evaluate video ablations clean

install:
	pip install -e .[dev]

test:
	pytest tests/ -v

train:
	cd apps/beam_rider && python train.py

evaluate:
	cd apps/beam_rider && python evaluate.py

video:
	cd apps/beam_rider && python record_video.py

ablations:
	cd apps/beam_rider && python run_ablations.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

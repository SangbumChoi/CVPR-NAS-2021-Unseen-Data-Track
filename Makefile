ifndef data
$(error data is undefined)
endif

ifndef sample_output
$(error sample_output is undefined)
endif

ingest:
	rm -f $(sample_output)/*
	python3 ingestion_program/ingestion.py _ $(data) $(sample_output) _ _ ebnas

score:
	rm -Rf scoring_tmp
	mkdir scoring_tmp
	mkdir scoring_tmp/res
	mkdir scoring_tmp/ref
	cp -r $(data)/* scoring_tmp/ref
	cp $(sample_output)/* scoring_tmp/res
	python3 scoring_program/score.py scoring_tmp .
	rm -Rf scoring_tmp

all:
	make ingest data=$(data)
	make score data=$(data)

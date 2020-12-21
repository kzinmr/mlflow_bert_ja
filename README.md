# mlflow_bert_ja

## train
- build: `docker build -t ner-train .`
- run: `docker run --rm --gpus all  -v $(pwd)/workspace:/app/workspace ner-train`

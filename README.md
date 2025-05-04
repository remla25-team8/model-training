# Model Training
This repository will contain the ML training pipeline.
- Library dependencies can be seen in `requirements.txt`
- Data is preproccessed via our `lib-ml` library.
- The `train.py` script will execute the training steps and make the model accessible publicly via huggingface.

## Running locally
First setup environment using:
```bash
conda env -f environment.yaml
conda activate remla-model-training
```
To execute model training without uploading to the model registry run:
```bash
python train.py local-dev
```

To execute training and upload run:
```bash
python train.py production --version <version>
```
Note: You will need to have a unique version.

## Running via Docker Container
You can configure build details in the `docker-compose`. For details on the container environment look at `Dockerfile`.

To run the training via container run the following command:
```bash
docker compose up
```

## How to access the resulting model externally.
We use huggingface as the model registy so to use our model you will first need to have the huggingface_hub dependency. Then you can access the model via the link: TODO

Where <version> correlates to the release version. Latest release is TODO
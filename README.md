# WaterWise REST API ML Service

This is the REST API service for the machine learning models for the WaterWise project.

## Endpoints

- `POST /water-segmentation`: Segment/extract the water from the image. Returns an image.
- `POST /clean-water`: Predict if the water is clean or not. Returns a float value from 0 to 1. Values above 0.5 indicate clean water.
- `POST /clean-water/with-extraction`: Predict if the water is clean or not and extract the water from the image. Returns a float value from 0 to 1. Values above 0.5 indicate clean water.

## Usage

Prepare the models. Place the models in the [models](models) folder.

```bash
wget "https://example.com/models/clean-water.h5" -O "clean-water.h5"
wget "https://example.com/models/water-segmentation.pth" -O "water-segmentation.pth"
```

Install the required packages using pip.

```bash
pip install -r requirements.txt
```

Run the service.

```bash
python app.py
```

The service will be running on `http://localhost:5000`.

## Deployment

The service can be deployed using Docker. Build the Docker image.

```bash
docker build -t ml-service .
```

Run the Docker container.

```bash
docker run -p 5000:5000 ml-service
```

You can also use Buildpacks to deploy the service to Google Cloud.

```bash
gcloud builds submit --pack image=gcr.io/PROJECT_ID/ml-service
# run it on Google Cloud Run
gcloud run deploy --image gcr.io/PROJECT_ID/ml-service
```

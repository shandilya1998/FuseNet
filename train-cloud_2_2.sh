#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This scripts performs cloud training for a PyTorch model.
echo "Training cloud ML model"

# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
IMAGE_REPO_NAME=assignment3

# IMAGE_TAG: an easily identifiable tag for your docker image
IMAGE_TAG=fusenet_2_2

PROJECT_ID=celtic-buttress-291509

BUCKET_ID=assignment3cs6886

# IMAGE_URI: the complete URI location for Cloud Container Registry
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=custom_gpu_container_job_$(date +%Y%m%d_%H%M%S)

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
REGION=us-central1

# Build the docker image
docker build -f Dockerfile -t ${IMAGE_URI} ./

# Deploy the docker image to Cloud Container Registry
docker push ${IMAGE_URI}

# Submit your training job
echo "Submitting the training job"

# These variables are passed to the docker image
JOB_DIR=gs://${BUCKET_ID}/models_2_2/gpu
# Note: these files have already been copied over when the image was built

gcloud beta ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier BASIC_GPU \
    -- \
    --job-dir ${JOB_DIR} \
    --gpu True \
    --batch-size 200 \
    --learning-rate 0.005 \
    --momentum 0.5 \
    --height 224 \
    --width 224 \
    --channels 3 \
    --epochs 600 \
    --seed 42 \
    --log-interval 40 \
    --gamma 0.75 \
    --weight-decay 0.01 

# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}

# Verify the model was exported
echo "Verify the model was exported:"
gsutil ls ${JOB_DIR}/checkpoint

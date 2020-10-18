# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install pytorch
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4
# OR
# FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

WORKDIR /root

# Copies the trainer code to the docker image.
COPY src/model.py ./src/model.py
COPY src/fuse.py ./src/fuse.py
COPY src/utils.py ./src/utils.py
COPY conf/__init__.py ./conf/__init__.py
COPY conf/global_settings.py ./conf/global_settings.py
COPY requirements.txt ./requirements.txt
COPY train.py ./train.py
COPY README.md ./README.md

RUN pip install -r requirements.txt
RUN wandb login 4a30a34490a130dc21329fd04548bfd9f01cb1ec

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-u", "train.py", "gpu", "True", "-b", "128", "-warm", "1", "-lr", "0.01", "-m", "0.6", "-H", "224", "-W", "224", "-C", "3"]

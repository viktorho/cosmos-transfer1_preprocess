
## NOTICE: This setup has already refine because of the model conflict, so it will be different compare to the main repo

### Inference using conda

Please also make sure you have `conda` installed ([instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).


### [Source](https://github.com/nvidia-cosmos/cosmos-transfer1/issues/181)
The below commands create the `cosmos-transfer1` conda environment and install the dependencies for inference:
```bash
# Create the cosmos-transfer1 conda environment.
conda env create --file cosmos-transfer1.yaml
# Activate the cosmos-transfer1 conda environment.
conda activate cosmos-transfer1
# Install the dependencies.
pip install -r requirements.txt

# *ADDED THIS!*
# Install PyTorch 2.7 with CUDA 12.8 wheels
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
# *ADDED THIS*

# Install vllm
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
export VLLM_ATTENTION_BACKEND=FLASHINFER
pip install vllm==0.9.0
# Install decord
pip install decord==0.6.0
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12
# Install Transformer engine.
pip install transformer-engine[pytorch]
```

Install apex:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-build-isolation --no-cache-dir \
  --config-settings="--build-option=--cpp_ext" \
  --config-settings="--build-option=--cuda_ext" .
```

Update transformer to avoid conflict:

```
pip install transformers==4.53.0 --upgrade
```


To test the environment setup for inference run
```bash
PYTHONPATH=$(pwd) python scripts/test_environment.py
```

Check this example for testing further:
  - [Inference guide](examples/inference_cosmos_transfer1_7b.md#example-2-distilled-single-control-edge)

### Inference using docker

If you prefer to use a containerized environment, you can build and run this repo's dockerfile to get an environment with all the packages pre-installed. This environment does not use conda. So, there is no need to specify `CUDA_HOME=$CONDA_PREFIX` when invoking this repo's scripts.

This requires docker to be already present on your system with the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

```bash
docker build -f Dockerfile . -t nvcr.io/$USER/cosmos-transfer1:latest
```

Note: In case you encounter permission issues while mounting local files inside the docker, you can share the folders from your current directory to all users (including docker) using this helpful alias
```
alias share='sudo chown -R ${USER}:users $PWD && sudo chmod g+w $PWD'
```
before running the docker.

### Training

The below commands creates the `cosmos-transfer` conda environment and installs the dependencies for training. This is the same as required for inference.
```bash
# Create the cosmos-transfer1 conda environment.
conda env create --file cosmos-transfer1.yaml
# Activate the cosmos-transfer1 conda environment.
conda activate cosmos-transfer1
# Install the dependencies.
pip install -r requirements.txt
# Install vllm
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
export VLLM_ATTENTION_BACKEND=FLASHINFER
pip install vllm==0.9.0
# Install decord
pip install decord==0.6.0
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12
# Install Transformer engine.
pip install transformer-engine[pytorch]
# Install Apex for full training with bfloat16.
git clone https://github.com/NVIDIA/apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex
```

You can test the environment setup for post-training with
```bash
PYTHONPATH=$(pwd) python scripts/test_environment.py --training
```

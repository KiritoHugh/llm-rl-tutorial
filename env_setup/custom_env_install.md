https://verl.readthedocs.io/en/latest/start/install.html

- create conda env
```
conda create -n verl python==3.10
conda activate verl

# Make sure you have activated verl conda env
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh

# ... two thousands years latter

2025 年 09 月 15 日 12:13:27
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torch 2.6.0 requires nvidia-cudnn-cu12==9.1.0.70; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cudnn-cu12 9.8.0.87 which is incompatible.
vllm 0.8.5.post1 requires opentelemetry-api<1.27.0,>=1.26.0, but you have opentelemetry-api 1.37.0 which is incompatible.
vllm 0.8.5.post1 requires opentelemetry-sdk<1.27.0,>=1.26.0, but you have opentelemetry-sdk 1.37.0 which is incompatible.

I think that's ok. maybe not affect normally running.
```

- install apex

```
# change directory to anywher you like, in verl source code directory is not recommended
git clone https://github.com/NVIDIA/apex.git && \
cd apex && \
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# I only have cuda12.6 to use, but it says needing cuda12.4 to match the pytorch installed. so I choose to comment the check in setup.py to avoid runtime error raised.

```

- install verl

```
git clone https://github.com/volcengine/verl.git
cd verl
pip install --no-deps -e .
```

- quick start test

https://verl.readthedocs.io/en/latest/start/quickstart.html

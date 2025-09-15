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
```
CUDA_VISIBLE_DEVICE=0 VERL_USE_MODELSCOPE=True PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 | tee verl_demo.log
```

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_HOME="${CHECKPOINT_SAVE}/huggingface"
export TRITON_HOME="${CHECKPOINT_SAVE}/triton"
export WANDB_DIR="${CHECKPOINT_SAVE}/wandb"
export VLLM_CACHE_ROOT="${CHECKPOINT_SAVE}/vllm"
export VLLM_CONFIG_ROOT="${CHECKPOINT_SAVE}/vllm"

PROMPT_KEY=question
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=16
MAX_PROMPT_LENGTH=1536
MAX_RESPONSE_LENGTH=14848
APPLY_CHAT=True
ACTOR_MODEL_PATH=Qwen2.5-7B-Instruct
REWARD_MANAGER=hier_search
ROLLOUT_N=5
BASE_URL="http://127.0.0.1"
SEARCH_URL_MUSIQUE="${BASE_URL}:18007"
SEARCH_URL_OMNIEVAL="${BASE_URL}:18009"
SEARCH_URL_BIOASQ="${BASE_URL}:18010"
WEB_SEARCH_URL_MUSIQUE="${BASE_URL}:15005"
WEB_SEARCH_URL_OMNIEVAL="${BASE_URL}:15005"
WEB_SEARCH_URL_BIOASQ="${BASE_URL}:15005"
PROJECT_NAME=YOUR_PROJECT_NAME
EXPERIMENT_NAME=YOUR_EXPERIMENT_NAME
NNODES=1
N_GPUS_PER_NODE=8
SAVE_FREQ=10
TEST_FREQ=10
TOTAL_EPOCHS=1
WANDB_API_KEY=YOUR_WANDB_API_KEY
SAVE_PATH=$CHECKPOINT_SAVE
TRAIN_FILES=train.parquet
TEST_FILES=test.parquet

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt_key) PROMPT_KEY="$2"; shift 2;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2;;
        --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2;;
        --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2;;
        --apply_chat) APPLY_CHAT="$2"; shift 2;;
        --prompt_template_name) PROMPT_TEMPLATE_NAME="$2"; shift 2;;
        --actor_model_path) ACTOR_MODEL_PATH="$2"; shift 2;;
        --reward_manager) REWARD_MANAGER="$2"; shift 2;;
        --rollout_n) ROLLOUT_N="$2"; shift 2;;
        --project_name) PROJECT_NAME="$2"; shift 2;;
        --experiment_name) EXPERIMENT_NAME="$2"; shift 2;;
        --nnodes) NNODES="$2"; shift 2;;
        --n_gpus_per_node) N_GPUS_PER_NODE="$2"; shift 2;;
        --save_freq) SAVE_FREQ="$2"; shift 2;;
        --test_freq) TEST_FREQ="$2"; shift 2;;
        --total_epochs) TOTAL_EPOCHS="$2"; shift 2;;
        --wandb_api_key) WANDB_API_KEY="$2"; shift 2;;
        --save_path) SAVE_PATH="$2"; shift 2;;
        --train_files) TRAIN_FILES="$2"; shift 2;;
        --test_files) TEST_FILES="$2"; shift 2;;
        *)
            echo "unknown argument '$1'" >&2
            exit 1;;
    esac
done

if [ "$WANDB_API_KEY" != "None" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

ROLLOUT_SAVE_PATH=${SAVE_PATH}/rollout
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.apply_chat=${APPLY_CHAT} \
    data.filter_overlong_prompts=True \
    data.prompt_template_name=${PROMPT_TEMPLATE_NAME} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_with_search \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.search_agent_url=${SEACH_AGENT_URL} \
    actor_rollout_ref.rollout.search_url.musique=${SEARCH_URL_MUSIQUE} \
    actor_rollout_ref.rollout.search_url.omnieval=${SEARCH_URL_OMNIEVAL} \
    actor_rollout_ref.rollout.search_url.bioasq=${SEARCH_URL_BIOASQ} \
    actor_rollout_ref.rollout.web_search_url.musique=${WEB_SEARCH_URL_MUSIQUE} \
    actor_rollout_ref.rollout.web_search_url.omnieval=${WEB_SEARCH_URL_OMNIEVAL} \
    actor_rollout_ref.rollout.web_search_url.bioasq=${WEB_SEARCH_URL_BIOASQ} \
    actor_rollout_ref.rollout.max_rollout_turns=8 \
    actor_rollout_ref.rollout.supported_tools=${SUPPORTED_TOOLS} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=True \
    trainer.rollout_save_path=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs | tee ${SAVE_PATH}/run.log
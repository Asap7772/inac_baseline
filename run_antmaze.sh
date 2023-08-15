eval "$(conda shell.bash hook)"
conda activate inac_baseline

env_names=(AntMaze)
datasets=(medium_play medium_diverse)
taus=(0.01 0.1 0.33 0.5)
exp_num=0
debug=false
export PYTHONPATH=$PYTHONPATH:/nfs/kun2/users/asap7772/antmaze_gen

gpus=(0 1 2 3 4 5 6 7)
lrs=(0.001 0.0003 0.0001 0.00003)

for env_name in ${env_names[@]}; do
for dataset in ${datasets[@]}; do
for tau in ${taus[@]}; do
for lr in ${lrs[@]}; do
    which_gpu=${gpus[$exp_num % ${#gpus[@]}]}
    export CUDA_VISIBLE_DEVICES=$which_gpu

    echo -e "\nRunning $exp_num: ${env_name} with ${dataset} dataset and tau=${tau} on GPU ${which_gpu}"

    command="python run_ac_offline.py \
        --seed 0 \
        --env_name $env_name \
        --dataset $dataset \
        --discrete_control 0 \
        --state_dim 29 \
        --action_dim 8 \
        --tau $tau \
        --learning_rate $lr \
        --hidden_units 256 \
        --batch_size 256 \
        --timeout 1000 \
        --max_steps 1000000 \
        --log_interval 10000 \
        --wandb_project inac_reward_shape \
        --seed 42"
    echo $command
    if [ "$debug" = false ] ; then
        eval $command &
    else
        export WANDB_MODE=dryrun
        eval "$command"
    fi

    exp_num=$((exp_num+1))
    sleep 30

    if [ "$debug" = true ] ; then
        break
    fi
done
done
done
done
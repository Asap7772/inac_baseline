eval "$(conda shell.bash hook)"
conda activate inac_baseline
export PYTHONPATH=$PYTHONPATH:/nfs/kun2/users/asap7772/antmaze_gen
export CUDA_VISIBLE_DEVICES=0

python run_ac_offline.py --seed 0 --env_name Ant --dataset expert --discrete_control 0 --state_dim 111 --action_dim 8 --tau 0.01 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 --wandb_project ant_rerun &
sleep 30
python run_ac_offline.py --seed 0 --env_name Ant --dataset medexp --discrete_control 0 --state_dim 111 --action_dim 8 --tau 0.01 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 --wandb_project ant_rerun &
sleep 30
python run_ac_offline.py --seed 0 --env_name Ant --dataset medium --discrete_control 0 --state_dim 111 --action_dim 8 --tau 0.5 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 --wandb_project ant_rerun &
sleep 30
python run_ac_offline.py --seed 0 --env_name Ant --dataset medrep --discrete_control 0 --state_dim 111 --action_dim 8 --tau 0.5 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000 --wandb_project ant_rerun &

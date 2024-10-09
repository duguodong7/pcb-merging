### PCB ia3_base
# eval_split=test
# scale=linear+1.0+1.01+0.1
# ratio=20
# exp_name=tpa${ratio}_ia3_base

# python src/merging.py -c src/configs/ia3_base.json -i T0_held_out -m T0_held_out \
#                     -f TPA_${ratio}_${scale} \
#                     --multiple_prompts \
#                     --kwargs pretrained_model=bigscience/T0_3B split=${eval_split} \
#                         project_name=merging experiment_name=${exp_name}

# ### PCB-Merging T5
# scale=linear+0.8+2.21+0.1
# ratio=10
# exp_name=tpa${ratio}_t5_base2
# CUDA_VISIBLE_DEVICES=1 \
# python src/merging.py -c src/configs/t5_base.json -i t5_mixture -m t5_mixture \
#                     -f TPA_${ratio}_${scale} \
#                     --kwargs split=test project_name=merging experiment_name=${exp_name} &

# scale=linear+1.5+2.21+0.1
# ratio=10
# exp_name=tpa${ratio}_t5_large2
# CUDA_VISIBLE_DEVICES=2 \
# python src/merging.py -c src/configs/t5_large.json -i t5_mixture -m t5_mixture \
#                     -f TPA_${ratio}_${scale} \
#                     --kwargs split=test project_name=merging experiment_name=${exp_name} &

# scale=linear+0.8+2.21+0.1
# ratio=20
# exp_name=tpa${ratio}_t5_base
# CUDA_VISIBLE_DEVICES=3 \
# python src/merging.py -c src/configs/t5_base.json -i t5_mixture -m t5_mixture \
#                     -f TPA_${ratio}_${scale} \
#                     --kwargs split=test project_name=merging experiment_name=${exp_name} &

# scale=linear+0.8+2.21+0.1
# ratio=20
# exp_name=tpa${ratio}_t5_large
# CUDA_VISIBLE_DEVICES=4 \
# python src/merging.py -c src/configs/t5_large.json -i t5_mixture -m t5_mixture \
#                     -f TPA_${ratio}_${scale} \
#                     --kwargs split=test project_name=merging experiment_name=${exp_name} &

# scale=linear+0.8+2.21+0.1
# ratio=5
# exp_name=tpa${ratio}_t5_base
# CUDA_VISIBLE_DEVICES=5 \
# python src/merging.py -c src/configs/t5_base.json -i t5_mixture -m t5_mixture \
#                     -f TPA_${ratio}_${scale} \
#                     --kwargs split=test project_name=merging experiment_name=${exp_name} &

# scale=linear+0.8+2.21+0.1
# ratio=5
# exp_name=tpa${ratio}_t5_large
# CUDA_VISIBLE_DEVICES=6 \
# python src/merging.py -c src/configs/t5_large.json -i t5_mixture -m t5_mixture \
#                     -f TPA_${ratio}_${scale} \
#                     --kwargs split=test project_name=merging experiment_name=${exp_name} &


### TIES_Merging ia3_base
# redundant=topk20
# scale=linear+1.0+1.11+0.1
# exp_name=ties_ia3_base_${redundant}
# eval_split=test
# CUDA_VISIBLE_DEVICES=3 \
# python src/merging.py -c src/configs/ia3_base.json -i T0_held_out -m T0_held_out \
#                 -f TIES_${redundant}_mass_dis-mean_${scale} \
#                 --multiple_prompts \
#                 --kwargs pretrained_model=bigscience/T0_3B split=${eval_split} \
#                     project_name=merging experiment_name=${exp_name}

### TIES_Merging T5
# redundant=topk5
# scale=linear+0.8+2.21+0.1
# exp_name=ties_t5_base_${redundant}
# CUDA_VISIBLE_DEVICES=3 \
# python src/merging.py -c src/configs/t5_base.json -i t5_mixture -m t5_mixture \
#                 -f TIES_${redundant}_mass_dis-mean_${scale} \
#                 --kwargs split=test project_name=merging experiment_name=${exp_name} &

# redundant=topk5
# scale=linear+2.3+2.91+0.1
# exp_name=ties_t5_large_${redundant}
# CUDA_VISIBLE_DEVICES=4 \
# python src/merging.py -c src/configs/t5_large.json -i t5_mixture -m t5_mixture \
#                 -f TIES_${redundant}_mass_dis-mean_${scale} \
#                 --kwargs split=test project_name=merging experiment_name=${exp_name} &

### Task-vector T5
# exp_name=task-vector_t5_base
# CUDA_VISIBLE_DEVICES=5 \
# python src/merging.py -c src/configs/t5_base.json -i t5_mixture -m t5_mixture \
#                 -f task-vector_linear+0.1+1.21+0.1 \
#                 --kwargs split=test project_name=merging experiment_name=${exp_name} &

# exp_name=task-vector_t5_large
# CUDA_VISIBLE_DEVICES=6 \
# python src/merging.py -c src/configs/t5_large.json -i t5_mixture -m t5_mixture \
#                 -f task-vector_linear+0.1+1.21+0.1 \
#                 --kwargs split=test project_name=merging experiment_name=${exp_name} &


### Averaging / Mean  ia3_base
# python src/merging.py -c src/configs/ia3_base.json -i T0_held_out -m T0_held_out -f basic_mean \
#             --multiple_prompts \
#             --kwargs split=test project_name=merging experiment_name=mean_ia3_base

### Averaging / Mean  t5_base
# python src/merging.py -c src/configs/t5_base.json -i t5_mixture -m t5_mixture -f basic_mean \
#             --kwargs split=test project_name=merging experiment_name=mean_t5_base

### Averaging / Mean  t5_large
# python src/merging.py -c src/configs/t5_large.json -i t5_mixture -m t5_mixture -f basic_mean \
#             --kwargs split=test project_name=merging experiment_name=mean_t5_large



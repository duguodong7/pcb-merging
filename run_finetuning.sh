### Evaluate Finetuned IA3 
# for dataset in rte cb winogrande wic wsc copa h-swag story_cloze anli-r1 anli-r2 anli-r3; do
#     path_to_checkpoint=checkpoints/T0_3B-finetuned/${dataset}/best.pt   #pretrained.pt
#     exp_name=${dataset}_finetune_test_multi_props
#     python src/inference.py -c src/configs/ia3_base.json -i ${dataset} --multiple_prompts \
#                                 --kwargs split=test project_name=test experiment_name=${exp_name} \
#                                     checkpoint_to_directly_load_model=${path_to_checkpoint} 
# done

### Evaluate Zeroshot IA3 
# rte, cb, winogrande, wic, wsc, copa, h-swag, story_cloze, anli-r1, anli-r2, anli-r3
# path_to_checkpoint=checkpoints/T0_3B-finetuned/pretrained.pt
# dataset=T0_held_out
# exp_name=T0_held_out_zeroshot_test_multi_props
# python src/inference.py -c src/configs/ia3_base.json -i ${dataset} --multiple_prompts \
#                             --kwargs split=test project_name=test experiment_name=${exp_name} \
#                                 checkpoint_to_directly_load_model=${path_to_checkpoint} \


### Evaluate Zeroshot T5-base
# for dataset in paws qasc quartz story_cloze wiki_qa winogrande wsc; do
#     exp_name=${dataset}_zeroshot_test
#     python src/inference.py -c src/configs/t5_base.json -i ${dataset} \
#                                 --kwargs split=test project_name=test experiment_name=${exp_name} 
# done

### Evaluate Zeroshot T5-large
# for dataset in paws qasc quartz story_cloze wiki_qa winogrande wsc; do
#     exp_name=${dataset}_zeroshot_test
#     python src/inference.py -c src/configs/t5_large.json -i ${dataset} \
#                                 --kwargs split=test project_name=test experiment_name=${exp_name} 
# done

### Evaluate Finetuned T5-base
# for dataset in paws qasc quartz story_cloze wiki_qa winogrande wsc; do
#     path_to_checkpoint=checkpoints/t5-base-finetuned/${dataset}/best_model.pt   #pretrained.pt
#     exp_name=${dataset}_finetune_test
#     python src/inference.py -c src/configs/t5_base.json \
#                                 -i ${dataset} \
#                                 --kwargs checkpoint_to_directly_load_model=${path_to_checkpoint} \
#                                 split=test project_name=test experiment_name=${exp_name}
# done

### Evaluate Finetuned T5-large
# for dataset in paws qasc quartz story_cloze wiki_qa winogrande wsc; do
#     path_to_checkpoint=checkpoints/t5-large-finetuned/${dataset}/best_model.pt   #pretrained.pt
#     exp_name=${dataset}_finetune_test
#     python src/inference.py -c src/configs/t5_large.json \
#                                 -i ${dataset} \
#                                 --kwargs checkpoint_to_directly_load_model=${path_to_checkpoint} \
#                                 split=test project_name=test experiment_name=${exp_name}
# done


### Training IA3
# exp_name=train_ia3_rte2
# CUDA_VISIBLE_DEVICES=0 \
# python src/training.py -c src/dataset/rte/config.json -k train_batch_size=32 \
#                 gradient_accumulation_factor=4 project_name=training \
#                 experiment_name=${exp_name} train_dataset=rte train_dataset_mixture=None num_batches=100


# Train T5-base Models  # paws qasc quartz story_cloze wiki_qa winogrande wsc;   rte cb copa wic
# 4,5,6,7    0,1,2,3
# train_dataset=paws     
# exp_name=paws
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python src/training.py -c src/configs/t5_base.json \
#                        -k train_batch_size=256 \
#                        gradient_accumulation_factor=1 \
#                        project_name=training \
#                        experiment_name=${exp_name} \
#                        train_dataset=${train_dataset} \
#                        train_dataset_mixture=None \
#                        inference_dataset=${train_dataset} \
#                        inference_dataset_mixture=None \
#                        num_batches=750000 \
#                        world_size=4 \
#                        should_save_every_checkpoint=False \

# Train T5-large Models  # paws qasc quartz story_cloze wiki_qa winogrande wsc;   rte cb copa wic
# 4,5,6,7    0,1,2,3
# train_dataset=paws     
# exp_name=paws
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python src/training.py -c src/configs/t5_large.json \
#                        -k train_batch_size=64 \
#                        gradient_accumulation_factor=1 \
#                        project_name=training \
#                        experiment_name=${exp_name} \
#                        train_dataset=${train_dataset} \
#                        train_dataset_mixture=None \
#                        inference_dataset=${train_dataset} \
#                        inference_dataset_mixture=None \
#                        num_batches=750000 \
#                        world_size=4 \
#                        should_save_every_checkpoint=False \



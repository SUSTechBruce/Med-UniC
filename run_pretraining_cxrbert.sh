#!/bin/bash

export pregenerated_data=$6
export nas_output_dir=$7
export cache_dir=${8}
export epochs=${9}
export gradient_accumulation_steps=${10}
export train_batch_size=${11}
export learning_rate=${12}
export max_seq_length=${13}
export model=${14}
export model_type=${15}
export load_model=${16}
export warmup_steps=${18}
export mask_ratio=${19}
export tensor_dir=${20}


echo "\$1: $1"
echo "\$2: $2"
echo "\$3: $3"
echo "\$4: $4"
echo "\$5: $5"
echo "\$6: $6"
echo "\$7: $7"
echo "\$8: $8"
echo "\$9: $9"
echo "\$10: ${10}"
echo "\$11: ${11}"
echo "\$12: ${12}"
echo "\$13: ${13}"
echo "\$14: ${14}"
echo "\$15: ${15}"
echo "\$16: ${16}"
echo "\$17: ${17}"
echo "\$18: ${18}"
echo "\$19: ${19}"
echo "\$20: ${20}"


python -m torch.distributed.launch \
    --nproc_per_node=${17} \
    --nnodes=$5 \
    --node_rank=$4 \
    --master_addr=$1 \
    --master_port=$2 \
    pretraining_CXRBert.py \
    --pregenerated_data ${pregenerated_data} \
    --nas_output_dir ${nas_output_dir} \
    --cache_dir ${cache_dir} \
    --epochs ${epochs} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --train_batch_size ${train_batch_size} \
    --learning_rate ${learning_rate} \
    --max_seq_length ${max_seq_length} \
    --model ${model} \
    --load_model ${load_model} \
    --warmup_steps ${warmup_steps} \
	--masked_lm_prob ${mask_ratio} \
   --tensor_dir ${tensor_dir} \
    --do_lower_case --fp16 


#!/bin/bash

export en_img_path=${6}
export en_text_csv_path=${7}
export sp_img_path=${8}
export sp_text_csv_path=${9}
export model=${10}
export batch_size=${11}
export lr=${12}
export max_epochs=${13}
export max_seq_length=${14}
export freeze_layers=${15}
export nas_output_dir=${16}
export gradient_accumulation_steps=${17}
export cache_dir=${18}
export text_dim=${19}



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


python -m torch.distributed.launch \
    --nproc_per_node=${5} \
    --nnodes=$4 \
    --node_rank=$3 \
    --master_addr=$1 \
    --master_port=$2 \
    main_mmodal_local.py \
    --en_img_path ${en_img_path} \
    --en_text_csv_path ${en_text_csv_path} \
    --sp_img_path ${sp_img_path} \
    --sp_text_csv_path ${sp_text_csv_path} \
    --model ${model} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --max_epochs ${max_epochs} \
    --max_seq_length ${max_seq_length} \
    --freeze_layers ${freeze_layers} \
    --nas_output_dir ${nas_output_dir} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --cache_dir ${cache_dir} \
    --text_dim ${text_dim} \
    --do_lower_case --fp16


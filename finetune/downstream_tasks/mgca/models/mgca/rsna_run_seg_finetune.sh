# CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50

# CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 15
# CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 2500
# CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 1500
# CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 15
# CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 2500
# CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 1500
# CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 15
# CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 2500
# CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 1500

CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 15
CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 1500
CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 2500
CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 3500

CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 15
CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 1500
CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 2500
CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 3500

CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 15
CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 1500
CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 2500
CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 3500

CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 0.01 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 15 --batch_size 4
CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 0.01 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 1500 --batch_size 4
CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 0.01 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 2500 --batch_size 2
CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 0.01 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 3500 --batch_size 2

CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 0.1 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 15 --batch_size 4
CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 0.1 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 1500 --batch_size 4
CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 0.1 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 2500 --batch_size 2
CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 0.1 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 3500 --batch_size 2

CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 1 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 15 --batch_size 4
CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 1 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 1500 --batch_size 4
CUDA_VISIBLE_DEVICES=0 python camp_vit_segmenter.py --gpus 1 --data_pct 1 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 2500 --batch_size 2
CUDA_VISIBLE_DEVICES=1 python camp_vit_segmenter.py --gpus 1 --data_pct 1 --dataset siim --learning_rate 5e-4 --epochs 100 --seed 3500 --batch_size 2
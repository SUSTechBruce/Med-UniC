CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset chexpert --data_pct 0.01 --batch_size 32
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset chexpert --data_pct 0.1 --batch_size 32
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset chexpert --data_pct 1 --batch_size 32




CUDA_VISIBLE_DEVICES=0 python medklip_finetuner.py --gpus 1 --dataset covidx --data_pct 0.01
CUDA_VISIBLE_DEVICES=0 python medklip_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1
CUDA_VISIBLE_DEVICES=0 python medklip_finetuner.py --gpus 1 --dataset covidx --data_pct 1


CUDA_VISIBLE_DEVICES=0 python MRM_finetuner.py --gpus 1 --dataset rsna --data_pct 0.1

CUDA_VISIBLE_DEVICES=0 python camp_vit_finetuner.py --gpus 1 --dataset chexpert --data_pct 0.01 --batch_size 8 
CUDA_VISIBLE_DEVICES=1 python camp_vit_finetuner.py --gpus 1 --dataset chexpert --data_pct 0.1 --batch_size 8
CUDA_VISIBLE_DEVICES=0 python camp_vit_finetuner.py --gpus 1 --dataset chexpert --data_pct 1 --batch_size 8

CUDA_VISIBLE_DEVICES=1 python camp_vit_finetuner.py --gpus 1 --dataset rsna --data_pct 0.01 --batch_size 8 --seed 2500
CUDA_VISIBLE_DEVICES=0 python camp_vit_finetuner.py --gpus 1 --dataset rsna --data_pct 0.1 --batch_size 8 --seed 2500
CUDA_VISIBLE_DEVICES=0 python camp_vit_finetuner.py --gpus 1 --dataset rsna --data_pct 1 --batch_size 8 --seed 2500

CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 1 --batch_size 8
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1 --batch_size 8
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.01 --batch_size 8


CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1 --batch_size 8 --seed 42
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1 --batch_size 8 --seed 2500
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1 --batch_size 4 --seed 1500

CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.01 --batch_size 8 --seed 42
CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.01 --batch_size 8 --seed 2500
CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.01 --batch_size 4 --seed 1500

CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1 --batch_size 8 --seed 42
CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1 --batch_size 8 --seed 2500
CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1 --batch_size 4 --seed 1500

CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 1 --batch_size 8 --seed 42
CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 1 --batch_size 8 --seed 2500
CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 1 --batch_size 4 --seed 1500


CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset rsna --data_pct 1 --batch_size 8
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset rsna --data_pct 0.1 --batch_size 8
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset rsna --data_pct 0.01 --batch_size 8




CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.01 --batch_size 8 --seed 3500
CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.01 --batch_size 8 --seed 2500
CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.01 --batch_size 4 --seed 1500
CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.01 --batch_size 4 --seed 42

CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1 --batch_size 8 --seed 3500
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1 --batch_size 8 --seed 2500
CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1 --batch_size 4 --seed 1500
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 0.1 --batch_size 4 --seed 42

CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 1 --batch_size 8 --seed 3500
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 1 --batch_size 8 --seed 2500
CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 1 --batch_size 4 --seed 1500
CUDA_VISIBLE_DEVICES=0 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 1 --batch_size 4 --seed 42



CUDA_VISIBLE_DEVICES=1 python mgca_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 15
CUDA_VISIBLE_DEVICES=1 python mgca_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 1500
CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 3500

CUDA_VISIBLE_DEVICES=1 python mgca_segmenter.py --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 15
CUDA_VISIBLE_DEVICES=1 python mgca_segmenter.py --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 1500
CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 3500

CUDA_VISIBLE_DEVICES=1 python mgca_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 8 --learning_rate 5e-4 --max_epochs 50 --seed 15
CUDA_VISIBLE_DEVICES=1 python mgca_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 1500
CUDA_VISIBLE_DEVICES=1 python mgca_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 4 --learning_rate 5e-4 --max_epochs 50 --seed 3500



CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.01 --learning_rate 2.5e-4 --batch_size 4 --seed 42 --max_epochs 50
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.01 --learning_rate 2.5e-4 --batch_size 2 --seed 1500 --max_epochs 50
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.01 --learning_rate 2.5e-4 --batch_size 2 --seed 2500 --max_epochs 50
CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.01 --learning_rate 2.5e-4 --batch_size 2 --seed 2500 --max_epochs 50

CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.1 --learning_rate 5e-4 --batch_size 4 --seed 42 --max_epochs 50
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.1 --learning_rate 5e-4 --batch_size 4 --seed 1500 --max_epochs 50
CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.1 --learning_rate 5e-4 --batch_size 2 --seed 2500 --max_epochs 50
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.1 --learning_rate 5e-4 --batch_size 2 --seed 2500 --max_epochs 50


CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4 --batch_size 4 --seed 42 --max_epochs 50
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4 --batch_size 4 --seed 1500 --max_epochs 50
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4 --batch_size 2 --seed 2500 --max_epochs 50
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4 --batch_size 2 --seed 2500 --max_epochs 50
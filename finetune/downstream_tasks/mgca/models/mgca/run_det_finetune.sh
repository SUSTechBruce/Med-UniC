CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name MGCA --devices 1 --dataset object_cxr --data_pct 0.01 --learning_rate 2.5e-4 --batch_size 4
# CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --dataset object_cxr --data_pct 0.1 --learning_rate 5e-4
# CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4
# CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --dataset rsna --data_pct 0.01 --learning_rate 2e-4 --batch_size 8
# CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --dataset rsna --data_pct 0.1 --learning_rate 5e-4
# CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --dataset rsna --data_pct 1 --learning_rate 5e-4

CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.01 --learning_rate 2.5e-4 --batch_size 2 --seed 42
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.01 --learning_rate 2.5e-4 --batch_size 2 --seed 1500
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.01 --learning_rate 2.5e-4 --batch_size 2 --seed 2500

CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.1 --learning_rate 5e-4 --batch_size 2 --seed 42
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.1 --learning_rate 5e-4 --batch_size 2 --seed 1500
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 0.1 --learning_rate 5e-4 --batch_size 2 --seed 2500

CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4 --batch_size 2 --seed 42
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4 --batch_size 2 --seed 1500
CUDA_VISIBLE_DEVICES=1 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4 --batch_size 2 --seed 2500


CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4 --batch_size 2 --seed 2500 --max_epochs 50


CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --model_name mmodal --devices 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4 --batch_size 4 --seed 42 --max_epochs 50
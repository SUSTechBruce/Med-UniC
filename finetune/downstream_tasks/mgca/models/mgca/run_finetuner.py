import os

if __name__ == '__main__':
    os.system("CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset rsna --data_pct 0.01 --seed 2020")
    
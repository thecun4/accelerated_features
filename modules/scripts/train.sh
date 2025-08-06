cd .modules/training
python modules/training/train_hpatches.py \
    --hpatches_root_path /home/mol/work/hpatches-sequences-release \
    --ckpt_save_path ./checkpoints \
    --batch_size 4 \
    --n_steps 1000 \
    --lr 1e-4
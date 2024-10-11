
export CUDA_VISIBLE_DEVICES=3,4,7

bash /home/users/junming.wang/SD-origin/tools/dist_test.sh \
    /home/users/junming.wang/SD-origin/projects/configs/sparsedrive_small_stage2.py \
    /home/users/junming.wang/SD-origin/sparsedrive_stage2.pth \
    2 \
    --deterministic \
    --eval bbox \
    --result_file /home/users/xingyu.zhang/workspace/SD-origin/scripts/work_dirs/sparsedrive_small_stage2/results_eval.pkl
    
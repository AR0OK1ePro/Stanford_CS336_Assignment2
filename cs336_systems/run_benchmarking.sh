uv run nsys profile --trace=cuda,osrt,nvtx \
    --pytorch=autograd-nvtx -o result \
    --python-backtrace=cuda \ 
    python cs336_systems/benchmarking_script.py \
    --model_size 'small' \
    --vocab 10000 \
    --context_length 256 \
    --d_model 768 \
    --d_ff 3072 \
    --num_layers 12 \
    --num_heads 12 \
    --batch_size 4 \
    --num_steps 1 \
    --num_warmups 5 \
    --num_trials 10 \
    --data_path ../TinyStoriesV2-GPT4-train.npy \
    --output ./results/benchmark_results.json
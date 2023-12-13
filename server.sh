mlvcli config -r $R_TOKEN -w $W_TOKEN 
echo "mixed_precision: '$MIXED_PRECISION'" >> runpod.yaml
echo "set auth config completed!"
cp runpod.yaml /root/.cache/huggingface/accelerate/default_config.yaml
python server.py

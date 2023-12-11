mlvcli config -r $R_TOKEN -w $W_TOKEN 
echo "set auth config completed!"
RUN cp runpod.yaml /root/.cache/huggingface/accelerate/default_config.yaml
python server.py

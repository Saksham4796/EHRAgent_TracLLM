import os
attr_type = 'self_citation' # 'tracllm', 'vanilla_perturb' or 'self_citation'
dataset_name= 'nq-poison' #choose from 'musique','narrativeqa','qmsum', 'nq-poison', 'hotpotqa-poison', and 'msmarco-poison'
prompt_injection_attack = 'default' 
inject_times = 5 # number of injected instructions
model_name = "llama3.1-8b"
data_num = 100 # number of evaluation data points
explanation_level = "segment" # 'sentence','paragraph' or 'segment'
K = 5 # find top-K most important text segments
gpu_id = 0 # GPU ID

if not os.path.exists('./out'):
    os.makedirs('./out')

cmd = f'nohup python -u main.py \
--dataset_name {dataset_name} \
--attr_type {attr_type} \
--K {K} \
--explanation_level {explanation_level} \
--prompt_injection_attack {prompt_injection_attack} \
--inject_times {inject_times} \
--model_name {model_name} \
--gpu_id {gpu_id} \
--verbose 0 \
--data_num {data_num} \
> ./out/{dataset_name}_{model_name}_{prompt_injection_attack}_{inject_times}_{attr_type}_{K}.out &'
os.system(cmd)

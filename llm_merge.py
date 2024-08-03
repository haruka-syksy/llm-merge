import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from dotenv import load_dotenv

load_dotenv(verbose=True)

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

HUGGINGFACE_HUB_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
# VOLUME_PATH = os.environ.get("VOLUME_PATH")
print(HUGGINGFACE_HUB_TOKEN)

import time

import torch
from torch.quantization import quantize_dynamic, QuantType, quantize
from transformers import AutoTokenizer 
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from mlx_lm import load, generate, stream_generate
from datasets import load_dataset

from huggingface_hub import login
login(token=HUGGINGFACE_HUB_TOKEN)

import gc

gc.collect()
torch.cuda.empty_cache() 

def save_chat_vecter(model_path):
    # path = f"{VOLUME_PATH}/huggingface-cache/' #任意で設定してください。
    path = './huggingface-cache/'
    os.environ['HF_HUB_CACHE'] = path
    os.environ['HF_HOME'] = path

    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        cache_dir=path
    )
    inst_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        cache_dir=path
    )

    cp_model = AutoModelForCausalLM.from_pretrained(
        "tokyotech-llm/Swallow-MS-7b-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        cache_dir=path
    )

    base_tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        cache_dir=path
        )
    inst_tokenizer = AutoTokenizer.from_pretrained(
        "tokyotech-llm/Swallow-MS-7b-v0.1",
        cache_dir=path
        )
    cp_tokenizer = AutoTokenizer.from_pretrained(
        "tokyotech-llm/Swallow-MS-7b-v0.1",
        cache_dir=path
        )

    # 除外対象
    skip_layers = ["model.embed_tokens.weight", "lm_head.weight"]

    # #k: 層 291
    # #v: パラメータ
    # # a 0.0-1.0
    for k, v in tqdm(cp_model.state_dict().items()):
        # layernormも除外
        if (k in skip_layers) or ("layernorm" in k):
            continue
        chat_vector = inst_model.state_dict()[k] - base_model.state_dict()[k]
        # 重み調整するとき
        # new_v = v + weight_list[k]*chat_vector.to(v.device)
        new_v = v + chat_vector.to(v.device)
        v.copy_(new_v)

    cp_tokenizer.save_pretrained(f"{model_path}")
    cp_model.save_pretrained(f"{model_path}")

if __name__=='__main__':
    start = time.time()
    # model_name = f"{VOLUME_PATH}chat_model"
    model_name = './chat_model'
    save_chat_vecter(model_name)


    """
    以下作成したモデルを読み込んで実験
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")


    # # https://huggingface.co/datasets/elyza/ELYZA-tasks-100
    # ds = load_dataset("elyza/ELYZA-tasks-100")

    # for i, test in enumerate(ds['test']):
    #     print(test)

    # 入力データの準備
    # prompt = "東京工業大学の主なキャンパスは、"
    prompt = "日本の首都はどこですか？"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    input_ids = input_ids.to('mps')

    # トークン生成
    tokens = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=500,
        temperature=0.99,
        top_p=0.95,
        do_sample=True
    )

    # 出力をデコード
    output_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(output_text)
    print(f'かかった時間（秒） {time.time()-start}')

    end = time.time()
    os.makedirs('logs/', exist_ok=True)
    with open(f'logs/log{end}.txt', 'w') as f:
        f.write(f'model_name:  {model_name}\n')
        f.write(f'prompt: {prompt}\n')
        f.write(f'かかった時間（秒） {end-start}\n')
        f.write(f'{output_text}')

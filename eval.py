import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from dotenv import load_dotenv
load_dotenv(verbose=True)
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

import torch
from groq import Groq
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from mlx_lm import load, generate, stream_generate
from datasets import load_dataset
import csv

"""
https://huggingface.co/tokyotech-llm/Swallow-MS-7b-v0.1
model_name = "tokyotech-llm/Swallow-MS-7b-v0.1"

https://huggingface.co/DataPilot/Llama3-ArrowSE-8B-v0.3
model_name = "DataPilot/Llama3-ArrowSE-8B-v0.3"
"""

class LLMEvaluator:
    """
    LLMの回答を評価するためのクラスです。

    Attributes:
        groq_api_key (str): Groq APIキー
        llm_model_name (str, optional): 評価に使用するLLMモデル名。デフォルトは"DataPilot/ArrowPro-7B-KUJIRA"
        device (str, optional): 使用するデバイス ("cuda" or "cpu")。デフォルトは"cuda"

    Example:
        >>> evaluator = LLMEvaluator(groq_api_key="your-groq-api-key")
        >>> score = evaluator.evaluate(
        ...     question="日本の首都はどこですか？",
        ...     correct_answer="東京都",
        ...     criteria="回答が「東京都」であれば5点、そうでなければ1点",
        ... )
        >>> print(score) 
        4
    """

    def __init__(
        self,
        groq_api_key: str,
        llm_model_name: str = "DataPilot/ArrowPro-7B-KUJIRA",
        device: str = 'mps'#"cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.groq_api_key = groq_api_key
        self.client = Groq(api_key=groq_api_key)
        self.llm_model_name = llm_model_name
        self.device = 'mps'#device
        # print("tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name,
            # cache_dir=self.path
            )
        print("llm_model")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.bfloat16, device_map="mps"
            # device_map="auto"
            # , use_safetensors=True,local_files_only=True
        )#.to(self.device)
        # self.llm_model, self.tokenizer = load(self.llm_model_name)
        print(self.tokenizer)
        self.llm_model.eval()

    def __call__(
        self,
        question: str,
        correct_answer: str,
        criteria: str,
    ) -> int:
        """
        モデルの回答を評価します。

        Args:
            question (str): 質問文
            correct_answer (str): 正解
            criteria (str): 採点基準

        Returns:
            int: 1~5点のスコア
        """
        return self.evaluate(question, correct_answer, criteria)

    def evaluate(
        self,
        question: str,
        correct_answer: str,
        criteria: str,
    ) -> int:
        """
        モデルの回答を評価します。

        Args:
            question (str): 質問文
            correct_answer (str): 正解
            criteria (str): 採点基準

        Returns:
            int: 1~5点のスコア
        """
        model_answer = self._generate_answer(question)
        print('model_answer', model_answer)
        prompt = f"""問題, 正解例, 採点基準, 言語モデルが生成した回答が与えられます。

# 指示
「採点基準」と「正解例」を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 問題
{question}

# 正解例
{correct_answer}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{criteria}

# 言語モデルの回答
{model_answer}

# ここまでが'言語モデルの回答'です。回答が空白だった場合、1点にしてください。

# 指示
「採点基準」と「正解例」を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。
"""
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model="llama3-70b-8192"
        )
        try:
            score = int(chat_completion.choices[0].message.content)
        except ValueError:
            print(
                f"Error: Could not convert score to int. Response was: {chat_completion.choices[0].message.content}"
            )
            score = 1  # エラー時は1点とする
        return score, model_answer
    
    def _generate_answer(self, question: str) -> str:
        """
        質問に対してLLMモデルに回答を生成させます。

        Args:
            question (str): 質問文

        Returns:
            str: モデルが生成した回答
        """
        prompt = self._build_prompt(question)
        print(prompt)
        print('prompt')
        # input_ids = self.tokenizer.encode(
        #     prompt, add_special_tokens=True, return_tensors="pt"
        # ).to(self.device)
        # print('input_ids')
        # output = self.llm_model.generate(
        #     input_ids,
        #     max_new_tokens=10, #500,
        #     temperature=1,
        #     top_p=0.95,
        #     do_sample=True,
        #     pad_token_id=self.tokenizer.eos_token_id,
        #     # eos_token_id=self.tokenizer.eos_token_id,
        #     attention_mask=input_ids.ne(self.tokenizer.eos_token_id)
        # )
        # print('output')
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # inputs = self.tokenizer.encode_plus(
        #     prompt,
        #     return_tensors='pt',
        #     padding=True,
        #     truncation=True,
        #     max_length=50
        # )

        # input_ids = inputs['input_ids']
        
        # # Create the attention mask
        # attention_mask = input_ids.ne(self.tokenizer.eos_token_id)

        # # Generate the answer
        # output = self.llm_model.generate(
        #     input_ids=input_ids,
        #     max_new_tokens=500, # Set to 500 if needed
        #     temperature=1,
        #     top_p=0.95,
        #     do_sample=True,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     attention_mask=attention_mask
        # )
        # answer = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        # print('answer')
        # self.tokenizer.pad_token = self.tokenizer.eos_token

        # inputs = self.tokenizer(
        #     prompt, 
        #     return_tensors="pt", 
        #     add_special_tokens=False,
        #     padding=True,
        #     truncation=True,
        #     max_length=50
        #     )
        # input_ids = inputs.input_ids
        # attention_mask = inputs.attention_mask

        # input_ids = input_ids.to('mps')

        # # トークン生成
        # tokens = self.llm_model.generate(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=500,
        #     temperature=0.99,
        #     top_p=0.95,
        #     do_sample=True,
        #     pad_token_id=self.tokenizer.eos_token_id,
        # )

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # トークン生成
        tokens = self.llm_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=500,
            temperature=0.99,
            top_p=0.95,
            do_sample=True
        )


        # 出力をデコード
        answer = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        return answer

    def _build_prompt(self, user_query: str) -> str:
        """ユーザーの質問からLLMへのプロンプトを作成する"""
        sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"
        template = """[INST] <<SYS>>
{}
<</SYS>>

{}[/INST]"""
        return template.format(sys_msg, user_query)


if __name__=='__main__':
    start = time.time()
    model_path = "./chat_model"
    os.makedirs('logs/', exist_ok=True)
    with open(f"logs/log{start}.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(['model_path', 'index', 'score', 'tmp_end'])

    
    evaluator = LLMEvaluator(
        groq_api_key=GROQ_API_KEY,
        llm_model_name=model_path
        )
    print('init終わり')

    # """
    # お試し
    # """
    # index = 0
    # task_input = "日本の首都はどこですか？"
    # task_output = "東京都"
    # task_eval_aspect = "回答が「東京都」であれば5点、そうでなければ1点"

    # score, model_answer = evaluator.evaluate(
    #     question=task_input,
    #     correct_answer=task_output,
    #     criteria=task_eval_aspect,
    # )
    # print(score) 
    # tmp_end = time.time()
    # print('かかった秒数', tmp_end-former_end)
    # with open(f'log{tmp_end}.txt', 'w') as f:
    #     f.write(f'model_path:  {model_path}\n')
    #     f.write(f'input: {task_input}\n')
    #     f.write(f'output: {model_answer}\n')
    #     f.write(f'かかった時間（秒） {tmp_end-former_end}\n')
    #     f.write(f'{model_answer}')
    

    # with open(f"log{start}.csv", "a") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([model_path, index, score, tmp_end])
    former_end = time.time()

    """
    ELYZA-tasks-100
    """
    # https://huggingface.co/datasets/elyza/ELYZA-tasks-100
    ds = load_dataset("elyza/ELYZA-tasks-100")

    for index, test in enumerate(ds['test']):
        print(test)
        task_input = test['input']
        task_output = test['output']
        task_eval_aspect = test['eval_aspect']

        score, model_answer = evaluator.evaluate(
            question=task_input,
            correct_answer=task_output,
            criteria=task_eval_aspect,
        )
        print(score) 
        tmp_end = time.time()
        print('かかった秒数', tmp_end-former_end)
        
        with open(f'logs/log{tmp_end}.txt', 'w') as f:
            f.write(f'model_path:  {model_path}\n')
            f.write(f'input: {task_input}\n')
            f.write(f'かかった時間（秒）: {tmp_end-former_end}\n')
            f.write(f'score: {score}')
            f.write(f'{model_answer}')

        

        with open(f"logs/log{start}.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([model_path, index, score, tmp_end])
        former_end = time.time()

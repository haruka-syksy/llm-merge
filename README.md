# llm merge automation by black-box optimization using an Ising machine

## llm_merge.py
chat vectorの足し引きをして、モデルを保存する

## eval.py
モデルの評価(elyza/ELYZA-tasks-100)

## main.py 

イジングマシンによるブラックボックス最適化
### TODO
- 重みとバイナリ変数を相互変換する関数作成
- モデルの作成と評価部分の関数作成(llm_merge.py, eval.pyを利用)


## 環境変数(localでは.envに記載)
- HUGGINGFACE_HUB_TOKEN: [hugging_face](https://huggingface.co/)のtoken
- GROQ_API_KEY: [groq](https://groq.com/)のtoken 
- FIXSTARS_AMPLIFY_TOKEN: [Fixstars Amplify](https://amplify.fixstars.com/ja/) のtoken
- VOLUME_PATH: 保存したい記憶装置のパス(任意)
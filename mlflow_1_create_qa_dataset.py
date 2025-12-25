
# mlflow_1_create_qa_dataset.py
"""
MLflow GenAI 工作流程 - 階段 1: 建立 Q&A 評估資料集
"""

import mlflow
from mlflow.genai import datasets
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# 階段 1: 環境設定
# ============================================================================

EXPERIMENT_NAME = "mlflow"
MODEL_NAME = "gpt-oss-20b-local"
EXPERIMENT_IDS = ["3"]
DATASET_NAME = "qa_eval_dataset"


mlflow.set_experiment(EXPERIMENT_NAME)

def create_qa_dataset(experiment_ids: list[str], dataset_name: str, dataset_records: list[dict] | None = None):
    """建立 Q&A 評估資料集"""
    
    if dataset_records is None:
        dataset_records = [
            {
                "inputs": {"question": "法國的首都是什麼?"},
                "expectations": {"expected_response": "巴黎"},
            },
            {
                "inputs": {"question": "第一個製造飛機的人是誰?"},
                "expectations": {"expected_response": "萊特兄弟"},
            },
            {
                "inputs": {"question": "誰寫了羅密歐與茱麗葉?"},
                "expectations": {"expected_response": "威廉·莎士比亞"},
            },
            {
                "inputs": {"question": "水的化學式是什麼?"},
                "expectations": {"expected_response": "H2O"},
            },
            {
                "inputs": {"question": "太陽系中最大的行星是什麼?"},
                "expectations": {"expected_response": "木星"},
            },
            {
                "inputs": {"question": "地球上最高的山峰是什麼?"},
                "expectations": {"expected_response": "聖母峰"},
            },
            {
                "inputs": {"question": "世界上人口最多的國家是哪個?"},
                "expectations": {"expected_response": "印度"},
            },
            {
                "inputs": {"question": "DNA 代表什麼?"},
                "expectations": {"expected_response": "脫氧核糖核酸"},
            },
            {
                "inputs": {"question": "誰是現代物理學之父?"},
                "expectations": {"expected_response": "艾伯特·愛因斯坦"},
            },
            {
                "inputs": {"question": "美國獨立戰爭在哪一年開始?"},
                "expectations": {"expected_response": "1776年"},
            },
            {
                "inputs": {"question": "光的速度是多少?"},
                "expectations": {"expected_response": "每秒 299,792,458 公尺"},
            },
            {
                "inputs": {"question": "莎翁四大悲劇中哪部最著名?"},
                "expectations": {"expected_response": "哈姆雷特"},
            },
            {
                "inputs": {"question": "中國最長的河流是什麼?"},
                "expectations": {"expected_response": "長江"},
            },
            {
                "inputs": {"question": "人類有多少條染色體?"},
                "expectations": {"expected_response": "46 條"},
            },
            {
                "inputs": {"question": "奧林匹克運動會每幾年舉辦一次?"},
                "expectations": {"expected_response": "4 年"},
            },
            {
                "inputs": {"question": "最古老的文明之一是什麼?"},
                "expectations": {"expected_response": "美索不達米亞文明"},
            },
            {
                "inputs": {"question": "聲音在空氣中的速度大約是多少?"},
                "expectations": {"expected_response": "每秒 343 公尺"},
            },
            {
                "inputs": {"question": "誰發明了電話?"},
                "expectations": {"expected_response": "亞歷山大·格拉漢·貝爾"},
            },
            {
                "inputs": {"question": "歐盟有多少個成員國?"},
                "expectations": {"expected_response": "27 個"},
            },
            {
                "inputs": {"question": "火星上有多少個衛星?"},
                "expectations": {"expected_response": "2 個"},
            },
            {
                "inputs": {"question": "誰寫了《1984》?"},
                "expectations": {"expected_response": "喬治·歐威爾"},
            },
            {
                "inputs": {"question": "最大的哺乳動物是什麼?"},
                "expectations": {"expected_response": "藍鯨"},
            },
            {
                "inputs": {"question": "印度洋和太平洋的分界線在哪?"},
                "expectations": {"expected_response": "東經 120 度"},
            },
            {
                "inputs": {"question": "古埃及最著名的陵墓是什麼?"},
                "expectations": {"expected_response": "吉薩大金字塔"},
            },
            {
                "inputs": {"question": "量子力學的創始人是誰?"},
                "expectations": {"expected_response": "馬克斯·普朗克"},
            },
            {
                "inputs": {"question": "地球的自轉週期是多少?"},
                "expectations": {"expected_response": "24 小時"},
            },
            {
                "inputs": {"question": "誰是現代化學之父?"},
                "expectations": {"expected_response": "安托萬·拉瓦節"},
            },
            {
                "inputs": {"question": "最小的素數是什麼?"},
                "expectations": {"expected_response": "2"},
            },
            {
                "inputs": {"question": "人體最長的骨骼是什麼?"},
                "expectations": {"expected_response": "股骨"},
            },
            {
                "inputs": {"question": "地球自太陽有多遠?"},
                "expectations": {"expected_response": "約 1.5 億公里"},
            },
        ]
    
    # 建立資料集
    dataset = datasets.create_dataset(
        experiment_id=experiment_ids,
        name=dataset_name,
    )
    
    # 合併記錄
    dataset.merge_records(dataset_records)
    
    print(f"✅ 資料集已建立: {dataset_name}")
    return dataset

create_qa_dataset(experiment_ids=EXPERIMENT_IDS, dataset_name=DATASET_NAME)
print("="*70)
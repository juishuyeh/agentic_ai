from typing import Any
import openai
import mlflow
from mlflow.genai.scorers import Correctness, Guidelines
from dotenv import load_dotenv
from mlflow.genai import scorer, datasets

load_dotenv()


EXPERIMENT_NAME = "mlflow"
MODEL_NAME = "gemma-3-12b"
PREDICT_MODEL_NAME = "gemma-3-4b"
PROMPTS = "prompts:/qa-agent-user-prompt@latest"
TEMPERATURE = 0.5
DATASET_ID = "d-c420717904b24382b6313a023802d3dd"

mlflow.set_experiment(EXPERIMENT_NAME)
dataset = datasets.get_dataset(dataset_id=DATASET_ID)


@scorer
def is_concise(outputs: str) -> bool:
    """檢查答案是否簡潔 (少於 5 個詞)"""
    return len(outputs.split()) <= 5


@scorer
def is_match_dataset_result(outputs: str, expectations: dict[str, Any]) -> bool:
    """
    檢查答案是否與資料集答案完全匹配

    Args:
        outputs: 模型輸出
        expectations: 包含 expected_response 或其他期望值的字典
    """
    # 從 expectations 字典中取得 expected_response
    expected_response = expectations.get("expected_response", "")
    return outputs.strip() == expected_response.strip()


def predict_fn(question: str) -> str:
    prompt = mlflow.genai.load_prompt(PROMPTS)
    # prompt.format() 返回的是完整的 messages 列表，直接使用即可
    formatted_prompt = prompt.format(question=question)

    # 檢查 formatted_prompt 是列表還是字串
    if isinstance(formatted_prompt, list):
        messages = formatted_prompt
    else:
        messages = [{"role": "user", "content": formatted_prompt}]

    response = openai.OpenAI().chat.completions.create(
        model=PREDICT_MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content


def main():
    # 3.Run the evaluation
    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=[
            Correctness(model=f"openai:/{MODEL_NAME}"),
            Guidelines(
                name="is_traditional_chinese",
                guidelines="The answer must be in Traditional Chinese",
                model=f"openai:/{MODEL_NAME}",
            ),
            is_concise,
            is_match_dataset_result,
        ],
    )

    print(results)


if __name__ == "__main__":
    main()

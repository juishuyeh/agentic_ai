import mlflow
import openai
from typing import Any
from mlflow.genai.optimize import GepaPromptOptimizer
from dotenv import load_dotenv
from mlflow.genai import scorer, datasets


load_dotenv()

mlflow.set_experiment("mlflow")

MODEL='gemma-3-27b'
PROMPTS = "prompts:/qa-agent-user-prompt@latest"
TEMPERATURE = 0.5
DATASET_ID = 'd-c420717904b24382b6313a023802d3dd'

# Step 1: Register your initial prompt
# 建立 user prompt，如果已存在則不重複建立
def create_or_load_prompt():
    user_prompt = mlflow.genai.load_prompt(PROMPTS)
    if not user_prompt:
        mlflow.genai.register_prompt(
            name="qa-agent-user-prompt",
            template="""You are a helpful assistant. Answer questions concisely in Traditional Chinese.
Question: {{question}}
Answer:""",
        )
        print("Prompt not found. Registered a new prompt.")
    return mlflow.genai.load_prompt(PROMPTS)



# Step 2: Create a prediction function

def predict_fn(question: str, **kwargs) -> str:
    # Load prompt from registry
    user_prompt = create_or_load_prompt()
    formatted_prompt = user_prompt.format(question=question, **kwargs)

    # 檢查 formatted_prompt 是列表還是字串
    if isinstance(formatted_prompt, list):
        messages = formatted_prompt
    else:
        messages = [{"role": "user", "content": formatted_prompt}]

    response = openai.OpenAI().chat.completions.create(
        model=MODEL, messages=messages, temperature=TEMPERATURE,
    )
    return response.choices[0].message.content


# Step 3: Prepare training data
train_data = datasets.get_dataset(dataset_id=DATASET_ID)


# Step 4: Prepare scorer
@scorer
def exact_match(outputs: str, expectations: dict[str, Any]) -> bool:
    expected_response = expectations.get('expected_response', '')
    return outputs.strip() == expected_response.strip()


@scorer
def is_concise(outputs: str) -> bool:
    """檢查答案是否簡潔 (少於 5 個詞)"""
    return len(outputs.split()) <= 5


scorers: list = [exact_match, is_concise]
user_prompt = create_or_load_prompt()

if __name__ == "__main__":
    # Step 5: Optimize the prompt
    # result = mlflow.genai.optimize_prompts(...)
    result = mlflow.genai.optimize_prompts(
        predict_fn=predict_fn,
        train_data=train_data,
        prompt_uris=[user_prompt.uri],
        optimizer=GepaPromptOptimizer(
            reflection_model=f"openai:/{MODEL}",
            max_metric_calls=100,
        ),
        scorers=scorers,
    )

    # Step 6: Use the optimized prompt
    # Access optimized prompts
    for prompt in result.optimized_prompts:
        print(f"Name: {prompt.name}")
        print(f"Version: {prompt.version}")
        print(f"Template: {prompt.template}")
        print("\n")
        print(f"URI: {prompt.uri}")

    # Check optimizer used
    print(f"Optimizer: {result.optimizer_name}")

    # View evaluation scores (if available)
    print(f"Initial score: {result.initial_eval_score}")
    print(f"Final score: {result.final_eval_score}")


    ###### # You can also directly test the predict_fn with a question
    # predict_fn(
    #     question="What is MLflow",
    # )
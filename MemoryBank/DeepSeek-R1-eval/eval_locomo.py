import os
import sys

os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'

API_KEY = ""
BASE_URL = ""
#from memory_bank.memory_retrieval.forget_memory import MemoryForgetterLoader
from memory_bank.memory_retrieval.local_doc_qa_locomo import LocalMemoryRetrieval, load_memory_file
import json, time
import openai
import pathlib
PROJECT = pathlib.Path(__file__).resolve().parents[1]


def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def clean_reasoning_model_output(text):
    # 清理推理模型输出中的<think>标签，适配推理模型（如o1系列）的输出格式
    if not text:
        return text

    import re
    # 移除<think>...</think>标签及其内容
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 清理可能产生的多余空白行
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    # 移除开头和结尾的空白
    cleaned_text = cleaned_text.strip()

    return cleaned_text


# ---- OpenAI Client ----
class OpenAIClient:
    # openai调用
    def __init__(self, api_key, base_url=None):
        openai.api_key = api_key
        openai.api_base = base_url

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):

        import time, random
        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                raw_content = response.choices[0].message.content.strip()
                # 自动清理推理模型的<think>标签
                cleaned_content = clean_reasoning_model_output(raw_content)
                return cleaned_content
            except Exception as e:
                print(f"尝试 {attempt + 1}/{max_retries} 失败：{e}")
                if attempt < max_retries - 1:
                    # 指数退避 + 抖动
                    delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"等待 {delay:.2f} 秒后重试...")
                    time.sleep(delay)
                else:
                    print(f"所有重试都失败了")
                    print(f"Error calling OpenAI API: {e}")
                    return "Error: Could not get response from LLM."


print("开始初始化记忆系统...")
client = OpenAIClient(api_key=API_KEY, base_url=BASE_URL)
memory_system = LocalMemoryRetrieval()
memory_system.init_cfg(embedding_model="all-MiniLM-L6-v2", embedding_device="cpu",
                       language='en')

json_file = PROJECT/"eval_data"/"locomo10.json"
USER_ID = "LoCoMo"
PROCESS_INDEX = 9

vs_path = None
print(f"正在为{USER_ID}加载记忆文件")
vs_path, loaded_files = memory_system.init_memory_vector_store(
                          filepath=json_file,
                          vs_path=vs_path,
                          user_name=PROCESS_INDEX)
if vs_path is None:
    print("记忆加载失败")
    sys.exit(1)

vector_store = memory_system.load_memory_index(vs_path)
print("记忆索引加载完成，开始处理问题！")

# -----------------------------------------

qa_file = PROJECT/"eval_data"/"locomo10.json"
results = []
output_file = "all_loco_results.json"
with open(qa_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)
sample = dataset[PROCESS_INDEX]
sample_id = sample.get("sample_id", "unknown_sample")
conversation_data = sample["conversation"]
speaker_a = conversation_data["speaker_a"]
speaker_b = conversation_data["speaker_b"]
qa_pairs = sample["qa"]
qa_count = len(qa_pairs)
for qa_idx, qa in enumerate(qa_pairs):
    print(f"  处理问答 {qa_idx + 1}/{qa_count}")
    question = qa["question"]
    original_answer = qa.get("answer", "")
    category = qa["category"]
    if category == 5:
        print(f"category == 5 跳过！！")
        continue
    evidence = qa.get("evidence", "")
    if (original_answer == ""):
        original_answer = qa.get("adversarial_answer", "")

    # -----------------------------
    related_memories, memory_dates = memory_system.search_memory(
        query=question, vector_store=vector_store)
    system_prompt = (
        f"""You are role-playing as {speaker_b} in a conversation with the user is playing is  {speaker_a}. 
        You possess memory, emotions, and preferences. 
        Based on the current user's question, you begin to recall past conversations between the two of you, and the most relevant [memory] is: 
        "{related_memories}\nThe date of this [memory] is {memory_dates}."
        Any content referring to 'User' in the prompt refers to {speaker_a}'s content, and any content referring to 'AI'or 'assiant' refers to {speaker_b}'s content.
        Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n
        When the question is: \"What did the charity race raise awareness for?\", you should not answer in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: \"mental health\", as this is more concise."""
    )
    user_prompt = (f"""the question is: {question}\n
                    Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n
                    Please only provide the content of the answer, without including 'answer:'\n
                    For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. 
                    For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" 
                    Only provide one year, date, or time, without any extra responses.\n
                    If the question is about the duration, answer in the form of several years, months, or days.\n
                    Generate answers primarily composed of concrete entities, such as Mentoring program, school speech, etc""")

    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat_completion(
        model="deepseek-reasoner",
        messages=message)
    print(f"AI回答：{response}")

    results.append({
        "sample_id": sample_id,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "question": question,
        "system_answer": response,
        "original_answer": original_answer,
        "category": category,
        "evidence": evidence,
        "timestamp": get_timestamp(),
    })

try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"样本 {PROCESS_INDEX + 1} 处理完成，结果已保存到 {output_file}")
except Exception as e:
    print(f"保存结果时出错：{e}")
# -----------------------------------------


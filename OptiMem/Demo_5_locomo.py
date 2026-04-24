
import os
import time
import json

os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'

from memoryos import Memoryos

USER_ID = "demo_5_user"
ASSISTANT_ID = "demo_5_assistant"
API_KEY = ""
BASE_URL = ""
DATA_STORAGE_PATH = "./demo_5"
# LLM_MODEL = "deepseek-reasoner"
LLM_MODEL = "deepseek-chat"


def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def process_conversation(conversation_data):
    processed = []
    speaker_a = conversation_data["speaker_a"]
    speaker_b = conversation_data["speaker_b"]

    session_keys = [key for key in conversation_data.keys() if
                    key.startswith("session_") and not key.endswith("_date_time")]

    for session_key in session_keys:
        # 获取当前session发生的时间戳
        timestamp_key = f"{session_key}_date_time"
        timestamp = conversation_data.get(timestamp_key, "")

        for dialog in conversation_data[session_key]:  # 当前session内的所有对话
            speaker = dialog["speaker"]
            text = dialog["text"]
            # 处理图像数据
            if "blip_caption" in dialog and dialog["blip_caption"]:
                text = f"{text} (image description: {dialog['blip_caption']})"

            if speaker == speaker_a:
                processed.append({
                    "user_input": text,
                    "agent_response": "",
                    "timestamp": timestamp
                })
            else:
                if processed:
                    processed[-1]["agent_response"] = text
                else:
                    processed.append({
                        "user_input": "",
                        "agent_response": text,
                        "timestamp": timestamp
                    })
    return processed


def demo_5():
    print("开始运行demo_5文件")

    print("初始化MemoryOS中...")
    try:
        memo=Memoryos(
            user_id=USER_ID,
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            data_storage_path=DATA_STORAGE_PATH,
            llm_model=LLM_MODEL,
            assistant_id=ASSISTANT_ID,
            short_term_capacity=7,
            mid_term_capacity=200,
            mid_term_similarity_threshold=0.6,
            mid_term_heat_threshold=5.0,
            long_term_knowledge_capacity=100,
            retrieval_queue_capacity=10,
            embedding_model_name="all-MiniLM-L6-v2"
        )
    except Exception as e:
        print(f"错误：{e}")
        return
    print("创建成功，开始测试！")
    # 直接处理整个数据集，不需要命令行参数
    print("开始处理整个locomo10数据集...")

    # 加载locomo10数据集
    try:
        with open("locomo10.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
    except FileNotFoundError:
        print("错误：找不到 locomo10.json 文件，请确保文件在当前目录中")
        return
    except Exception as e:
        print(f"加载数据集时出错：{e}")
        return

    # 设置固定的输出文件名
    output_file = "all_loco_results.json"
    results = []
    total_samples = len(dataset)
    start_idx = 0
    for idx in range(start_idx, len(dataset)):
        sample = dataset[idx]
        print(f"正在处理样本 {idx + 1}/{total_samples}: {sample.get('sample_id', 'unknown')}")
        sample_id = sample.get("sample_id", "unknown_sample")
        conversation_data = sample["conversation"]
        qa_pairs = sample["qa"]
        # 处理对话数据
        processed_dialogs = process_conversation(conversation_data)
        if not processed_dialogs:
            print(f"样本 {sample_id} 没有有效的对话数据，跳过")
            continue

        speaker_a = conversation_data["speaker_a"]
        speaker_b = conversation_data["speaker_b"]
        # ----------------------------------
        # 把对话存储在记忆模块中
        for dialog in processed_dialogs:
            memo.add_memory(user_input=dialog['user_input'],
                            agent_response=dialog['agent_response'],
                            timestamp=dialog['timestamp'],
                            speaker_a=speaker_a, speaker_b=speaker_b)
        # add和search分离
        # break
        # -----------------------------------

        # 处理QAs
        qa_count = len(qa_pairs)
        for qa_idx, qa in enumerate(qa_pairs):
            print(f"  处理问答 {qa_idx + 1}/{qa_count}")
            question = qa["question"]
            original_answer = qa.get("answer", "")
            category = qa["category"]
            # ------------去掉category5-------------
            if category == 5:
                print(f"  跳过问答 {qa_idx + 1}/{qa_count} (category 5 - 超出对话范围)")
                continue
            # ---------------------------------
            evidence = qa.get("evidence", "")
            if (original_answer == ""):
                original_answer = qa.get("adversarial_answer", "")
            # 检索并生成回答
            system_answer = memo.get_response(query=question, speaker_a=speaker_a, speaker_b=speaker_b)

            results.append({
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": question,
                "system_answer": system_answer,
                "original_answer": original_answer,
                "category": category,
                "evidence": evidence,
                "timestamp": get_timestamp(),
            })

        # 每处理完一个样本就保存一次结果（实时保存）
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"样本 {idx + 1} 处理完成，结果已保存到 {output_file}")
            return
        except Exception as e:
            print(f"保存结果时出错：{e}")
        break  # 处理完一个样本就跳出


if __name__ == "__main__":
    demo_5()

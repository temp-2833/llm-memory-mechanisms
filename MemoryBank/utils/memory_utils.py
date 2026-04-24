import time, json
import gradio as gr
import sys
import os
from llama_index import GPTSimpleVectorIndex
# 把上级memory-bank目录加入系统路径，便于导入自定义模块
bank_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../memory_bank')
sys.path.append(bank_path)

from memory_bank.build_memory_index import build_memory_index
memory_bank_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../memory_bank')
sys.path.append(memory_bank_path)
from memory_bank.summarize_memory import summarize_memory


def enter_name_llamaindex(name, memory, data_args, update_memory_index=True):
    # 简化版用户登录入口，用于命令行
    user_memory_index = None
    if name in memory.keys():
        user_memory = memory[name]
        memory_index_path = os.path.join(data_args.memory_basic_dir,f'memory_index/{name}_index.json')
        if not os.path.exists(memory_index_path) or update_memory_index:
            print(f'Initializing memory index {memory_index_path}...')
            build_memory_index(memory,data_args,name=name)
        if os.path.exists(memory_index_path):
            user_memory_index = GPTSimpleVectorIndex.load_from_disk(memory_index_path)
            print(f'Successfully load memory index for user {name}!')
            
        return f"Wellcome Back, {name}！",user_memory,user_memory_index
    else:
        memory[name] = {}
        memory[name].update({"name":name}) 
        return f"Welcome new user{name}！I will remember your name and call you by your name in the next conversation",memory[name],user_memory_index


def summarize_memory_event_personality(data_args, memory, user_name):
    # 对指定的用户记忆进行总结
    if isinstance(data_args,gr.State):
        data_args = data_args.value
    if isinstance(memory,gr.State):
        memory = memory.value
    memory_dir = os.path.join(data_args.memory_basic_dir,data_args.memory_file)
    memory = summarize_memory(memory_dir,user_name,language=data_args.language)
    user_memory = memory[user_name] if user_name in memory.keys() else {}
    return user_memory#, user_memory_index 


def save_local_memory(memory,b,user_name,data_args):
    # 把最新的单论对话保存到本地记忆文件
    if isinstance(data_args,gr.State):
        data_args = data_args.value
    if isinstance(memory,gr.State):
        memory = memory.value

    memory_dir = os.path.join(data_args.memory_basic_dir,data_args.memory_file)
    date = time.strftime("%Y-%m-%d", time.localtime())
    if memory[user_name].get("history") is None:
        memory[user_name].update({"history":{}})
    if memory[user_name]['history'].get(date) is None:
        memory[user_name]['history'][date] = []

    memory[user_name]['history'][date].append({'query':b[-1][0],'response':b[-1][1]})
    json.dump(memory,open(memory_dir,"w",encoding="utf-8"),ensure_ascii=False)
    return memory
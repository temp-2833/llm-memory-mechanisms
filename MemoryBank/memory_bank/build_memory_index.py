from llama_index import SimpleDirectoryReader, Document
from llama_index import GPTTreeIndex, GPTSimpleVectorIndex
from llama_index.indices.composability import ComposableGraph
import json, openai
from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI
# from langchain.llms import AzureOpenAI,OpenAIChat
import os
openai.api_key = os.environ["OPENAI_API_KEY"]
# os.environ["OPENAI_API_BASE"] = openai.api_base


def generate_memory_docs(data,language):
    all_user_memories = {}
    for user_name, user_memory in data.items():
        # print(user_memory)
        all_user_memories[user_name] = []
        if 'history' not in user_memory.keys():
            continue
        # 针对当前用户的记忆
        for date, content in user_memory['history'].items():
            memory_str = f'日期{date}的对话内容为：' if language=='cn' else f'Conversation on {date}：'
            for dialog in content:
                query = dialog['query']
                response = dialog['response']
                memory_str += f'\n{user_name}：{query.strip()}'
                memory_str += f'\nAI：{response.strip()}'
            memory_str += '\n'
            if 'summary' in user_memory.keys():
                if date in user_memory['summary'].keys():
                    summary = f'时间{date}的对话总结为：{user_memory["summary"][date]}' if language=='cn' else f'The summary of the conversation on {date} is: {user_memory["summary"][date]}'
                    memory_str += summary
            all_user_memories[user_name].append(Document(memory_str))
    return all_user_memories


index_set = {}


def build_memory_index(all_user_memories,data_args,name=None):
    # 构建内存索引
    # 生成记忆文档
    all_user_memories = generate_memory_docs(all_user_memories,data_args.language)
    # LLM
    ds_llm = ChatOpenAI(model="deepseek-reasoner",
                        openai_api_base="",
                        openai_api_key="",
                        temperature=0.7)
    llm_predictor = LLMPredictor(llm=ds_llm)
    # 最大输入(token数)
    max_input_size = 4096
    # 输出tokens数量
    num_output = 256
    # 最大块重叠大小
    max_chunk_overlap = 20
    # prompt
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    for user_name, memories in all_user_memories.items():
        # print(all_user_memories[user_name])
        if name:
            if user_name != name:
                continue
        print(f'为用户 {user_name} 构建索引')
        cur_index = GPTSimpleVectorIndex.from_documents(memories,service_context=service_context)
        index_set[user_name] = cur_index
        os.makedirs(f'../memories/memory_index/llamaindex',exist_ok=True)
        cur_index.save_to_disk(f'../memories/memory_index/llamaindex/{user_name}_index.json')
 

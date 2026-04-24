# -*- coding: utf-8 -*-
import sys 
sys.path.append('../memory_bank')  # 添加模块搜索路径
# from azure_client import LLMClientSimple
import openai, json, os
import argparse
import copy


class LLMClientSimple:

    def __init__(self,gen_config=None):
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = "https://api.deepseek.com"
        self.disable_tqdm = False  # 是否禁用进度条显示
        self.gen_config = gen_config  # 文本生成的配置参数

    def generate_text_simple(self,prompt,prompt_num,language='en'):
        # 生成文本的主函数
        self.gen_config['n'] = prompt_num  # 生成文本的数量
        retry_times,count = 5,0  # 重试次数
        response = None
        while response is None and count<retry_times:  # 重试循环
            try:
                request = copy.deepcopy(self.gen_config)
                # print(prompt)
                if language=='cn':  # 如果是中文，则用中文prompt
                    message = [
                    {"role": "system", "content": "以下是一个人类和一个聪明、懂心理学的AI助手之间的对话记录。"},
                    {"role": "user", "content": "你好！请帮我对对话内容归纳总结"},
                    {"role": "system", "content": "好的，我会尽力帮你的。"},
                    {"role": "user", "content": f"{prompt}"}]
                else:
                    message = [
                    {"role": "system", "content": "Below is a transcript of a conversation between a human and an AI assistant that is intelligent and knowledgeable in psychology."},
                    {"role": "user", "content": "Hello! Please help me summarize the content of the conversation."},
                    {"role": "system", "content": "Sure, I will do my best to assist you."},
                    {"role": "user", "content": f"{prompt}"}]
                # 调用API
                response = openai.ChatCompletion.create(
                    **request, messages=message)
            except Exception as e:
                print(e)
                if 'This model\'s maximum context' in str(e):
                        cut_length = 1800-200*(count)
                        print('max context length reached, cut to {}'.format(cut_length))
                        prompt = prompt[-cut_length:]
                        response=None
                count+=1
        if response:
            task_desc = response['choices'][0]['message']['content'] #[response['choices'][i]['text'] for i in range(len(response['choices']))]
        else:
            task_desc = ''
        return task_desc


chatgpt_config = {"model": "deepseek-reasoner",
        "temperature": 0.7,
        "max_tokens": 400,
        "top_p": 1.0
        # "frequency_penalty": 0.4,
        # "presence_penalty": 0.2,
        # "stop": ["<|im_end|>", "¬人类¬"]
        }
# 创建上面那个类的实例
llm_client = LLMClientSimple(chatgpt_config)


def summarize_content_prompt(content,user_name,boot_name,language='en'):
    # 构造可以总结content的prompt
    prompt = '请总结以下的对话内容，尽可能精炼，提取对话的主题和关键信息。如果有多个关键事件，可以分点总结。对话内容：\n' if language=='cn' else 'Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. If there are multiple key events, you may summarize them separately. Dialogue content:\n'
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        prompt += f"\n{user_name}：{query.strip()}"
        prompt += f"\n{boot_name}：{response.strip()}"
    prompt += ('\n总结：' if language=='cn' else '\nSummarization：')
    return prompt


def summarize_overall_prompt(content,language='en'):
    # 构造可以总结历史事件的prompt
    prompt = '请高度概括以下的事件，尽可能精炼，概括并保留其中核心的关键信息。概括事件：\n' if language=='cn' else "Please provide a highly concise summary of the following event, capturing the essential key information as succinctly as possible. Summarize the event:\n"
    for date,summary_dict in content:
        summary = summary_dict['content']
        prompt += (f"\n时间{date}发生的事件为{summary.strip()}" if language=='cn' else f"At {date}, the events are {summary.strip()}")
    prompt += ('\n总结：' if language=='cn' else '\nSummarization：')
    return prompt


def summarize_overall_personality(content,language='en'):
    # 构造可以总结性格的prompt
    prompt = '以下是用户在多段对话中展现出来的人格特质和心情，以及当下合适的回复策略：\n' if language=='cn' else "The following are the user's exhibited personality traits and emotions throughout multiple dialogues, along with appropriate response strategies for the current situation:"
    for date,summary in content:
        prompt += (f"\n在时间{date}的分析为{summary.strip()}" if language=='cn' else f"At {date}, the analysis shows {summary.strip()}")
    prompt += ('\n请总体概括用户的性格和AI恋人最合适的回复策略，尽量简洁精炼，高度概括。总结为：' if language=='cn' else "Please provide a highly concise and general summary of the user's personality and the most appropriate response strategy for the AI lover, summarized as:")
    return prompt


def summarize_person_prompt(content,user_name,boot_name,language):
    # 构造可以分析性格的prompt
    prompt = f'请根据以下的对话推测总结{user_name}的性格特点和心情，并根据你的推测制定回复策略。对话内容：\n' if language=='cn' else f"Based on the following dialogue, please summarize {user_name}'s personality traits and emotions, and devise response strategies based on your speculation. Dialogue content:\n"
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        prompt += f"\n{user_name}：{query.strip()}"
        prompt += f"\n{boot_name}：{response.strip()}"

    prompt += (f'\n{user_name}的性格特点、心情、{boot_name}的回复策略为：' if language=='cn' else f"\n{user_name}'s personality traits, emotions, and {boot_name}'s response strategy are:")
    return prompt


def summarize_memory(memory_dir,name=None,language='cn'):
    # 主函数：总结记忆文件中的所有对话
    boot_name = 'AI'
    gen_prompt_num = 1  # 每次生成的prompt数量
    # 【加载长期记忆文件】
    memory = json.loads(open(memory_dir,'r',encoding='utf8').read())

    for k, v in memory.items():
        if name != None and k != name:  # k是名字，只针对指定用户
            continue
        user_name = k  # 当前用户名
        print(f'为用户 {user_name} 更新记忆')
        if v.get('history') == None:    # 从v.get也可以看出来v有哪些key
            continue
        history = v['history']
        if v.get('summary') == None:
            memory[user_name]['summary'] = {}
        if v.get('personality') == None:
            memory[user_name]['personality'] = {}
        for date, content in history.items():  # 先处理history
            his_flag = False if (date in v['summary'].keys() and v['summary'][date]) else True
            person_flag = False if (date in v['personality'].keys() and v['personality'][date]) else True
            # 调用上面的函数构造prompt

            # 总结主题和关键信息
            hisprompt = summarize_content_prompt(content,user_name,boot_name,language)
            # 性格特点和心情
            person_prompt = summarize_person_prompt(content,user_name,boot_name,language)
            if his_flag:
                # 对对话内容进行总结
                his_summary = llm_client.generate_text_simple(prompt=hisprompt,prompt_num=gen_prompt_num,language=language)
                memory[user_name]['summary'][date] = {'content':his_summary}
            if person_flag:
                person_summary = llm_client.generate_text_simple(prompt=person_prompt,prompt_num=gen_prompt_num,language=language)
                memory[user_name]['personality'][date] = person_summary
        # 整体的prompt
        overall_his_prompt = summarize_overall_prompt(list(memory[user_name]['summary'].items()),language=language)
        overall_person_prompt = summarize_overall_personality(list(memory[user_name]['personality'].items()),language=language)
        memory[user_name]['overall_history'] = llm_client.generate_text_simple(prompt=overall_his_prompt,prompt_num=gen_prompt_num,language=language)
        memory[user_name]['overall_personality'] = llm_client.generate_text_simple(prompt=overall_person_prompt,prompt_num=gen_prompt_num,language=language)
 
    with open(memory_dir,'w',encoding='utf8') as f:
        print(f'成功为用户 {name} 更新了记忆')
        json.dump(memory,f,ensure_ascii=False)
    return memory


if __name__ == '__main__':
    summarize_memory('../memories/eng_memory_cases.json',language='en')

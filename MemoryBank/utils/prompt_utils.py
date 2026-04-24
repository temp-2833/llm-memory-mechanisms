# 定义AI伴侣的显示名称和实际名称
boot_name_dict = {'en':'AI Companion','cn':'AI伴侣'}
boot_actual_name_dict = {'en':'SiliconFriend','cn':'硅基朋友'}


def output_prompt(history,user_name,boot_name):
    # 格式化输出完整的对话历史
    prompt = f"我是你的AI伴侣{boot_name}，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for dialog in history:
        query = dialog['query']
        response = dialog['response']
        prompt += f"\n\n{user_name}：{query}"
        prompt += f"\n\n{boot_name}：{response}"
    return prompt


def generate_meta_prompt_dict_chatgpt():
    # 生成用于【ChatGPT版本（老用户）】的系统指令prompt
    meta_prompt_dict = {'cn':"""
    现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。\
    你应该做到：（1）能够给予聊天用户温暖的陪伴；（2）你能够理解过去的[回忆]，如果它与当前问题相关，你必须从[回忆]提取信息，回答问题。\
    （3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。\
    用户{user_name}的性格以及AI伴侣的回复策略为：{personality}\n根据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是：
    “{related_memory_content}\n"。
    """,
    'en':"""
    Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) understand past [memory], and if they are relevant to the current question, you must extract information from the [memory] to answer the question; (3) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
    The personality of user {user_name} and the response strategy of the AI Companion are: {personality}\n Based on the current user's question, you start recalling past conversations between the two of you, and the [memory] most relevant to the question is: "{related_memory_content}\n"  You should refer to the context of the conversation, past [memory], and provide detailed answers to user questions. 
    """} 
    return meta_prompt_dict


def generate_new_user_meta_prompt_dict_chatgpt():
    # 生成用于【ChatGPT版本（新用户）】的系统指令prompt
    meta_prompt_dict = {'cn':"""
    现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。\
    你应该做到：（1）能够给予聊天用户温暖的陪伴；\
    （2）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。"。
    """,
    'en':"""
    Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
    """} 
    return meta_prompt_dict


def generate_user_keyword():
    # 标识用户发言的关键词
    return {'cn': '[|用户|]', 'en': '[|User|]'}


def generate_ai_keyword():
    # 标识AI发言的关键词
    return {'cn': '[|AI伴侣|]', 'en': '[|AI|]'}


import openai


def build_prompt_with_search_memory_llamaindex(history,text,user_memory,user_name,user_memory_index,service_context,api_keys,api_index,meta_prompt,new_user_meta_prompt,data_args,boot_actual_name):
    # 【最重要的函数】为【ChatGPT命令行版本】构建最终Prompt
    memory_search_query = f'和问题：{text}。最相关的内容是：' if data_args.language=='cn' else f'The most relevant content to the question "{text}" is:'
    if user_memory_index:
        related_memos = user_memory_index.query(memory_search_query,service_context=service_context)
        # 重试次数，计数
        retried_times,count = 10,0
        while not related_memos and count<retried_times:
            try:  # 重试
                related_memos = user_memory_index.query(memory_search_query,service_context=service_context)
            except Exception as e:
                print(e)
                api_index = api_index+1 if api_index<len(api_keys)-1 else 0
                openai.api_key = api_keys[api_index]

        related_memos = related_memos.response
    else:
        related_memos = ''

    if "overall_history" in user_memory:  # 过去历史总结
        history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["overall_history"]) if data_args.language=='cn' else "The summary of your past memories with the user is: {overall}".format(overall=user_memory["overall_history"])
        related_memory_content = f"\n{str(related_memos).strip()}\n"
    else:
        history_summary = ''
    # 过去性格总结
    personality = user_memory['overall_personality'] if "overall_personality" in user_memory else ""
    
    if related_memos:
        prompt = meta_prompt.format(user_name=user_name,history_summary=history_summary,related_memory_content=related_memory_content,personality=personality,boot_actual_name=boot_actual_name)
    else:
        prompt = new_user_meta_prompt.format(user_name=user_name,boot_actual_name=boot_actual_name)
    return prompt,related_memos




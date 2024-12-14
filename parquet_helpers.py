import pandas as pd

def load_parquet(file_path, debug=False, parallel=False, total_shard=4, shard_idx=0):
    df = pd.read_parquet(file_path)

    if debug:
        df = df.head(100)

    if parallel:
        chunk_size = len(df) // total_shard
        start_idx = shard_idx * chunk_size
        end_idx = (shard_idx + 1) * chunk_size if shard_idx != total_shard - 1 else len(df)
        df = df.iloc[start_idx:end_idx]
        print("Parallel mode: shard_idx={}, start_idx={}, end_idx={}".format(shard_idx, start_idx, end_idx), flush=True)

    query_colname = "question" if "question" in df.columns else "query" if "query" in df.columns else None
    list_data_dict = []
    for idx in range(len(df)):
        new_item = dict(
            data_index = df.iloc[idx]['id'],
            query = df.iloc[idx][query_colname],
            context = df.iloc[idx]['context'],
        )
        list_data_dict.append(new_item)

    return list_data_dict

SYSTEM_MSG_RAG_SHORT = """
    You are a helpful assistant. Your job will be to answer questions accurately based on the given context and not your internal knowledge.
    If you can not answer the question only based on the provided context, return the answer: `Nie mogę udzielić odpowiedzi na to pytanie na podstawie podanego kontekstu`.
"""

QUERY_INTRO_NO_ANS = """Given the context `CONTEXT` and the query `QUERY` below, please provide an answer `ANSWER` to the question. 
    `CONTEXT`: {context} 

    `QUERY`: {query}

    `ANSWER`:
"""

def generatellama2__chat_prompt(messages):
    prompt = "<s>"
    for message in messages:
        role = message.get("role", "user")  # Domyślnie "user"
        content = message.get("content", "").strip()
        if role == "system":
            prompt += f"[INST] <<SYS>>\n{content}\n<</SYS>>\n"
        elif role == "user":
            prompt += f"[INST] {content} [/INST] "
        elif role == "assistant":
            prompt += f"{content} [/INST]"
    prompt = prompt.strip()
    return prompt

def build_hallu_ds_prompt(query, context, has_system_role):
    user_input = QUERY_INTRO_NO_ANS.format(context=context, query=query)

    messages = []

    if has_system_role:
        messages.append({"role": "system", "content": SYSTEM_MSG_RAG_SHORT})

    messages.append(
        {
            "role": "user",
            "content": (
                f"{SYSTEM_MSG_RAG_SHORT}{user_input}"
                if not has_system_role
                else user_input
            ),
        },
     )

    prompt = generatellama2__chat_prompt(messages)

    return prompt

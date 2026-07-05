import os
import re
import string
import json
import torch
import numpy as np
import transformers

from graphs.contriever_graph import LLaMAContrieverGraph, ContrieverGraph
from agents.llama_agent import LLaMAagent
from agents.parent_agent import GPTagent
from utils.utils import Logger


log_path = "MusiqueTestGPTmini"
# musique | hotpotqa 
task_name = "musique"
topk_episodic = 4
graph_model, qa_model = "gpt-4o-mini", "gpt-4o-mini"
log = Logger(log_path)

def run():
    tasks = get_data(task_name)
    agent_items, agent_qa, graph = load_setup(graph_model, qa_model)
    trueP, pred_len, true_len, EM = [], [], [], []
    for task in tasks:
        graph.clear()
        for text in task["paragraphs"]:
            triplets, episodic = graph.update_without_retrieve(text["paragraph_text"], [], log)

        question = task["question"]
        log("-" * 15)
        log("QUESTION: " + str(question))
        items = agent_items.item_processing_scores_qa(question)[0]
        if not isinstance(items, dict):
            true_answer = task["answer"].strip('''. \n'"?''').lower().split()
            trueP.append(0), pred_len.append(0), true_len.append(len(true_answer))
            EM.append(False)
            log("INCORRECT FORMAT OF ITEMS: " + str(items))
            continue
        log("CRUCIAL ITEMS: " + str(items))

        subgraph, episodic = graph.retrieve(items, question, [], topk_episodic)
        log("ASSOCIATED SUBGRAPH: " + str(subgraph))
        log("EPISODIC MEMORY: " + str(episodic))

        answer = get_answer(agent_qa, question, subgraph, episodic)
        log("AGENT ANSWER: " + str(answer))
        log("TRUE ANSWER: " + str(task["answer"]))

        compute_and_print_metrics(answer, task, trueP, true_len, pred_len, EM)
        log("="* 56 + "\n")



def get_data(task_name):
    if task_name == "musique":
        with open('qa_data/musique_ans_v1.0_dev.jsonl', 'r') as json_file:
            json_list = list(json_file)

        tasks = []
        for json_str in json_list:
            result = json.loads(json_str)
            tasks.append(result)
    if task_name == "hotpotqa":
        with open('qa_data/hotpot_dev_distractor_v1.json', 'r') as inp:
            data = json.load(inp)
        tasks = [" ".join(task["context"][-1]) for task in data]
    ids = np.random.RandomState(seed=42).permutation(len(tasks))[:200]
    tasks = [tasks[i] for i in ids]
    return tasks

def get_answer(agent, question, subgraph, episodic):
    prompt = f'''Your task is answer the following question: "{question}"

    Relevant facts from your memory: {subgraph}

    Relevant texts from your memory: {episodic}

    Answer the question "{question}" with Chain of Thoughts in the following format:
    "CoT: your chain of thoughts
    Direct answer: your direct answer to the question"
    IMPORTANT RULES FOR DIRECT ANSWER:
    - Direct answer must be extremely concise (typically 1-5 words).
    - DO NOT write a full sentence. For example, if the answer is "John Doe", output exactly "John Doe", not "The answer is John Doe".
    - Direct answer must not contain alternatives, descriptions or reasoning.
    If you cannot find the answer in the retrieved facts or texts, output "Unknown". DO NOT use your internal knowledge.
    Do not write anything except answer in the given format.

    Your answer: '''
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return agent.generate(prompt)[0]
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API Error in get_answer: {e}. Retrying in 10s...")
                time.sleep(10)
            else:
                print(f"API Error in get_answer: {e}. Max retries reached.")
                return "CoT: Error during API call.\nDirect answer: Unknown"

def normalize_answer(s):
    """MuSiQue公式評価スクリプト(stonybrooknlp/musique, metrics/answer.py)と
    同じ正規化: 小文字化・句読点除去・冠詞(a/an/the)除去・空白正規化。"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_and_print_metrics(answer, task, trueP, true_len, pred_len, EM):
    try:
        answer_str = answer.split("Direct answer:")[1]
        if not answer_str.strip():
            answer_str = "unknown"
    except IndexError:
        answer_str = "unknown"

    answer_str = normalize_answer(answer_str)
    true_answer_str = normalize_answer(task["answer"])

    answer_words = answer_str.split()
    true_words = true_answer_str.split()

    true_P = len({word for word in answer_words if word in true_words})
    trueP.append(true_P)
    pred_len.append(len(answer_words))
    true_len.append(len(true_words))
    EM.append(answer_str == true_answer_str)
    prec = np.sum(trueP) / np.sum(pred_len) if np.sum(pred_len) > 0 else 0
    rec = np.sum(trueP) / np.sum(true_len) if np.sum(true_len) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    em = np.mean(EM)
    log(f"F1: {f1}, RECALL: {rec}, PRECISION: {prec}, EXACT MATCH: {em}")

def load_setup(graph_model, qa_model):
    if "llama" in graph_model or "llama" in qa_model:
        pipeline = transformers.pipeline(
            "text-generation",
            model="Undi95/Meta-Llama-3-70B-Instruct-hf",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
    
    if "llama" in graph_model:
        graph = LLaMAContrieverGraph("", "You are a helpful assistant", "", pipeline, "cuda")
        agent_items = LLaMAagent("You are a helpful assistant", pipeline)

    else:
        graph = ContrieverGraph(graph_model, "You are a helpful assistant", os.environ.get("GEMINI_API_KEY"), "cuda")
        agent_items = GPTagent(graph_model, "You are a helpful assistant", os.environ.get("GEMINI_API_KEY"))

    if "llama" in qa_model:
        agent_qa = LLaMAagent("You are a helpful assistant", pipeline)

    else:
        agent_qa = GPTagent(qa_model, "You are a helpful assistant", os.environ.get("GEMINI_API_KEY"))

    return agent_items, agent_qa, graph


if __name__ == "__main__":
    run()
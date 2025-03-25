from argparse import ArgumentParser
from torch.cuda.amp import autocast, GradScaler #混合精度
import json, tqdm
import re
import time
from typing import Optional
import requests
import os
import torch
from model import (
    ClipRetriever,
    MistralAnswerGenerator,
    LLaMA3AnswerGenerator,
    reconstruct_wiki_article,
    PaLMAnswerGenerator,
    reconstruct_wiki_sections,
    WikipediaKnowledgeBaseEntry,
    BgeTextReranker,
)

from utils import load_csv_data, get_test_question, get_image, remove_list_duplicates
import PIL

iNat_image_path = "/mnt/gaojuanru/EchoSight/Questions/VQA/inat"

import requests

# 假设 remove_punctuation 函数用于去除标点符号
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# 计算 F1 分数的函数
def f1_score(set1, set2):
    if not set1 or not set2:
        return 0
    intersection = len(set1.intersection(set2))
    precision = intersection / len(set1) if len(set1) > 0 else 0
    recall = intersection / len(set2) if len(set2) > 0 else 0
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def eval_recall(candidates, ground_truth, top_ks=[1, 5, 10]):
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall

def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """
    提取文本中两个标签之间的内容
    """
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

def get_step_by_step_reasoning_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform section searches to help "
        "you answer the user's question accurately based on visual content. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant sections, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        "Process Termination: If at any point you determine that no further questions need to be generated for reasoning, simply output'stop'. Additionally, if your answer contains the word'stop' (regardless of case), immediately halt the reasoning process.\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the height of the tallest mountain in the image?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about the tallest mountain or similar images.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>tallest mountain height<|end_search_query|>\n\n"
        "(System returns processed information from relevant sections)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a section search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
        "- If you determine that no further questions need to be generated for reasoning, output'stop'.\n"
    )


def local_knowledge_search(search_query, top_5_sections):
    """
    在 top5 的 section 中进行内部搜索

    :param search_query: 搜索查询
    :param top_5_sections: top5 的 section 列表
    :return: 类似 Bing 搜索结果的字典
    """
    relevant_results = []
    for i, section in enumerate(top_5_sections):
        if search_query.lower() in section.lower():
            result = {
                "id": i + 1,
                "title": f"Section {i + 1}",
                "url": f"internal_section_{i + 1}",
                "snippet": section[:200],  # 截取前200个字符作为摘要
                "context": section
            }
            relevant_results.append(result)

    search_results = {
        "webPages": {
            "value": relevant_results
        }
    }
    return search_results

def extract_relevant_info(search_results: dict) -> list:
    """
    从 Bing 搜索结果中提取相关信息。

    Parameters:
    - search_results: 从 `bing_web_search` 返回的搜索结果字典。

    Returns:
    - 提取的相关信息列表，通常包含标题和摘要。
    """
    relevant_info = []

    # 处理搜索结果
    if "webPages" in search_results:
        for page in search_results["webPages"]["value"]:
            info = {
                "url": page["url"],
                "snippet": page["snippet"],  # 摘要
                "title": page["name"],  # 页面标题
            }
            relevant_info.append(info)

    return relevant_info

def weight_search_results(relevant_info, search_query):
    """
    为搜索结果加权整合。此函数可以根据相关性得分调整搜索结果对推理步骤的影响。

    :param relevant_info: 从搜索结果中提取的相关信息
    :param search_query: 搜索查询
    :return: 加权后的搜索信息
    """
    # 示例：根据某种策略加权搜索结果
    weighted_info = ""
    for info in relevant_info:
        # 假设根据某种评分机制给每个信息加权
        relevance_score = compute_relevance_score(info, search_query)  # 计算相关性得分
        weighted_info += f"Relevance Score: {relevance_score}\n{info['snippet']}\n"
    return weighted_info

def compute_relevance_score(info, search_query):
    """
    计算搜索结果的相关性得分，使用 F1 分数衡量与搜索查询的相关性。

    :param info: 搜索结果中的一个信息项
    :param search_query: 搜索查询
    :return: 相关性得分
    """
    snippet = info['snippet'].lower()
    snippet = remove_punctuation(snippet)
    snippet_words = set(snippet.split())

    query = search_query.lower()
    query = remove_punctuation(query)
    query_words = set(query.split())

    return f1_score(snippet_words, query_words)

def run_test(
    test_file_path: str,
    knowledge_base_path: str,
    faiss_index_path: str,
    top_ks: list,
    retrieval_top_k: int,
    **kwargs
):

    scaler = GradScaler() #混合精度

    test_list, test_header = load_csv_data(test_file_path)
    
    with open(iNat_image_path + "/val_id2name.json", "r") as f:
        inat_id2name = json.load(f) #N
    

    if kwargs["resume_from"] is not None:
        resumed_results = json.load(open(kwargs["resume_from"], "r"))
        kb_dict = json.load(open(knowledge_base_path, "r"))
    else:
        retriever =  ClipRetriever(device="cuda:0", model=kwargs["retriever_vit"])
        # retriever.save_knowledge_base_faiss(knowledge_base_path, scores_path=score_dict, save_path=faiss_index_path)
        retriever.load_knowledge_base(knowledge_base_path)
        retriever.load_faiss_index(faiss_index_path)

    recalls = {k: 0 for k in top_ks}
    reranked_recalls = {k: 0 for k in top_ks}

    hits = 0

    if kwargs["perform_vqa"]:
        from utils import evaluate_example
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")  # disable GPU for tensorflow
        question_generator = LLaMA3AnswerGenerator(
            model_path="/home/gaojuanru/mnt_link/gaojuanru/EchoSight/cache/huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
            device="cuda:1",
            # use_embedding_model=False,
        )
    if kwargs["perform_text_rerank"]:
        text_reranker = BgeTextReranker(
            model_path="/mnt/gaojuanru/EchoSight/cache/huggingface/BAAI/bge-reranker-v2-m3",
            device="cuda:1",
        )
        eval_score = 0

    if kwargs["perform_qformer_reranker"]:
        from lavis.models import load_model_and_preprocess

        blip_model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_reranker", model_type="pretrain", is_eval=True, device="cuda:0"
        )
        checkpoint_path = kwargs["qformer_ckpt_path"]

        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
        msg = blip_model.load_state_dict(checkpoint, strict=False)
        blip_model = blip_model.half()
        blip_model.use_vanilla_qformer = True
        print("Missing keys {}".format(msg.missing_keys))
        from data_utils import squarepad_transform, targetpad_transform

        preprocess = targetpad_transform(1.25, 224)

         # 修改设备分配逻辑
        available_gpus = [0,1]
        device_ids = available_gpus[:torch.cuda.device_count()]
        blip_model = blip_model.to(f'cuda:{device_ids[0]}')
        if len(device_ids) >= 1:
            blip_model = torch.nn.DataParallel(blip_model, device_ids=device_ids)

       # 添加日志记录设备分配情况
        print(f"Model is on devices: {list(blip_model.device_ids)}")


    metric = "url matching"

    retrieval_result = {}
    retriever_time = 0
    reranker_time = 0
    answer_time = 0
    for it, test_example in tqdm.tqdm(enumerate(test_list)):
        example = get_test_question(it, test_list, test_header)
        image = PIL.Image.open(
            get_image(
                example["dataset_image_ids"].split("|")[0],
                example["dataset_name"],
                inat_id2name, #N
            )
        )
        ground_truth = example["wikipedia_url"]
        target_answer = example["answer"].split("|")
        if example["dataset_name"] == "infoseek":
            data_id = example["data_id"]
        else:
            data_id = "E-VQA_{}".format(it)
        print("wiki_url: ", example["wikipedia_url"])
        print("question: ", example["question"])
        if kwargs["resume_from"] is not None:
            resumed_result = resumed_results[data_id]
            top_k_wiki, retrieval_simlarities = resumed_result["retrieved_entries"]
            reranked_sections = resumed_result["reranked_sections"]
            retrieval_simlarities = retrieval_simlarities
            entries = [WikipediaKnowledgeBaseEntry(kb_dict[url]) for url in top_k_wiki]
        else:
            retriever_start_time = time.time()
            top_k = retriever.retrieve_image_faiss(image, top_k=retrieval_top_k)
            top_k_wiki = [retrieved_entry["url"] for retrieved_entry in top_k]
            top_k_wiki = remove_list_duplicates(top_k_wiki)
            entries = [retrieved_entry["kb_entry"] for retrieved_entry in top_k]
            entries = remove_list_duplicates(entries)
            seen = set()
            retrieval_simlarities = [
                top_k[i]["similarity"]
                for i in range(retrieval_top_k)
                if not (top_k[i]["url"] in seen or seen.add(top_k[i]["url"]))
            ]
            retriever_end_time = time.time()
            retriever_time_1 = retriever_end_time - retriever_start_time
            retriever_time += retriever_time_1
            print(f"Retriever took {retriever_time_1:.4f} seconds")

        if kwargs["save_result"]:
            retrieval_result[data_id] = {
                "retrieved_entries": [entry.url for entry in entries[:10]],
                "retrieval_similarities": [
                    sim.item() for sim in retrieval_simlarities[:10]
                ],
            }
        if metric == "answer matching":
            entry_articles = [reconstruct_wiki_article(entry) for entry in entries]
            found = False
            for i, entry in enumerate(entry_articles):
                for answer in target_answer:
                    if answer.strip().lower() in entry.strip().lower():
                        found = True
                        break
                if found:
                    break
            if found:
                for k in top_ks:
                    if i < k:
                        recalls[k] += 1

        else:
            # in url_matching
            recall = eval_recall(top_k_wiki, ground_truth, top_ks)
            for k in top_ks:
                recalls[k] += recall[k]
        for k in top_ks:
            print("Avg Recall@{}: ".format(k), recalls[k] / (it + 1))

        if kwargs["perform_qformer_reranker"]:
            reranker_start_time = time.time()
            reference_image = preprocess(image).to("cuda:0").unsqueeze(0)
            sections = []
            section_to_entry = []
            for entry_id, entry in enumerate(entries):
                entry_sections = reconstruct_wiki_sections(entry)
                sections.extend(entry_sections)
                section_to_entry.extend([entry_id] * len(entry_sections))

            qformer_question = example["question"]
            qformer_articles = [txt_processors["eval"](article) for article in sections]
            with torch.cuda.amp.autocast():
                if isinstance(blip_model, torch.nn.DataParallel):
                    blip_model = blip_model.module
                fusion_embs = blip_model.extract_features(
                    {"image": reference_image, "text_input": qformer_question},
                    mode="multimodal",
                )["multimodal_embeds"]
                rerank_step = 500  # to calculate the embedding in iteration
                for section_spilit in range(0, len(qformer_articles), rerank_step):
                    article_embs = blip_model.extract_features(
                        {
                            "text_input": qformer_articles[
                                section_spilit : section_spilit + rerank_step
                            ]
                        },
                        mode="text",
                    )["text_embeds_proj"][:, 0, :]
                    if section_spilit == 0:
                        article_embs_all = article_embs
                    else:
                        article_embs_all = torch.cat(
                            (article_embs_all, article_embs), dim=0
                        )
                print("article_embs_all shape: ", article_embs_all.shape)
                scores = torch.matmul(
                    article_embs_all.unsqueeze(1).unsqueeze(1),
                    fusion_embs.permute(0, 2, 1),
                ).squeeze()
                scores, _ = scores.max(-1)

                section_similarities = [
                    retrieval_simlarities[section_to_entry[i]]
                    for i in range(len(sections))
                ]
                alpha_1 = 0.5
                alpha_2 = 1 - alpha_1
                scores = (
                    alpha_1 * torch.tensor(section_similarities).to("cuda:0")
                    + alpha_2 * scores
                )
                # rank by scores high to low
                scores, reranked_index = torch.sort(scores, descending=True)
            top_k_wiki = remove_list_duplicates(
                [entries[section_to_entry[i]].url for i in reranked_index]
            )
            reranked_entries = remove_list_duplicates(
                [entries[section_to_entry[i]] for i in reranked_index]
            )
            reranked_sections = remove_list_duplicates(
                [sections[i] for i in reranked_index]
            )
            reranker_end_time = time.time()
            reranker_time_1 = reranker_end_time - reranker_start_time
            reranker_time += reranker_time_1
            print(f"reranker took {reranker_time_1:.4f} seconds")
            if kwargs["save_result"]:
                retrieval_result[data_id]["reranked_entries"] = [
                    entry.url for entry in reranked_entries[:10]
                ]
                retrieval_result[data_id]["reranked_sections"] = reranked_sections[:5]

        if metric == "answer matching":
            entry_sections = reranked_sections
            found = False
            for i, entry in enumerate(entry_sections):
                for answer in target_answer:
                    if answer.strip().lower() in entry_sections[i].strip().lower():
                        found = True
                        break
                if found:
                    break
            if found:
                for k in top_ks:
                    if i < k:
                        reranked_recalls[k] += 1

        else:
            recall = eval_recall(top_k_wiki, ground_truth, top_ks)
            for k in top_ks:
                reranked_recalls[k] += recall[k]

        for k in top_ks:
            print("Reranked Avg Recall@{}: ".format(k), reranked_recalls[k] / (it + 1))
        if kwargs["perform_text_rerank"]:
            if ground_truth in top_k_wiki[:5]:
                gt_index = top_k_wiki.index(ground_truth)
                index, hit = text_reranker.rerank_entry_sections(
                    example["question"], reranked_sections, top_k=5, gt_index=gt_index
                )
                temp = reranked_sections[0]
                reranked_sections[0] = reranked_sections[index]
                reranked_sections[index] = temp
            else:
                gt_index = -1
                hit = 0
            hits += hit
            print("Text Reranking Recalls", hits / (it + 1))

        # if kwargs["perform_vqa"]:
        #     answer = question_generator.llm_answering(
        #         question=example["question"], entry_section=reranked_sections[0]
        #     )

        #     print("answer: ", answer)
        #     print("target answer: ", target_answer)
        if kwargs["perform_vqa"]:
            # 初始化推理步骤历史
            current_answer = ""
            reasoning_steps = []
            turn = 0
            max_turn = 5  # 设置最大推理次数
            search_count = 0
            max_search_limit = 5  # 每回合最大搜索次数
            executed_search_queries = set()  # 已执行的搜索查询
            search_cache = {}  # 用于存储搜索结果的缓存
            answer_start_time = time.time()
            while turn < max_turn:
                # 每一步的推理生成
                prompt = f"Question: {example['question']}\n"
                prompt += f"Previous Answer: {current_answer}\n"
                prompt += f"Reasoning steps so far: {' '.join(reasoning_steps)}\n"
                prompt += "Generate the next step of reasoning: "

                if turn == 0:
                    # 第一次迭代，使用top 1的section
                    answer = question_generator.llm_answering(
                        question=example["question"],
                        entry_section=reranked_sections[0]
                    )
                else:
                    # 后续迭代，使用reranked_sections的前5个段落生成答案
                    top_5_sections = reranked_sections[:5]
                    answer = question_generator.llm_answering(
                        question=example["question"],
                        entry_section=" ".join(top_5_sections)  # 使用前5个段落生成答案
                    )

                reasoning_steps.append(answer)  # 将当前步骤加入推理链

                # 外部知识整合（此部分现在只在生成问题时使用，避免调用bing）
                search_query = extract_between(answer, "<|begin_search_query|>", "<|end_search_query|>")
                if search_query and search_count < max_search_limit:
                    if search_query in search_cache:
                        search_results = search_cache[search_query]
                        print(f"使用缓存的搜索结果：\"{search_query}\"")
                    else:
                        # 执行内部搜索
                        top_5_sections = reranked_sections[:5]
                        try:
                            search_results = local_knowledge_search(search_query, top_5_sections)
                            search_cache[search_query] = search_results  # 缓存搜索结果
                            print(f"执行并缓存了搜索查询：\"{search_query}\"")
                        except Exception as e:
                            print(f"搜索查询失败：'{search_query}': {e}")
                            search_results = {}

                    relevant_info = extract_relevant_info(search_results)  # 从搜索结果中提取相关信息
                    # 计算搜索结果的相关性得分并加权整合
                    weighted_knowledge = weight_search_results(relevant_info, search_query)
                    reasoning_steps.append(f"Weighted search results: {weighted_knowledge}")

                    # 更新推理步骤
                    current_answer = question_generator.llm_answering(
                        question=example["question"] + "\n".join(reasoning_steps),
                        entry_section=reranked_sections[0]
                    )
                    search_count += 1
                else:
                    current_answer = answer

                # 如果生成的答案包含"stop"或为空，结束推理过程
                if "stop" in answer.lower() or not answer.strip():
                    break

                turn += 1

            # 最终答案
            answer_end_time = time.time()
            answer_time_1 = answer_end_time - answer_start_time
            answer_time += answer_time_1
            print(f"answer took {answer_time_1:.4f} seconds")
            print("Final answer after reasoning:", current_answer)
            print("target answer: ", target_answer)
            print("Reasoning steps:", reasoning_steps)

            score = evaluate_example(
                example["question"],
                reference_list=target_answer,
                candidate=current_answer,
                question_type=example["question_type"],
            )

            eval_score += score
            print("score: ", score, "iter: ", it + 1)
            print("eval score: ", eval_score / (it + 1))
            # print("retrieval_result: ", retrieval_result)
            if kwargs["save_result"]: 
                retrieval_result[data_id]["prediction"] = current_answer

        if kwargs["save_result"]: 
            with open("/home/gaojuanru/mnt_link/gaojuanru/EchoSight/scripts/saveresult1.json", "w") as f:
                json.dump(retrieval_result, f, indent=4)
    
    print(f"retriever took total {retriever_time:.4f} seconds")
    print(f"reranker took total {reranker_time:.4f} seconds")
    print(f"answer took total {answer_time_1:.4f} seconds")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--knowledge_base", type=str, required=True)
    parser.add_argument("--faiss_index", type=str, required=True)
    parser.add_argument(
        "--top_ks",
        type=str,
        default="1,5,10",
        help="comma separated list of top k values, e.g. 1,5,10,20,100",
    )
    parser.add_argument("--perform_vqa", action="store_true")
    parser.add_argument("--perform_text_rerank", action="store_true")
    parser.add_argument("--perform_qformer_reranker", action="store_true")
    parser.add_argument("--qformer_ckpt_path", type=str, default=None)
    parser.add_argument("--retrieval_top_k", type=int, default=10)
    parser.add_argument("--save_result", action="store_true")
    # parser.add_argument("--save_result_path", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--retriever_vit", type=str, default="eva-clip", help="siglip-so400m, clip or eva-clip"
    )
    args = parser.parse_args()

    test_config = {
        "test_file_path": args.test_file,
        "knowledge_base_path": args.knowledge_base,
        "faiss_index_path": args.faiss_index,
        "top_ks": [int(k) for k in args.top_ks.split(",")],
        "retrieval_top_k": args.retrieval_top_k,
        "perform_vqa": args.perform_vqa,
        "perform_text_rerank": args.perform_text_rerank,
        "perform_qformer_reranker": args.perform_qformer_reranker,
        "qformer_ckpt_path": args.qformer_ckpt_path,
        "save_result": args.save_result,
        "resume_from": args.resume_from,
        "retriever_vit": args.retriever_vit,
    }
    print("test_config: ", test_config)
    run_test(**test_config)
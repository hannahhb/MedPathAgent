import json
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sentence_transformers import SentenceTransformer
from data import QADataset
import utils
import yaml
import os
import argparse
from tqdm import tqdm
import multiprocessing
import random
import re
import embedding_search
import ast

import multiprocessing as mp
mp.set_start_method("spawn", force=True)


def _format_options(options: dict) -> str:
    """Turn options dict (or string) into newline-delimited choices."""
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except Exception:
            return "Invalid options format."

    lines = []
    for k in sorted(options.keys()):
        lines.append(f"{k}. {options[k]}")
    return "\n".join(lines)


def _parse_answer_letter(text: str) -> str:
    """
    Try to pull 'ANSWER: <A-D>' from model output; 
    fallback to any single A-D letter; 
    else return the full text.
    """
    # Look for explicit ANSWER: <A-D>
    m = re.search(r'ANSWER\s*:\s*([A-D])', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Fallback to any standalone A-D letter
    m = re.search(r'\b([A-D])\b', text.strip(), flags=re.IGNORECASE)
    return m.group(1).upper() if m else text.strip()


total_cost = 0.0

def serialize_paths_with_relations(path_tuples):
    serialized = []
    for nodes, relations in path_tuples:
        # ensure lengths align: len(relations) == len(nodes) - 1
        parts = []
        for i in range(len(relations)):
            parts.append(nodes[i])
            if relations[i]!="parent-child":
                parts.append(relations[i])
        # append the final node
        parts.append(nodes[-1])
        serialized.append(" -> ".join(parts))
    serialized = "\n".join(serialized)
    return serialized


spacy_to_primekg_type = {
    "DISEASE": "disease",
    "CHEMICAL": "drug",
    "SIMPLE_CHEMICAL": "drug",
    "GENE_OR_GENE_PRODUCT": "gene/protein",
    "AMINO_ACID": "gene/protein",
    "ANATOMICAL_SYSTEM": "anatomy",
    "TISSUE": "anatomy",
    "CELL": "anatomy",
    "CELLULAR_COMPONENT": "cellular_component",
    "DEVELOPING_ANATOMICAL_STRUCTURE": "anatomy",
    "MULTI-TISSUE_STRUCTURE": "anatomy",
    "ORGAN": "anatomy",
    "ORGANISM": "anatomy",
    "ORGANISM_SUBDIVISION": "anatomy",
    "ORGANISM_SUBSTANCE": "anatomy",
    "IMMATERIAL_ANATOMICAL_ENTITY": "anatomy",
    "PATHOLOGICAL_FORMATION": "effect/phenotype",
    # … include as before …
    # Heuristics for pathway via keywords (handled separately rather than label mapping)
}


def reasoning_generation(
        question,
        kg,
        tokenizer,
        model, 
        nodeemb_dict,
        options: dict,
        topK_reasoning_paths=3,
        max_path_number_per_group=50,
        temperature=0.0,
        max_tokens=5000,
        filter_path=False
    ):
    
    """
    Generates (paths -> reasoning -> answer) for a question with options.
    Returns: dict with {'paths_text', 'reasoning', 'answer'}
    """
    import spacy
    nlps = ["en_ner_bc5cdr_md", "en_ner_bionlp13cg_md"]
    
    nlps = [spacy.load(nlp) for nlp in nlps]
   
    # 1) Extract entities from the question and options only
    text = f"Question: {question}. Options: {_format_options(options)}"
    logger.info(f"Total API cost so far (accumulated in run_llm calls): {utils.api_total_cost}")
    try:
        docs = [nlp(text) for nlp in nlps]
        ents_json = {"Entity": []}
        for doc in docs:
            for ent in doc.ents:
                ents_json["Entity"].append({"name": ent.text, "type": ent.label_})
       
        # Optionally dedupe by (name,type)
        seen = set()
        unique_entities = []
        for e in ents_json["Entity"]:
            key = (e["name"].lower(), e["type"])
            if key not in seen:
                unique_entities.append(e)
                seen.add(key)
                
        ents_json["Entity"] = unique_entities
        for e in ents_json["Entity"]:
            e_type = e["type"]
            mapped = spacy_to_primekg_type.get(e_type)
            e["type"] = mapped

        # Filter
        ents_json["Entity"] = [e for e in ents_json["Entity"] if e["type"]]
        # ents_text = utils.coarse_entity_extraction(text, temperature=temperature, max_tokens = max_tokens)
        # print("ENT TEXT")
        # print(ents_text)
        # ents_json = utils.parse_entities_block(ents_text)
        print("ENT JSON")
        print(ents_json)
    except Exception as e:
        logger.info(f"Entity extraction failed: {e}")
        ents_json = {"Entity": []}

    # 2) Map entities to KG nodes (via similarity + LLM tie-breaker)
    type_set = set(getattr(kg, 'graph', None) or [])  # placeholder if needed; we actually need the DF types
    # We don't actually know the types present from the graph, so pull types from nodeemb_dict
    type_set = set(nodeemb_dict.keys())
    

    mapped_nodes = []
    for entity in ents_json.get("Entity", []):
        etype = entity.get("type")
        ename = entity.get("name", "")
        if not etype or etype not in type_set or not ename:
            continue
        
        similar_entities, top1 = utils.get_topk_similar_entities(
            entity, tokenizer=tokenizer, model=model, nodeemb_dict=nodeemb_dict, node_name_dict= node_name_dict,
            k=100, filter_threshold=0.5, device=device
        )
        if not similar_entities:
            continue

        # perfect match or strong match
        selected = None
        for ent in similar_entities:
            if ename.lower() == ent.lower():
                selected = {"name": ent, "id": str(similar_entities.index(ent))}
                break
        if selected is None:
            selected = {"name": similar_entities[0], "id": "0"}

        if selected is None:
            # LLM-based selection without gold answer
            sel_text = utils.most_correlated_entity_selection(
                text, ename, similar_entities,
                temperature=temperature, max_tokens=max_tokens
            )
            # print(sel_text)
            try:
                selected = utils.parse_selected_entity(sel_text)["selected_entity"]
            except Exception:
                selected = {"name": "NONE", "id": "NONE"}

        if selected.get("name") != "NONE":
            mapped_nodes.append(similar_entities[int(selected["id"])])
    # 
    # mapped_nodes = utils.QA_reformat_with_entity_extraction(question=question, knowledge_graph=kg, emb_model=emb_model,
    #                                          nodeemb_dict=nodeemb_dict, temperature=temperature, max_tokens=max_tokens)
    mapped_nodes = list(dict.fromkeys(mapped_nodes))  # dedupe, keep order
    logger.info(f"Question entities mapped to KG nodes: {mapped_nodes}")

    # 3) Build shortest paths BETWEEN mapped question nodes
    path_all = []
    try:
        # for i in range(len(mapped_nodes)):
        #     for j in range(i + 1, len(mapped_nodes)):
        #         src, dst = mapped_nodes[i], mapped_nodes[j]
        #         try:
        #             path_all += list(nx.all_shortest_paths(kg, src.lower(), dst.lower()))
        #         except Exception as e:
        #             if type(e).__name__ == 'NodeNotFound':
        #                 logger.info(f"Node not found when pathing: {src} -> {dst}")
        #             continue
        # for i in range(len(mapped_nodes)):
        #     for j in range(i+1, len(mapped_nodes)):
        #         src = mapped_nodes[i].lower()
        #         dst = mapped_nodes[j].lower()
        #         paths = embedding_search.embedding_guided_beam_search(G, embeddings, src, dst,
        #                                             beam_width=20, max_depth=6, stop_when_found=5)
        #         for p in paths:
        #             path_all.append(p)
        path_all = []
        for i in range(len(mapped_nodes)):
            for j in range(i+1, len(mapped_nodes)):
                src, dst = mapped_nodes[i].lower(), mapped_nodes[j].lower()
                local_paths = embedding_search.induce_and_shortest_paths(G, name_to_idx, embeddings_mat, 
                                                                         src, dst, 
                                                                         node_list, k_each=200, device = device)
                # if local_paths:
                path_all.extend(local_paths)
                # else:
                #     # fallback to embedding-guided beam search
                #     beam_paths = beam_search_neighbor_ranking(G, name_to_idx, embeddings_mat, src, dst, beam_width=30, max_depth=6)
                #     path_all.extend(beam_paths)

    except Exception as e:

        logger.info(f"Path generation error: {e}")

    logger.info(f"the number of extracted paths: {len(path_all)}")

    if not path_all:
        # No KG paths — still answer from options using the question only
        options_text = _format_options(options)
        reasoning = utils.llm_generate_answer_with_reasoning(
            question=question,
            options=options_text,
            reasoning="No KG paths found. Use domain knowledge from the question.",
        )
        answer = _parse_answer_letter(reasoning)
        return {"paths_text": "nopaths", "reasoning": reasoning, "answer": answer}

    # Optional: filter down path set (uses LLM ranking; requires an 'answer' in original API,
    # so we skip that and just sample uniformly if requested).
    if filter_path:
        # path_sampling now requires the question and optional options
        sampled = utils.path_sampling(path_all=path_all,
                                question=question,
                                options=options,
                                topK_reasoning_paths=topK_reasoning_paths,
                                max_path_number_per_group=max_path_number_per_group,
                                max_tokens = max_tokens,
                                logger=logger)
        # sampled is a list of path lists
        if sampled:
            path_all = sampled
        else:
            # fallback: keep a small random subset
            path_all = random.sample(path_all, min(len(path_all), max_path_number_per_group))


    # 4) Serialize paths and guard size
    paths_text = serialize_paths_with_relations(path_all)
    # if len(paths_text) > max_tokens * 5:
    #     # too large; trim
    #     keep = max(1, int((max_tokens * 5) / (len(paths_text) / max(1, len(path_all)))))
    #     path_all = path_all[:keep]
    #     paths_text = '\n'.join([f"{idx+1}:" + '->'.join(inner) for idx, inner in enumerate(path_all)])

    logger.info(f"reasoning paths used: {paths_text}")

    # 5) Ask the model to answer using paths + options
    options_text = _format_options(options)
    prompt = f"""
You are an expert in the medical domain.
Use the following knowledge-graph paths (if helpful) to reason and answer the question by choosing ONE option. 
If no Knowledge Graphs are present then you may use internal reasoning. 

Question:
{question}

Options:
{options_text}

Knowledge-graph paths (sampled):
{paths_text}

Let's think step-by-step about the problem. Respond with your reasoning followed by the final choice (single letter) in format:
REASONING: <free-text>
ANSWER: <A-D>
"""
    reasoning = utils.run_llm(prompt, temperature=temperature, max_tokens=max_tokens)
    answer = _parse_answer_letter(reasoning)

    return {"paths_text": paths_text, "reasoning": reasoning, "answer": answer}

# def worker_init(new_dataset, new_G, new_primekg, new_nodeemb_dict, dataset_name):
#     global logger, dataset, G, primekg, emb_model, nodeemb_dict, node_name_dict
#     logger = utils.init_logger(name=f"{dataset_name}_{os.getpid()}")
#     dataset = new_dataset
#     G = new_G
#     primekg = new_primekg
#     nodeemb_dict = new_nodeemb_dict

#     # Now load model inside worker
#     emb_model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")

    
# ----------------------------------
# Worker function to process one sample.
# ----------------------------------
# def process_sample(sample_id):
#     global dataset, logger, G, primekg, emb_model, nodeemb_dict, node_name_dict
#     sample = dataset[sample_id]
#     qid = sample.get('id', sample_id)
#     question = sample['question']
#     options = sample.get('options', {})

#     logger.info(f"Processing sample id {qid}.")
#     out = reasoning_generation(
#         question=question,
#         kg=G,
#         emb_model=emb_model,
#         nodeemb_dict=nodeemb_dict,
#         options=options,
#         filter_path=True
#     )
#     logger.info(f"Finished sample id {qid} with answer: {out['answer']}")
#     return {
#         "id": qid,
#         "question": question,
#         "options": options,
#         "paths": out["paths_text"],
#         "reasoning": out["reasoning"],
#         "answer": out["answer"]
#     }

import uuid
from datetime import datetime

def get_unique_filename(base_path, ext):
    counter = 1
    filename = f"{base_path}.{ext}"
    while os.path.exists(filename):
        filename = f"{base_path}_v{counter}.{ext}"
        counter += 1
    return filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="medqa")
    parser.add_argument("--sample", type=int, default=-1, help="Number of samples to run; -1 = full dataset")
    parser.add_argument("--start", type=float, default=0.0, help="Start slice (0.0 - 1.0)")
    parser.add_argument("--end", type=float, default=1, help="End slice (0.0 - 1.0)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output", type=str, default="run0")

    args = parser.parse_args()

    base_dir = os.path.join("output", args.output)
    os.makedirs(base_dir, exist_ok=True)
    jsonl_path = os.path.join(base_dir, "log.jsonl")
    csv_path = os.path.join(base_dir, "result.csv")

    object_dataset_name = args.dataset
    logger = utils.init_logger(name=object_dataset_name)
    logger.info(f"Start reasoning generation for dataset: {object_dataset_name}")

    dataset = QADataset(**yaml.safe_load(open('./configs/dataset_configs.yml'))[object_dataset_name])
    total_len = len(dataset)

    # Compute slice
    start_idx = int(args.start * total_len)
    end_idx = int(args.end * total_len)
    if args.sample != -1:
        end_idx = min(start_idx + args.sample, total_len)
    test_samples = end_idx - start_idx

    logger.info(f"Processing from index {start_idx} to {end_idx} (total {test_samples})")

    # Load KG
    primekg = pd.read_csv('./kg.csv', low_memory=False)
    G = utils.build_graph(primekg[['x_name', 'display_relation', 'y_name']].values.tolist())

    # 1) Set model name
    emb_model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

    # 2) Load SapBERT tokenizer + model
    device = torch.device("cpu")
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
    model     = AutoModel.from_pretrained(emb_model_name).to(device)
    model.eval()

    cache_file = './node_embeddings_sapbert.pt'


    # 3) Update your node‐embedding generation call
    # nodeemb_dict, node_name_dict = utils.generate_node_embeddings(
    #     knowledge_graph_path='/Users/hannah_mac/Documents/rmit/rmit_hons_y4/MedReason/kg.csv',
    #     sapbert_tokenizer=tokenizer,
    #     sapbert_model=model,
    #     device=device,
    #     cache_file=cache_file
    # )   
    
    # nodeemb_dict = torch.load(
    #     '/Users/hannah_mac/Documents/rmit/rmit_hons_y4/MedReason/node_embeddings.pt',
    #     map_location='cpu', weights_only=False
    # )
    nodeemb_data = torch.load(cache_file, map_location='cpu')
    # if you saved just the dict and nothing else:
    nodeemb_dict = nodeemb_data['nodeemb_dict'] if 'nodeemb_dict' in nodeemb_data else nodeemb_data
    node_name_dict = nodeemb_data.get('node_name_dict', None)
    print("embeddings loaded")
    
    node_list, embeddings_mat, name_to_idx, node_type_map = embedding_search.build_node_embedding_index(primekg, nodeemb_dict,node_name_dict)
    
    # --- Resume logic ---
    processed_ids = set()
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            processed_ids.update(existing_df["id"].astype(str).tolist())
            logger.info(f"Resuming: {len(processed_ids)} IDs already processed")
        except Exception as e:
            logger.warning(f"Failed to read existing CSV for resume: {e}")

    results = []

    if args.batch_size == 1:
        csv_header_written = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(jsonl_path, 'a') as f_jsonl, open(csv_path, 'a', newline='') as f_csv:
            import csv
            writer = csv.DictWriter(f_csv, fieldnames=["id", "choice", "prediction", "reasoning"])
            if not csv_header_written:
                writer.writeheader()
                csv_header_written = True

            for ids in tqdm(range(start_idx, end_idx)):
                sample = dataset[ids]
                qid = str(sample.get('id', ids))

                # Skip if already processed
                if qid in processed_ids:
                    continue

                question = sample['question']
                options = sample.get('options', {})

                try:
                    out = reasoning_generation(
                        question=question,
                        kg=G,
                        tokenizer=tokenizer,
                        model=model, 
                        nodeemb_dict=nodeemb_dict,
                        topK_reasoning_paths=3,
                        max_tokens=1024,
                        options=options,
                        filter_path=False
                    )
                    row = {
                        "id": qid,
                        "question": question,
                        "options": options,
                        "paths": out["paths_text"],
                        "reasoning": out["reasoning"],
                        "answer": out["answer"]
                    }

                    # Append JSONL
                    f_jsonl.write(json.dumps(row) + "\n")
                    f_jsonl.flush()

                    # Append CSV (per-iteration)
                    writer.writerow({
                        "id": qid,
                        "choice": row.get("answer", "N/A"),
                        "prediction": row.get("reasoning", "N/A"),
                        "reasoning": row.get("paths", "")
                    })
                    f_csv.flush()

                    results.append(row)
                    processed_ids.add(qid)
                    
                except Exception as e:
                    print(e)
                    out = {"paths_text": "", "reasoning": f"Error: {e}", "answer": "N/A"}
                    print("token probably expired")
                

    else:
        print("Multi-processing TBD")

    logger.info(f"Saved: {jsonl_path} and {csv_path}")
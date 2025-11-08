# utils.py
# Refactor of your file to use Amazon Bedrock (Converse API) instead of Azure OpenAI.

import os
import time
import json
import random
import logging
from typing import Dict, Any, Optional, List

import boto3
from botocore.exceptions import BotoCoreError, ClientError

import networkx as nx
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import re
from typing import Union, Dict


# ------------------------------
# Global cost accumulator (unchanged)
# ------------------------------
api_total_cost = 0.0

# ------------------------------
# Model registry for Bedrock
# NOTE: Update pricing to your region’s latest numbers if you want exact costs.
# Prices below are EXAMPLES (USD per 1M tokens) and may be outdated—replace as needed.
# ------------------------------


llama70b3_3   = "us.meta.llama3-3-70b-instruct-v1:0"
gpt_oss_120b  = "openai.gpt-oss-120b-1:0"  # <-- add "us." prefix for consistency


BEDROCK_MODELS: Dict[str, Dict[str, Any]] = {
    # Claude 3.5 Sonnet (Converse-ready)
    "claude-3-5-sonnet": {
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "input_price": 3.00 / 10**6,   # example placeholder; update with real price
        "output_price": 15.00 / 10**6, # example placeholder; update with real price
        "inference_profile_arn": None, # set if your account requires one
    },
    # Llama 3.1 70B Instruct (may require inference profile)
    "llama3-70b-instruct": {
        "model_id": llama70b3_3,  # example; check your region
        "input_price": 0.00072 /10**3,   # placeholder
        "output_price": 0.00072 /10**3,  # placeholder
        # "inference_profile_arn": os.environ.get("BEDROCK_INFERENCE_PROFILE_ARN_L70B"),  # set env if needed
    },
    # Cohere Command R+ (example)
    "cohere-command-r-plus": {
        "model_id": "cohere.command-r-plus-v1:0",
        "input_price": 0.50 / 10**6,   # placeholder
        "output_price": 1.50 / 10**6,  # placeholder
        "inference_profile_arn": None,
    },
    "gpt-oss-120b": {
        "model_id": gpt_oss_120b,  # example; check your region
        "input_price": 0.00015 /10**3,   # placeholder
        "output_price": 0.00006 /10**3,  # placeholder
        # "inference_profile_arn": os.environ.get("BEDROCK_INFERENCE_PROFILE_ARN_L70B"),  # set env if needed
    },
}



# Default engine alias you can use in your calls
DEFAULT_ENGINE = "llama3-70b-instruct"

def extract_delimited_block(text: str,
                            start: str = "BEGIN OUTPUT",
                            end: str = "END OUTPUT") -> str:
    """
    Extracts content between `start` and `end` delimiters from LLM output.
    Raises ValueError if delimiters not found.
    """
    try:
        start_idx = text.index(start) + len(start)
        end_idx = text.index(end, start_idx)
        return text[start_idx:end_idx].strip()
    except Exception as e:
        preview = text[:300].replace("\n", " ")
        raise ValueError(f"Failed to extract delimited block. Preview: {preview}") from e


def parse_llm_output_block(text: str, start: str = "BEGIN OUTPUT", end: str = "END OUTPUT") -> Union[Dict, str]:
    """
    Extracts content between delimiters and parses it into a dict.

    Handles both:
    - JSON-style output inside delimiters
    - Key-value format (e.g., name: foo)

    Returns:
        - dict of parsed content if possible
        - raw string if structured parsing fails
    Raises:
        - ValueError if delimiters are not found or format is unsupported
    """
    try:
        block = extract_delimited_block(text, start, end)
        # Try JSON first
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            # Fallback: parse key: value pairs
            parsed = {}
            for line in block.strip().splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    parsed[key.strip()] = value.strip()
            if parsed:
                return parsed
            else:
                raise ValueError("No key-value pairs found.")
    except Exception as e:
        preview = text[:300].replace("\n", " ")
        raise ValueError(f"Failed to parse LLM output. Preview: {preview}") from e

def parse_entities_block(text: str) -> dict:
    """
    Parse LLM output with multiple entity blocks in 'id/type/name' format
    into the old JSON-style dict {"Entity": [ {...}, {...} ]}.
    """
    block = extract_delimited_block(text)  # between BEGIN OUTPUT / END OUTPUT

    entities = []
    current = {}
    for line in block.splitlines():
        line = line.strip()
        if not line:
            # blank line = end of one entity block
            if current:
                entities.append(current)
                current = {}
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            current[k.strip().lower()] = v.strip()

    # flush last entity
    if current:
        entities.append(current)

    # Normalize keys (id/type/name)
    norm_entities = []
    for ent in entities:
        norm_entities.append({
            "id": ent.get("id"),
            "type": ent.get("type"),
            "name": ent.get("name")
        })

    return {"Entity": norm_entities}


def parse_selected_entity(sel_text: str) -> Dict[str, str]:
    """
    Tolerant parser for LLM selection outputs.
    Supports:
      - JSON block (with or without delimiters)
      - BEGIN/END delimited simple key: value blocks
      - bare key:value lines
    Returns a dict e.g. {"name": "...", "id": "...", "reason": "..."}
    Raises ValueError if no name/id discovered.
    """
    # Try to find explicit BEGIN/END first; if not present, use the whole text.
    content = None
    try:
        content = extract_delimited_block(sel_text)
    except Exception:
        # fallback: try to use raw string
        content = sel_text.strip()

    # Try JSON parse (the block might be a JSON object or contain one)
    try:
        parsed = json.loads(content)
        # If JSON contains 'selected_entity' key, return that sub-dict
        if isinstance(parsed, dict):
            if "selected_entity" in parsed and isinstance(parsed["selected_entity"], dict):
                result = parsed["selected_entity"]
                # ensure 'name' and 'id' keys are str
                if "name" in result and "id" in result:
                    return {k: str(v) for k, v in result.items()}
            # sometimes model returns a dict mapping ranking->path; fall through
    except Exception:
        pass

    # fallback: parse key: value lines
    selected = {}
    for line in content.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip().strip('"').strip("'")
            selected[key] = val

    # If keys are nested like selected_entity.name, try to flatten
    if not selected:
        # try quick heuristic: find name= or id= patterns
        m_name = re.search(r'name\s*[:=]\s*["\']?([^,"\n]+)', content, flags=re.IGNORECASE)
        m_id = re.search(r'id\s*[:=]\s*["\']?([^,"\n]+)', content, flags=re.IGNORECASE)
        if m_name:
            selected['name'] = m_name.group(1).strip()
        if m_id:
            selected['id'] = m_id.group(1).strip()

    if "name" in selected and "id" in selected:
        return {k: selected[k] for k in ("name", "id", "reason") if k in selected}

    # Last attempt: maybe content itself is just a single token like "NONE" or "0. Aspirin"
    s = content.strip()
    if s.upper() == "NONE":
        return {"name": "NONE", "id": "NONE", "reason": "LLM returned NONE"}
    # try to extract a leading index + dot: "0.Aspirin" or "0. Aspirin"
    m = re.match(r'^\s*(\d+)\s*[\.\:\)]\s*(.+)$', s)
    if m:
        print({"name": m.group(2).strip(), "id": str(int(m.group(1).strip())), "reason": ""})
        return {"name": m.group(2).strip(), "id": str(int(m.group(1).strip())), "reason": ""}

    raise ValueError(f"Failed to extract selected_entity from LLM output. Preview: {content[:300]}")

# ------------------------------
# Logging (unchanged; ensure logs/ exists)
# ------------------------------
def init_logger(name=''):
    os.makedirs('logs', exist_ok=True)
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if reimported
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        handler = logging.FileHandler(
            f'logs/{name or "bedrock"}-{time.strftime("%Y%m%d-%H%M%S")}.log'
        )
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# ------------------------------
# Graph utils (unchanged)
# ------------------------------
def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h.lower(), t.lower(), relation=r.lower().strip())
    return G

def find_all_path_KG(question_entities, result_entities, G):
    path_all = []
    for q_entity in question_entities:
        for a_entity in result_entities:
            path_all += list(nx.all_shortest_paths(G, q_entity.lower(), a_entity.lower()))
    return path_all

# ------------------------------
# Embeddings (unchanged)
# ------------------------------
import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

def generate_node_embeddings(
        knowledge_graph_path: str = '/path/to/kg.csv',
        sapbert_tokenizer=None,
        sapbert_model=None,
        device="cpu",
        cache_file: str = 'node_embeddings_sapbert.pt',
    ):
    """
    Generate embeddings for all KG nodes (grouped by type) using SapBERT.
    Saves nodeemb_dict to cache_file and returns nodeemb_dict and node_name_dict.

    Returns:
        nodeemb_dict: dict mapping entity_type → torch.Tensor (N, d)
        node_name_dict: dict mapping entity_type → list of node names (strings)
    """
    # 1) load KG
    knowledge_graph = pd.read_csv(knowledge_graph_path, low_memory=False)

    # 2) load SapBERT tokenizer and model

    # 3) group by types and embed each group's names
    types = knowledge_graph['x_type'].unique()
    nodeemb_dict = {}
    node_name_dict = {}

    for t in types:
        print(f"Generating embeddings for type: {t}")
        # collect the unique names
        names = knowledge_graph.loc[knowledge_graph['x_type'] == t, 'x_name'].unique().tolist()
        node_name_dict[t] = names

        # embed in batches
        all_embs = []
        batch_size = 128
        for i in tqdm(range(0, len(names), batch_size), desc=f"Embedding {t}"):
            batch_names = names[i:i+batch_size]
            toks = sapbert_tokenizer(batch_names, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = sapbert_model(**toks)[0][:,0,:]  # CLS token embedding
            all_embs.append(out.cpu())

        embs = torch.cat(all_embs, dim=0)  # (N, d)
        nodeemb_dict[t] = embs

    # 4) save to cache
    torch.save({'nodeemb_dict': nodeemb_dict, 'node_name_dict': node_name_dict}, cache_file)
    print(f"Saved embeddings to {cache_file}")

    return nodeemb_dict, node_name_dict

# def generate_node_embeddings(knowledge_graph_path = '/path/to/kg.csv',
#                              emb_model_name = 'abhinand/MedEmbed-large-v0.1'):
#     knowledge_graph = pd.read_csv(knowledge_graph_path, low_memory=False)
#     emb_model = SentenceTransformer(emb_model_name)
#     emb_model.eval()

#     types = knowledge_graph['x_type'].unique()
#     nodeemb_dict = {}
#     for t in types:
#         print("generating embeddings for type: ", t)
#         entities_in_types = knowledge_graph.query('x_type=="{}"'.format(t))['x_name'].unique()
#         type_embeddings = emb_model.encode(list(entities_in_types))
#         nodeemb_dict[t] = type_embeddings
#     torch.save(nodeemb_dict, 'node_embeddings.pt')
#     return

# ------------------------------
# Similarity + KG select (unchanged logic)
# ------------------------------
def get_topk_similar_entities(entity, tokenizer, model, nodeemb_dict, node_name_dict,
                              k=100, filter_threshold=0.6, device=None):
    import torch
    import torch.nn.functional as F

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    entity_type = entity.get("type")
    if entity_type not in nodeemb_dict:
        return [], 0.0

    node_names = node_name_dict.get(entity_type, [])
    if not node_names:
        return [], 0.0

    # Move embeddings to device
    node_embs = nodeemb_dict[entity_type].to(device)                # (N, d)
    node_embs_norm = F.normalize(node_embs, p=2, dim=-1)

    # Tokenize and embed the query entity
    inputs = tokenizer([entity["name"]], padding=True, truncation=True,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        entity_emb = outputs.last_hidden_state[:,0,:]
    entity_emb = entity_emb.to(device)
    emb_norm   = F.normalize(entity_emb, p=2, dim=-1)                # (1, d)

    sims = (emb_norm @ node_embs_norm.T).squeeze(0).cpu()             # move to cpu for topk
    vals, idxs = torch.topk(sims, min(k, sims.size(0)))

    top1 = float(vals[0].item()) if vals.numel() > 0 else 0.0
    mask = vals > filter_threshold
    if mask.sum().item() == 0:
        return [], top1

    sel_idxs = idxs[mask].tolist()
    similar_entities = [node_names[i] for i in sel_idxs if i < len(node_names)]

    return similar_entities, top1


# ------------------------------
# Bedrock client + LLM wrappers
# ------------------------------
def _make_bedrock_client(region: Optional[str] = None):
    """
    Create a bedrock-runtime client. Region must be enabled for Bedrock.
    """
    region = "us-west-2"
    return boto3.client("bedrock-runtime", region_name=region)

def _engine_to_model(engine: str) -> Dict[str, Any]:
    if engine not in BEDROCK_MODELS:
        raise ValueError(f"Unknown engine '{engine}'. Known: {list(BEDROCK_MODELS.keys())}")
    return BEDROCK_MODELS[engine]

def compute_usage(response: Dict[str, Any], engine: str) -> Dict[str, float]:
    """
    Compute token usage and dollar cost from Bedrock Converse response.
    """
    usage = response.get("usage", {}) or {}
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)

    model_cfg = _engine_to_model(engine)
    cost = {
        "input": input_tokens * model_cfg["input_price"],
        "output": output_tokens * model_cfg["output_price"],
    }
    cost["total"] = cost["input"] + cost["output"]
    return cost

def _messages_for_converse(user_text: str,
                           system_text: Optional[str] = None) -> Dict[str, Any]:
    messages = [{"role": "user", "content": [{"text": user_text}]}]
    system = [{"text": system_text}] if system_text else None
    return messages, system

def run_llm(prompt: str,
            temperature: float = 0.0,
            max_tokens: int = 3000,
            engine: str = DEFAULT_ENGINE,
            max_attempt: int = 10,
            logger: Optional[logging.Logger] = None) -> str:
    """
    Bedrock Converse-based completion. Preserves your original signature.
    """
    global api_total_cost
    model_cfg = _engine_to_model(engine)
    model_id = model_cfg["model_id"]
    inference_profile_arn = model_cfg.get("inference_profile_arn")

    client = _make_bedrock_client()
    system_prompt = "You are an AI assistant that helps people find information."
    messages, system = _messages_for_converse(prompt, system_text=system_prompt)

    attempt = 0
    last_error = None
    while attempt < max_attempt:
        attempt += 1
        try:
            kwargs = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    # "topP": 0.9,  # optional
                },
            }
            if system:
                kwargs["system"] = system
            if inference_profile_arn:
                # Required for some models (e.g., certain Meta models in some accounts)
                kwargs["inferenceProfileId"] = inference_profile_arn

            resp = client.converse(**kwargs)

            # Extract text
            content_blocks = resp.get("output", {}).get("message", {}).get("content", [])
            result_text = ""
            for block in content_blocks:
                if "text" in block:
                    result_text += block["text"]
            if not result_text:
                # Some providers return tool results or empty text; guard against that
                result_text = json.dumps(resp.get("output", {}))

            # Cost accounting
            cost = compute_usage(resp, engine)
            api_total_cost += cost["total"]
            if logger:
                logger.info(f"[{engine}] inputTokens={resp.get('usage',{}).get('inputTokens',0)} "
                            f"outputTokens={resp.get('usage',{}).get('outputTokens',0)} "
                            f"cost=${cost['total']:.6f} (accum=${api_total_cost:.6f})")
            return result_text

        except ClientError as e:
            last_error = e
            # Common case: model requires an inference profile
            if e.response.get("Error", {}).get("Code") == "ValidationException":
                # Surface a clear message
                msg = e.response.get("Error", {}).get("Message", "")
                if logger:
                    logger.error(f"Bedrock ValidationException: {msg}")
                # If it mentions on-demand throughput, tell user to set inference profile
                if "on-demand throughput isn’t supported" in msg.lower() and not inference_profile_arn:
                    raise RuntimeError(
                        f"The model '{model_id}' requires an inference profile. "
                        f"Set BEDROCK_MODELS['{engine}']['inference_profile_arn'] to your profile ARN."
                    ) from e
            if logger:
                logger.warning(f"Bedrock client error (attempt {attempt}/{max_attempt}): {e}")
            time.sleep(2)

        except (BotoCoreError, Exception) as e:
            last_error = e
            if logger:
                logger.warning(f"Error calling Bedrock (attempt {attempt}/{max_attempt}): {e}")
            time.sleep(2)

    # If we exit loop without return, bubble up the last error
    raise RuntimeError(f"Bedrock call failed after {max_attempt} attempts: {last_error}")

# ------------------------------
# Your domain-specific LLM helpers (now backed by Bedrock)
# ------------------------------
def coarse_entity_extraction(text, temperature=0.0, max_tokens=3000, engine=DEFAULT_ENGINE):
    Extract_prompt = f"""You are a helpful, pattern-following medical assistant. 
Given the following clinical or biomedical text, extract **all entities** relevant to the domain.

### Allowed entity types (exact match only):
- gene/protein  
- drug  
- effect/phenotype  
- disease  
- biological_process  
- molecular_function  
- cellular_component  
- exposure  
- pathway  
- anatomy

### Output format
Respond only with the content between the delimiters below. Each entity should be written on its own line in the format:

    id: <unique string>
    type: <one of the 10 allowed types>
    name: <canonical entity string>

Delimiters:
BEGIN OUTPUT
id: 1
type: disease
name: glioblastoma

id: 2
type: drug
name: Gliadel Wafer
END OUTPUT

---

Text:
{text}
"""
    return run_llm(Extract_prompt, temperature, max_tokens, engine)


def most_correlated_entity_selection(question, query_entity, similar_entities,
    temperature = 0.0, max_tokens = 3000, engine=DEFAULT_ENGINE):
    prompt = f"""You are a helpful, pattern-following medical assistant.

Given a medical question and a query entity extracted from it, select the most correlated entity from a list.
If none are suitable, return "NONE".

Only include output between the delimiters.

### Format:
BEGIN OUTPUT
name: <selected_entity_name or NONE>
id: <index from 0 to N-1 or NONE>
reason: <short reason>
END OUTPUT

---

Input:
Question: {question}
Query Entity: {query_entity}
Similar Entities: {', '.join(f"{i}. {e}" for i, e in enumerate(similar_entities))}

Respond with only the selected output between BEGIN and END.
"""
    return run_llm(prompt, temperature, max_tokens, engine)


def most_correlated_path_selection(question: str,
                                   paths_text: str,
                                   options: Union[dict, None] = None,
                                   topK: int = 2,
                                   temperature: float = 0.0,
                                   max_tokens: int = 2000,
                                   engine: str = DEFAULT_ENGINE) -> list:
    """
    Ask the LLM to rank/select up to topK most relevant paths for the QUESTION (and optionally options).
    Returns: list of dicts: [{"ranking": "1", "path": "a->b->c", "reason": "...", "score": float}, ...]
    """
    # include options lightly in the prompt if provided
    options_block = ""
    if options:
        if isinstance(options, dict):
            options_block = "\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
        else:
            options_block = f"\nOptions: {str(options)}"

    prompt = f"""You are a helpful, pattern-following medical assistant.
Given the medical question below and candidate relation paths, select up to {topK} most relevant paths that help answer the question.
Respond **only** with the section between the delimiters. Each path block must contain:
ranking: <1..{topK}>
path: <node1->node2->...>
reason: <short justification of relevance>

BEGIN OUTPUT
Question: {question}
{options_block}

Candidate paths:
{paths_text}

# Example Output:
ranking: 1
path: drug->gene->disease
reason: Explains the drug's mechanistic link to the disease.
END OUTPUT
"""
    raw = run_llm(prompt, temperature=temperature, max_tokens=max_tokens, engine=engine)
    # parse_llm_output_block should return a dict mapping keys, or raise
    parsed = {}
    try:
        parsed = parse_llm_output_block(raw)  # returns dict or raises
    except Exception:
        # fallback: attempt to find path lines in the raw text
        parsed = {"raw": raw}

    # try to extract items in a robust way
    results = []
    if isinstance(parsed, dict) and parsed:
        # If parsed contains multiple ranking/path/reason entries, reconstruct them
        # parsed may be {'ranking': '1', 'path':'a->b', 'reason':'...'} or more complex.
        # We'll attempt to extract repeated groups by searching raw text as fallback.
        text = raw
        # Find all "ranking:" occurrences and parse following lines
        blocks = []
        current = []
        for line in text.splitlines():
            if line.strip().lower().startswith("ranking:"):
                if current:
                    blocks.append("\n".join(current))
                current = [line]
            elif current:
                current.append(line)
        if current:
            blocks.append("\n".join(current))

        for block in blocks:
            d = {}
            for ln in block.splitlines():
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    d[k.strip().lower()] = v.strip()
            if "path" in d:
                # attach a crude score placeholder (LLM doesn't output numeric score)
                d.setdefault("score", None)
                results.append({"ranking": d.get("ranking", None),
                                "path": d["path"],
                                "reason": d.get("reason", ""),
                                "llm_score": None})
    else:
        # fallback: parse raw lines with pattern "N: a->b->c"
        for ln in raw.splitlines():
            m = re.search(r'(\d+)\s*[:\-]\s*(.+->.+)', ln)
            if m:
                results.append({"ranking": m.group(1), "path": m.group(2).strip(), "reason": ""})

    # Return up to topK results
    return results[:topK]


def path_sampling(path_all: list,
                  question: str,
                  options: Union[dict, None] = None,
                  topK_reasoning_paths: int = 3,
                  max_path_number_per_group: int = 50,
                  engine: str = DEFAULT_ENGINE,
                  max_tokens=256, 
                  logger: Optional[logging.Logger] = None) -> list:
    """
    Groups paths by (start,end), optionally subsamples big groups, and uses the LLM
    to pick the top paths per group *based on the question/options*.
    Returns a flat list of sampled paths (each path is a list of node strings).
    """
    # Build groups: key = (start_node, end_node)
    path_groups = {}
    for path in path_all:
        if len(path) < 2:
            continue
        key = (path[0], path[-1])
        path_groups.setdefault(key, []).append(path)

    sampled_paths = []
    for key, paths in path_groups.items():
        if logger:
            logger.info(f"Sampling for Path group: {key} (size={len(paths)})")
        # limit group size before asking LLM
        if len(paths) > max_path_number_per_group:
            paths = random.sample(paths, max_path_number_per_group)

        # create text block for the LLM
        text_for_group_paths = '\n'.join([f"{idx+1}:{'->'.join(p)}" for idx, p in enumerate(paths)])
        # ask LLM to choose the most correlated paths for this question (no gold answer)
        group_result = most_correlated_path_selection(question, text_for_group_paths, options=options,
                                                     topK=topK_reasoning_paths, engine=engine, max_tokens = max_tokens)
        # parse returned list of dicts from most_correlated_path_selection
        if group_result:
            for item in group_result:
                path_str = item.get("path") or item.get("Path") or ""
                if path_str:
                    sampled_paths.append(path_str.split("->"))
        else:
            # if LLM failed, fall back to top-k shortest paths or random
            # choose the shortest paths (prefer shorter path length)
            paths_sorted = sorted(paths, key=lambda p: len(p))
            sampled_paths.extend(paths_sorted[:topK_reasoning_paths])

    # dedupe preserve order
    seen = set()
    unique = []
    for p in sampled_paths:
        key = "->".join(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


    
def llm_generate_answer_with_reasoning(question, options, reasoning, engine=DEFAULT_ENGINE):
    prompt = f"""
    You are an expert in the medical domain. You need to answer the following question based on the provided reasoning.
    YOU MUST USE THE PROVIDED REASONING TO ANSWER THE QUESTION.
    If the answer choices are provided, choose ONE answer from the answer choices.

    Question:
    {question}
    {options}
    Reasoning:
    {reasoning}
    """
    return run_llm(prompt, engine=engine)

def llm_judge_answer(llm_output, answer, engine=DEFAULT_ENGINE):
    prompt = f"""
    You are an expert in the medical domain. Given a correct answer, and the answer from a medical student,
    judge whether the student's answer matches the correct answer. Respond strictly with 'True' or 'False'.

    Correct answer:
    {answer}

    Answer from medical student:
    {llm_output}
    """
    return run_llm(prompt, engine=engine)
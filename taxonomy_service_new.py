# taxonomy_service_new.py by Caleb Lin
# ------------------------------------------------------------------------------
# Purpose
#   Parse a natural-language request into a 3-level taxonomy:
#       { "domain": <Domain>, "dimension": <Dimension>, "area": <Area> }
#
# Usage
#   - Public entrypoint: `parse_business_query(payload)`
#   - Payload can be:
#       * Metadata dict from Orchestrator, and ultimately from the competition platform(preferred)
#       * Raw string query (backward-compatible, should be avoided according to Joanna)
#   - Service returns only DB-valid labels (no "uncertain").
#
# How It Works
#   1. Receives the **entire metadata object** from Orchestrator.
#      - NOTE: Orchestrator must no longer pre-extract `query` itself, and # of bots is not needed.
#      - Taxonomy Service is responsible for extracting `metadata.business_taxonomy.query`
#        (or fallback keys like `metadata.query`, `prompt`, etc.).
#   2. Database narrowing:
#      - List all Domains from `domains` table.
#      - Ask LLM to pick the best Domain from candidates.
#      - With the chosen Domain ID, list Dimensions from `dimensions`.
#      - Again ask LLM to pick the best Dimension.
#      - With the chosen Dimension ID, list Areas from `areas`.
#      - LLM selects the best Area.
#   3. Return the structured result, only top 3 fields are needed by Orchestrator/Business Service:
#       {
#         "domain": <str>,
#         "dimension": <str>,
#         "area": <str>,
#         "per_level": [ { "level":..., "candidates":..., "label":..., ... }, ... ], # an explanation log
#         "reasoning": "resolved all levels"  # an explanation log in natural language
#       }
#
# Dependencies
#   - MySQL `domains`, `dimensions`, `areas` tables, and later `cm_collections`, which is still empty now.
#   - P2P data service call: `db_query` (no hard-coded SQL in this file).
#   - OpenAI Responses API with structured JSON Schema output. Note: it's connected to my personal secret for now, I set a budget limit, so DO NOT abuse it.
#
# Current Limitations
#   - Uses a general LLM (gpt-4o-mini) rather than a fine-tuned model.
#   - Fuzzy matching against Collection names in ChromaDB is NOT implemented yet (TODO, but need clarification).
#   - Confidence values from the LLM are uncalibrated; only relative. Also the logic can be improved, for example, to handle low confidence is not implemented.
#
# TODO (next steps)
#   - Orchestrator Service: ensure it forwards the full metadata (not just query string).
#   - Add fuzzy Collection matching (map parsed taxonomy area → ChromaDB collection) - TODO not clear yet, should be a separate service? At least should have another meeting with Joanna.
#   - Swap OpenAI API for the fine-tuned local model once available.
#   - Implement monitoring/logging of misclassifications to improve prompts. (Optional)
# ------------------------------------------------------------------------------

import os
import sys
import json
import logging
import asyncio
from typing import Optional, Dict, Any, List

import metaml
import node_connection
from metaml import NodeType
from messaging.message import Message, MessageType, MessageData

from openai import OpenAI
from dotenv import load_dotenv
from rapidfuzz import fuzz, process  # pip install rapidfuzz
USE_LOCAL = os.getenv("LLM_BACKEND", "").lower() in ("local", "none", "fuzzy")

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("TaxonomyService")
DEBUG = True

# ------------------------------------------------------------------------------
# OpenAI (Responses API + Structured Outputs) configuration - a pre-trained LLM Model
#  Todo: should be replaced by fine-tuned local LLM later
# ------------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")# validated in tests
from collections import defaultdict
import chromadb
from chromadb.config import Settings
DOMAIN_PRIOR_MIN_RATIO   = float(os.getenv("DOMAIN_PRIOR_MIN_RATIO", "1.3"))  # best / second-best
RETRIEVAL_DOMAIN_MIN_SIM = float(os.getenv("RETRIEVAL_DOMAIN_MIN_SIM", "0.60"))

VECTOR_HOST = os.getenv("VECTOR_HOSTNAME", "chroma")
VECTOR_PORT = int(os.getenv("VECTOR_PORT", "8000"))
AREA_COLL   = os.getenv("TAXONOMY_AREA_COLLECTION", "taxonomy_areas_v1")
RETRIEVAL_TOPK = int(os.getenv("RETRIEVAL_TOPK", "20"))
RETRIEVAL_STRONG_SIM = float(os.getenv("RETRIEVAL_STRONG_SIM", "0.55"))  # choose area directly if sim ≥ this
# top-level imports
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

EMBED_MODEL = os.getenv("VECTOR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBED_FN = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def _chroma():
    return chromadb.HttpClient(
        host=VECTOR_HOST, port=VECTOR_PORT,
        settings=Settings(allow_reset=False, anonymized_telemetry=False)
    )

async def _all_area_records() -> List[Dict[str, Any]]:
    out = []
    domains = await _list_domains()
    for dom in domains:
        dims = await _list_dimensions(dom["id"])
        for dim in dims:
            areas = await _list_areas(dim["id"])
            for a in areas:
                out.append({
                    "domain_id": dom["id"], "domain": dom["name"],
                    "dimension_id": dim["id"], "dimension": dim["name"],
                    "area_id": a["id"], "area": a["name"]
                })
    return out

AREA_SYNONYMS = {
    ("FinTech","Payments","Fraud Detection"): [
        "card", "cards", "fraud", "fraudulent", "chargeback", "chargebacks",
        "dispute", "disputes", "authorization", "risk score", "transaction monitoring"
    ],
    ("FinTech","Payments","Chargeback Management"): [
        "chargeback", "representment", "issuer", "acquirer", "dispute", "merchant", "friendly fraud"
    ],
    # add more tuples that exist in your DB as needed
}

def _doc_for_row(r):
    syn = AREA_SYNONYMS.get((r["domain"], r["dimension"], r["area"]), [])
    extra = f" — keywords: {' '.join(syn)}" if syn else ""
    return f"{r['area']} — {r['dimension']} — {r['domain']}{extra}"

async def _ensure_area_index() -> "chromadb.api.models.Collection.Collection":
    client = _chroma()
    try:
        # IMPORTANT: attach embedding_function even for existing collections
        coll = client.get_collection(AREA_COLL, embedding_function=EMBED_FN)
    except Exception:
        coll = client.get_or_create_collection(
            name=AREA_COLL,
            embedding_function=EMBED_FN,
            metadata={"schema": "areas_v1"}
        )

    if coll.count() == 0:
        rows = await _all_area_records()
        if rows:
            ids = [f"area:{r['area_id']}" for r in rows]
            # Enrich docs slightly to help domain disambiguation
            docs = [_doc_for_row(r) for r in rows]
            metas = [{
                "area_id": r["area_id"], "area": r["area"],
                "dimension_id": r["dimension_id"], "dimension": r["dimension"],
                "domain_id": r["domain_id"], "domain": r["domain"]
            } for r in rows]
            coll.add(ids=ids, documents=docs, metadatas=metas)
    return coll

    
async def _retrieve_area_prior(query: str, top_k: int = RETRIEVAL_TOPK) -> Optional[Dict[str, Any]]:
    if not query:
        return None
    coll = await _ensure_area_index()

    def _rows(res):
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [{"md": (m or {}), "sim": 1.0 - float(d)} for m, d in zip(metas, dists)]

    # --- Pass 1: global ---
    res1 = coll.query(query_texts=[query], n_results=top_k, include=["metadatas","distances"])
    rows1 = _rows(res1)
    if not rows1:
        return None

    # domain scores
    dom_scores = {}
    for r in rows1:
        dom = r["md"].get("domain")
        if dom: dom_scores[dom] = dom_scores.get(dom, 0.0) + r["sim"]
    if not dom_scores:
        return None
    best_domain = max(dom_scores.items(), key=lambda kv: kv[1])[0]

    # --- Pass 2: filter by domain ---
    res2 = coll.query(
        query_texts=[query],
        n_results=min(12, top_k),
        include=["metadatas","distances"],
        where={"domain": {"$eq": best_domain}}
    )
    rows2 = _rows(res2)
    dim_scores = {}
    for r in rows2:
        dim = r["md"].get("dimension")
        if dim: dim_scores[dim] = dim_scores.get(dim, 0.0) + r["sim"]
    # if somehow empty, fall back to best from pass1
    if not dim_scores:
        top1 = max(rows1, key=lambda r: r["sim"])
        md = top1["md"]; return {
            "domain": md.get("domain"),
            "dimension": md.get("dimension"),
            "area": md.get("area"),
            "top_sim": float(top1["sim"]),
            "domain_scores": dom_scores,
            "dimension_scores": {}
        }
    best_dim = max(dim_scores.items(), key=lambda kv: kv[1])[0]

    # --- Pass 3: filter by domain+dimension ---
    res3 = coll.query(
        query_texts=[query],
        n_results=min(8, top_k),
        include=["metadatas","distances"],
        where={
            "$and": [
                {"domain":    {"$eq": best_domain}},
                {"dimension": {"$eq": best_dim}}
            ]
        }
    )
    rows3 = _rows(res3)
    top = max(rows3 or rows2, key=lambda r: r["sim"])  # fallback if rows3 empty
    md = top["md"] or {}
    return {
        "domain":    md.get("domain", best_domain),
        "dimension": md.get("dimension", best_dim),
        "area":      md.get("area"),
        "top_sim":   float(top["sim"]),
        "domain_scores": dom_scores,
        "dimension_scores": dim_scores,
    }



def _get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set; put it in .env")
    return OpenAI(api_key=OPENAI_API_KEY)
    
def llm_pick_label(query: str, level: str, candidates: List[str], model: str = OPENAI_MODEL) -> Dict[str, Any]:
    if not candidates:
        raise ValueError(f"No candidates at level={level}")

    # ---- Local fuzzy (no OpenAI needed) ----

    if USE_LOCAL or not OPENAI_API_KEY:
        best, score, _ = process.extractOne(query or "", candidates, scorer=fuzz.token_set_ratio)
        return {
            "label": best,
            "confidence": float(score) / 100.0,
            "reasoning": f"local fuzzy match (token_set_ratio={score})"
        }
    # ---- OpenAI path (unchanged) ----
    client = _get_openai_client()

    instructions = (
        "You are a taxonomy router for the MetaML platform at Citi. "
        "MetaML taxonomy has 3 levels: Domain -> Dimension -> Area. "
        "Select exactly one label from the provided candidates for the current level. "
        "Choose one of the given candidates verbatim; do not invent labels. "
        "Return a short reasoning (1–2 sentences) and a confidence in [0,1]."
    )
    user_msg = (
        f'Query: "{query}"\n'
        f'Level: "{level}"\n'
        f"CandidateLabels: {json.dumps(candidates, ensure_ascii=False)}"
    )
    schema_only = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": candidates},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"}
        },
        "required": ["label", "confidence", "reasoning"],
        "additionalProperties": False
    }
    try:
        resp = client.responses.create(
            model=model,
            instructions=instructions,
            input=[{"role": "user", "content": [{"type": "input_text", "text": user_msg}]}],
            text={"format": {"type": "json_schema", "name": "TaxonomyPick", "strict": True, "schema": schema_only}}
        )
        parsed = getattr(resp, "output_parsed", None)
        if not parsed:
            txt = getattr(resp, "output_text", "") or ""
            parsed = json.loads(txt)
        if parsed.get("label") not in candidates:
            parsed["label"] = candidates[0]
        conf = parsed.get("confidence", 0.0)
        parsed["confidence"] = float(conf) if isinstance(conf, (int, float)) else 0.0
        parsed["confidence"] = max(0.0, min(1.0, parsed["confidence"]))
        return parsed
    except Exception as e:
        logger.error(f"OpenAI call failed at level={level}: {e}")
        return {"label": candidates[0], "confidence": 0.0, "reasoning": "fallback-first-candidate"}

#def llm_pick_label(query: str, level: str, candidates: List[str], model: str = OPENAI_MODEL) -> Dict[str, Any]:
#    """
#    Choose exactly ONE label from the provided `candidates` for the given taxonomy `level`.
#    Uses OpenAI Responses API with Structured Outputs (JSON Schema + enum).
#    Always returns a DB-valid candidate. On any error, returns the first candidate.
#    Return schema: { "label": str, "confidence": float, "reasoning": str }
#    """
#    if not candidates:
#        raise ValueError(f"No candidates at level={level}")
#
#    client = _get_openai_client()
#
#    instructions = (
#        "You are a taxonomy router for the MetaML platform at Citi. "
#        "MetaML taxonomy has 3 levels: Domain -> Dimension -> Area. "
#        "Select exactly one label from the provided candidates for the current level. "
#        "Domains: broad industries (e.g., FinTech, HealthTech). "
#        "Dimensions: sub-topics within a domain (e.g., Risk Assessment, Payments, Compliance). "
#        "Areas: fine-grained applications (e.g., Fraud Detection). "
#        "Choose one of the given candidates verbatim; do not invent labels. "
#        "Return a short reasoning (1–2 sentences) and a confidence in [0,1]."
#    )
#
#    user_msg = (
#        f'Query: "{query}"\n'
#        f'Level: "{level}"\n'
#        f"CandidateLabels: {json.dumps(candidates, ensure_ascii=False)}"
#    )
#
#    schema_only = {
#        "type": "object",
#        "properties": {
#            "label": {"type": "string", "enum": candidates},
#            "confidence": {"type": "number"},
#            "reasoning": {"type": "string"}
#        },
#        "required": ["label", "confidence", "reasoning"],
#        "additionalProperties": False
#    }
#
#    try:
#        resp = client.responses.create(
#            model=model,
#            instructions=instructions,
#            input=[{
#                "role": "user",
#                "content": [{"type": "input_text", "text": user_msg}]
#            }],
#            text={
#                "format": {
#                    "type": "json_schema",
#                    "name": "TaxonomyPick",
#                    "strict": True,
#                    "schema": schema_only
#                }
#            }
#        )
#
#        parsed = getattr(resp, "output_parsed", None)
#        if not parsed:
#            txt = getattr(resp, "output_text", "") or ""
#            parsed = json.loads(txt)
#
#        if parsed.get("label") not in candidates:
#            parsed["label"] = candidates[0]
#
#        conf = parsed.get("confidence", 0.0)
#        parsed["confidence"] = float(conf) if isinstance(conf, (int, float)) else 0.0
#        parsed["confidence"] = max(0.0, min(1.0, parsed["confidence"]))
#        return parsed
#
#    except Exception as e:
#        logger.error(f"OpenAI call failed at level={level}: {e}")
#        return {"label": candidates[0], "confidence": 0.0, "reasoning": "fallback-first-candidate"}

async def llm_pick_label_async(query: str, level: str, candidates: List[str], timeout: float = 8.0):
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, llm_pick_label, query, level, candidates),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"OpenAI timeout at level={level}; using fallback candidate")
        return {"label": candidates[0], "confidence": 0.0, "reasoning": "timeout-fallback"}
    except Exception as e:
        logger.error(f"OpenAI exception at level={level}: {e}; using fallback")
        return {"label": candidates[0], "confidence": 0.0, "reasoning": "exception-fallback"}


# ------------------------------------------------------------------------------
#Process the metadata from the function call to get the query, handles different formats
# ------------------------------------------------------------------------------
def _fingerprint_collection(payload: Any) -> Optional[str]:
    """
    Optional hint from upstream to bias retrieval.
    Looks for a collection/fingerprint id in common places.
    """
    if isinstance(payload, (list, tuple)) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None

    fp = payload.get("fingerprint") or payload.get("metadata", {}).get("fingerprint")
    if isinstance(fp, dict):
        return fp.get("collection") or fp.get("fingerprint_id")
    return None

def _extract_query_from_payload(payload: Any) -> str:
    """
    Extract the natural-language query from a payload that may be:
      - str: treated as the query directly (backward compatible)
      - dict: we try the following paths in order:
          payload["metadata"]["business_taxonomy"]["query"]
          payload["metadata"]["query"]
          payload["business_taxonomy"]["query"]
          payload["query"]
          payload["prompt"]
    If all missing, returns empty string "".
    """
    # Back-compat: plain string
    if isinstance(payload, str):
        return payload.strip()

    # If tuple/list, unwrap first element (how P2P args are often sent)
    if isinstance(payload, (list, tuple)) and payload:
        payload = payload[0]

    if not isinstance(payload, dict):
        return str(payload or "").strip()

    # Try nested keys in priority order
    def _get(d, path):
        cur = d
        for k in path:
            if isinstance(cur, dict) and k in cur and cur[k] is not None:
                cur = cur[k]
            else:
                return None
        return cur

    candidates = [
        ["metadata", "business_taxonomy", "query"],
        ["metadata", "query"],
        ["business_taxonomy", "query"],
        ["query"],
        ["prompt"],
    ]
    for p in candidates:
        v = _get(payload, p)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


# ------------------------------------------------------------------------------
# DB access via DataService (P2P): db_query (a method defined in the metaML class)
# ------------------------------------------------------------------------------
def _normalize_query(q):
    if isinstance(q, (list, tuple)):
        q = q[0] if q else ""
    if not isinstance(q, str):
        q = str(q)
    return q.strip()
    
def _unwrap_envelope(x):
    # accept python obj or JSON string
    x = _loads_if_str(x)
    if isinstance(x, dict):
        # Data node returns MessageData-like dict; rows are under 'response'
        r = x.get("response")
        if r is not None:
            r = _loads_if_str(r)
            return r
    return x
def _loads_if_str(x):
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", "ignore")
        except Exception:
            return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def _force_list_of_dicts(x, label_for_logs="payload"):
    x = _unwrap_envelope(x)
    if isinstance(x, list) and all(isinstance(i, dict) for i in x):
        return x
    logger.error("%s not list[dict]; got %s :: %r", label_for_logs, type(x), str(x)[:200])
    return []
def _domain_prior_strong(prior: Dict[str, Any]) -> bool:
    scores = sorted(prior["domain_scores"].items(), key=lambda kv: kv[1], reverse=True)
    if not scores:
        return False
    best_score = scores[0][1]
    second_score = scores[1][1] if len(scores) > 1 else 0.0
    ratio = best_score / (second_score + 1e-9)
    return prior.get("top_sim", 0.0) >= RETRIEVAL_DOMAIN_MIN_SIM and ratio >= DOMAIN_PRIOR_MIN_RATIO




async def _db_query_rows(sql: str) -> List[List]:
    """
    Call the Data node via P2P using node.db_query(None, sql).
    DataService 'db.query' returns JSON-encoded list of row tuples.
    """
    try:
        resp_str = await asyncio.wait_for(node.db_query(None, sql), timeout=5.0)
    except asyncio.TimeoutError:
        logger.error(f"DataService timeout running SQL: {sql}")
        return []
    except Exception as e:
        logger.error(f"DataService error running SQL: {e} (SQL={sql})")
        return []

    if not resp_str:
        return []
    try:
        rows = json.loads(resp_str)
        return rows if isinstance(rows, list) else []
    except Exception:
        logger.debug(f"Non-JSON payload from DataService: {resp_str[:100]}")
        return []

async def _list_domains() -> List[Dict[str, Any]]:
    rows_any = await node.db_list_domains(None)
    rows = _force_list_of_dicts(rows_any, "domains")
    logger.debug("domains fetched: %d", len(rows))
    return rows

async def _list_dimensions(domain_id: int) -> List[Dict[str, Any]]:
    rows_any = await node.db_list_dimensions(None, int(domain_id))
    rows = _force_list_of_dicts(rows_any, "dimensions")
    logger.debug("dimensions fetched (domain_id=%s): %d", domain_id, len(rows))
    return rows

async def _list_areas(dimension_id: int) -> List[Dict[str, Any]]:
    rows_any = await node.db_list_areas(None, int(dimension_id))
    rows = _force_list_of_dicts(rows_any, "areas")
    logger.debug("areas fetched (dimension_id=%s): %d", dimension_id, len(rows))
    return rows




# ------------------------------------------------------------------------------
# MVP hierarchical narrowing: Domain -> Dimension -> Area (LLM + DB), according to the meeting with Joanna on Aug 25th
# Steps: 1) list candidates from DB; 2) pick one via LLM; 3) narrow down for next level
# Overwrite the previous classify_hierarchy function by summer 2024, which is hard-coded and not DB-backed
# ------------------------------------------------------------------------------
async def classify_hierarchy_mvp(user_query: str) -> Dict[str, Any]:
    per_level: List[Dict[str, Any]] = []

    # ---------- Retrieval prior ----------
    prior = await _retrieve_area_prior(user_query)
    prior_reason = None

    # ---------- Domain ----------
    domains = await _list_domains()
    if not domains:
        raise RuntimeError("No domains available in DB.")
    domain_names = [str(d.get("name", "")) for d in domains]

    # Prefer Chroma prior first (no strict gating), then model pick
    if prior and prior.get("domain") in domain_names:
        ordered_domains = [prior["domain"]] + [n for n in domain_names if n != prior["domain"]]
        r1 = {
            "label": prior["domain"],
            "confidence": 1.0,  # you can compute a ratio if you prefer
            "reasoning": "vector prior"
        }
    else:
        r1 = await llm_pick_label_async(user_query, "domain", domain_names)
        ordered_rest = sorted(
            [n for n in domain_names if n != r1["label"]],
            key=lambda n: fuzz.WRatio(user_query, n),
            reverse=True
        )
        ordered_domains = [r1["label"]] + ordered_rest

    # pick first domain that actually has dimensions
    domain, dims = None, []
    for name in ordered_domains:
        cand = next((d for d in domains if d["name"] == name), None)
        if not cand:
            continue
        cand_dims = await _list_dimensions(cand["id"])
        if cand_dims:
            domain = cand
            dims = cand_dims
            break
    if not domain:
        raise RuntimeError("No dimensions available in DB.")

    per_level.append({
        "level": "domain",
        "candidates": domain_names,
        "label": domain["name"],
        "confidence": r1.get("confidence", 0.0),
        "reasoning": r1.get("reasoning", "")
    })
    domain_id = domain["id"]

    # ---------- Dimensions under chosen domain ----------
    dim_names = [str(x.get("name", "")) for x in dims]
    if prior and prior.get("dimension") in dim_names and prior.get("domain") == domain["name"]:
        r2 = {
            "label": prior["dimension"],
            "confidence": max(0.0, min(1.0, prior.get("top_sim", 0.0))),
            "reasoning": "vector prior (top area lies in this dimension)"
        }
    else:
        r2 = await llm_pick_label_async(user_query, "dimension", dim_names)
    per_level.append({"level": "dimension", "candidates": dim_names, **r2})
    dimension = next((x for x in dims if x["name"] == r2["label"]), dims[0])
    dimension_id = dimension["id"]

    # ---------- Areas under chosen dimension ----------
    areas = await _list_areas(dimension_id)
    if not areas:
        logger.warning(f"No areas under DIMENSION_ID={dimension_id}. Searching siblings for fallback.")
        fallback_area = None
        for d in dims:
            if d["id"] == dimension_id:
                continue
            cand = await _list_areas(d["id"])
            if cand:
                fallback_area, areas = cand[0], cand
                break
        if not fallback_area:
            raise RuntimeError(f"No areas available under DOMAIN_ID={domain_id}.")
        r3 = {"label": fallback_area["name"], "confidence": 0.0, "reasoning": "fallback-first-available-area"}
        per_level.append({"level": "area", "candidates": [a["name"] for a in areas], **r3})
        chosen_area = fallback_area
    else:
        area_names = [str(a.get("name", "")) for a in areas]
        if prior and prior["area"] in area_names and prior.get("top_sim", 0.0) >= RETRIEVAL_STRONG_SIM \
           and prior["dimension"] == dimension["name"] and prior["domain"] == domain["name"]:
            r3 = {"label": prior["area"], "confidence": prior["top_sim"], "reasoning": "vector prior (high similarity)"}
        else:
            r3 = await llm_pick_label_async(user_query, "area", area_names)
        per_level.append({"level": "area", "candidates": area_names, **r3})
        chosen_area = next((a for a in areas if a["name"] == r3["label"]), areas[0])

    return {
        "domain": domain["name"],
        "dimension": dimension["name"],
        "area": chosen_area["name"],
        "per_level": per_level,
        "reasoning": "resolved all levels"
    }


    

# ------------------------------------------------------------------------------
# Public entrypoint
# ------------------------------------------------------------------------------
async def parse_business_query(payload: Any) -> Dict[str, Any]:
    """
    NL -> taxonomy (domain, dimension, area)
    Primary:
      - Accept ANY payload (metadata/full JSON body/string).
      - Extract the NL query inside this function (no pre-processing required by Orchestrator, which has aligned with Joanna).
      - Run LLM + DB narrowing.
    Fallback:
      - If anything fails, return first DB-backed candidate per level (still no 'uncertain').
    """

    query = _extract_query_from_payload(payload)
    query = _normalize_query(query)
    logger.info(f"PROMPT_RECEIVED(parse_business_query): {query}")

    if not query:
        logger.warning("No query found in payload; continuing (will still select DB-backed first candidates if needed).")

    # 1) LLM + DB narrowing (primary path)
    
    try:
        narrowed = await classify_hierarchy_mvp(query)
        # Keep full detail (per_level + reasoning), and add num_bots
        narrowed["num_bots"] = 1
        # Normalize the wording of reasoning to match your sample exactly
        narrowed["reasoning"] = "resolved all levels"
        logger.info(f"[llm+db] Taxonomy parsed: {narrowed}")
        return narrowed

    except Exception as e:
        logger.warning(f"[llm+db] failed: {e}")

    # 2) Last-resort fallback (DB-backed first choices), still no 'uncertain'
    logger.warning("[fallback] using first-available candidates as a last resort.")
    try:
        domains = await _list_domains()
        d = domains[0]
        dims = await _list_dimensions(d["id"])
        dim = dims[0]
        areas = await _list_areas(dim["id"])
        area = areas[0] if areas else {"name": "general_area"}
        return {"domain": d["name"], "dimension": dim["name"], "area": area["name"], "num_bots": 1}
    except Exception as e:
        logger.error(f"[fallback] failed to pull from DB as well: {e}")
        return {"domain": "FinTech", "dimension": "general", "area": "general_area", "num_bots": 1}

# ------------------------------------------------------------------------------
# P2P message handler (MetaML framework)
# ------------------------------------------------------------------------------
async def handle_client(reader: 'asyncio.StreamReader', writer: 'asyncio.StreamWriter'):
    """
    Taxonomy node handler.
    Supports: parse_business_query(query: str) -> {domain, dimension, area}
    """
    connection = node_connection.NodeConnection(reader, writer)
    msg = await connection.read_message()
    logger.info(f"Handle_client is called: {msg}")

    if msg.message_type == MessageType.QUERY:
        request_body = metaml.parse_message_data(msg.message_data)
        func, args = request_body.method, request_body.args

        response: Optional[Message] = None

        if func in metaml.taxonomy_service_apis:
            try:
                if func == 'parse_business_query':
                    logger.info("Handle_client(parse_business_query)")
                    # Treat args as a generic payload (string OR dict OR list/tuple)
                    payload = args if not isinstance(args, (list, tuple)) else (args[0] if args else {})
                    taxonomy = await parse_business_query(payload)
                    response = Message(
                        message_type=MessageType.RESPONSE,
                        message_data=MessageData(method=func, args=None, response=taxonomy)
                    )
                    logger.info(f"Sending RESPONSE(parse_business_query): {taxonomy}")

                else:
                    response = Message(MessageType.ERROR)

            except Exception as e:
                logger.error(f"TaxonomyService error in {func}: {e}")
                fallback = {"domain": "FinTech", "dimension": "general", "area": "general_area", "num_bots": 1}
                response = Message(
                    message_type=MessageType.RESPONSE,
                    message_data=MessageData(method=func, args=None, response=fallback)
                )

        elif func in metaml.business_service_apis:
            await metaml.forward_to(NodeType.Business, node, connection, msg)
            return
        elif func in metaml.context_service_apis:
            await metaml.forward_to(NodeType.Context, node, connection, msg)
            return
        elif func in metaml.metrics_service_apis:
            await metaml.forward_to(NodeType.Metrics, node, connection, msg)
            return
        else:
            response = Message(MessageType.ERROR)

        if response:
            await connection.send_message(response)

    elif msg.message_type == MessageType.LIST:
        nearest_nodes = await node.nearest_of_type()
        await connection.send_message(Message(
            MessageType.RESPONSE,
            MessageData(method=None, args=None, response=nearest_nodes)
        ))
    elif msg.message_type == MessageType.PING:
        await connection.send_message(Message(MessageType.HELO, ))

# Bootstrap node
port, bootstrap_node = metaml.fetch_args()
node = metaml.MetaMLNode(node_type=metaml.NodeType.Taxonomy,
                         client_handler=handle_client,
                         port=port,
                         bootstrap_node=bootstrap_node)
asyncio.run(node.init_dht())


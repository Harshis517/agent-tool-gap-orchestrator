import sys
import json
import metaml
import pymysql
from sqlalchemy import create_engine, text
import traceback
import logging
import asyncio
from itertools import chain  # (unused but kept if you reference it elsewhere)
import pandas as pd
import time
import os  
import node_connection
from metaml import NodeType
from messaging.message import *
from vectordb_jumptable import handle_collection_request
import jumptable_constants

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('DataService')

# top of data_service.py
CHROMA_HOST = os.getenv("VECTOR_HOSTNAME", "localhost")
CHROMA_PORT = int(os.getenv("VECTOR_PORT", "8000"))
CHROMA_PATH = os.getenv("VECTOR_DATA_PATH", "/home/mmluser/chroma_db")
EMBED_MODEL = os.getenv("VECTOR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# inside DataService class

# ------------------------- SQL helpers -------------------------
import re
from typing import Iterable, Mapping, Sequence, Tuple, Any
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
def _quote_ident(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return f"`{name}`"
    
def insert(conn, auth_key, table, values):
    """Insert a row into `table` with `values`."""
    if not validate_auth_key(auth_key, 'insert', table):
        return None 
    if len(values) < 1:
        return None
    logger.debug("values are: %s, len(values) = %d, type: %s", values, len(values), type(values))

    # Safer: parameterized insert with quoted table name
    placeholders = ", ".join(["%s"] * len(values))
    query = f"INSERT INTO {_quote_ident(table)} VALUES ({placeholders})"
    logger.debug("built query: %s", query)
    logger.debug("need to insert values: %s", values)

    with conn.cursor() as cursor:
        try:
            cursor.execute(query, values)
            conn.commit()
        except Exception as e:
            logger.debug(f'insert query failed ({query}): {e}')
            return None

    return True


def delete(conn, auth_key, table, values):
    if not validate_auth_key(auth_key, 'delete', table):
        return None
    if not values:
        return None
    if not all(isinstance(t, (list, tuple)) and len(t) == 2 for t in values):
        logger.debug('delete() expects list of 2-tuples (field, value)')
        return None

    clauses, params = [], []
    for field, val in values:
        # (optional) quote identifiers more safely:
        # field = _quote_ident(field)
        clauses.append(f"{field}=%s")
        params.append(val)

    where_clause = " AND ".join(clauses)
    query = f'DELETE FROM `{table}` WHERE {where_clause}'

    with conn.cursor() as cursor:
        try:
            cursor.execute(query, params)
            conn.commit()
        except Exception as e:
            logger.debug(f'delete query failed: {e}')
            return None
    return True



def execute(conn, auth_key, query):
    """Execute an arbitrary SQL query (no result)."""
    if not validate_auth_key(auth_key, 'execute'):
        return None
    with conn.cursor() as cursor:
        try:
            cursor.execute(query)
            conn.commit()  # commit on the connection (not the cursor)
        except Exception as e:
            logger.debug(f'failed to execute query: {query}, err={e}')
            return None
    return True


def query(conn, auth_key, query, numrows=10):
    if not validate_auth_key(auth_key, 'query'):
        return None
    with conn.cursor() as cursor:
        try:
            cursor.execute(query)
            rows = cursor.fetchall()            # <-- was fetchmany(numrows)
        except Exception as e:
            logger.debug(f'select query ({query}) failed with exception: {e}')
            return None
    return rows


def save_df(conn, auth_key, table, df: str, if_exists='replace'):
    """
    Save a pandas DataFrame (json string, orient='index') into `table`.
    Respects `if_exists` param ('fail' | 'replace' | 'append').
    """
    logger.debug(f'recv\'d save_df request: auth_key: {auth_key}, table:{table}')
    if not validate_auth_key(auth_key, 'save_df'):
        return None

    try:
        df = pd.read_json(df, orient='index', dtype='object')
        engine_str = f'mysql+pymysql://{metaml.db_user}:{metaml.db_password}@{metaml.db_host}/{metaml.database}?charset=utf8mb4'
        engine = create_engine(engine_str)
        df.to_sql(table, con=engine, if_exists=if_exists, index=False)
        return True
    except Exception as e:
        logger.debug(f'save_df failed with exception: {e}')
        return None


from sqlalchemy import create_engine, text

def load_df(conn, auth_key, query):
    if not validate_auth_key(auth_key, 'load_df'):
        return None
    try:
        engine_str = f'mysql+pymysql://{metaml.db_user}:{metaml.db_password}@{metaml.db_host}/{metaml.database}'
        engine = create_engine(engine_str)
        with engine.connect() as eng_conn:
            df = pd.read_sql_query(text(query), eng_conn)
        return df.to_json(orient='index')
    except Exception as e:
        logger.debug(f'load_df query ({query}) failed with exception: {e}')
        return None


# ------------------------- Candidate list helpers -------------------------

def list_domains(conn, auth_key):
    """Return all domains: [{id, name, desc}, ...]"""
    if not validate_auth_key(auth_key, 'query', 'domains'):
        return None
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DOMAIN_ID, DOMAIN_NAME, COALESCE(DOMAIN_DESCRIPTION,'')
            FROM domains
            ORDER BY DOMAIN_NAME;
        """)
        rows = cur.fetchall()
    return [{"id": r[0], "name": r[1], "desc": r[2]} for r in rows]

def list_dimensions(conn, auth_key, domain_id):
    """Return dimensions for a domain: [{id, name, attr}, ...]"""
    if not validate_auth_key(auth_key, 'query', 'dimensions'):
        return None
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DIMENSION_ID, DIMENSION_NAME, COALESCE(DIMENSION_ATTRIBUTE,'')
            FROM dimensions
            WHERE DOMAIN_ID=%s
            ORDER BY DIMENSION_NAME;
        """, (int(domain_id),))
        rows = cur.fetchall()
    return [{"id": r[0], "name": r[1], "attr": r[2]} for r in rows]

def list_areas(conn, auth_key, dimension_id):
    """Return areas for a dimension: [{id, name}, ...]"""
    if not validate_auth_key(auth_key, 'query', 'areas'):
        return None
    with conn.cursor() as cur:
        cur.execute("""
            SELECT AREA_ID, AREA_NAME
            FROM areas
            WHERE DIMENSION_ID=%s
            ORDER BY AREA_NAME;
        """, (int(dimension_id),))
        rows = cur.fetchall()
    return [{"id": r[0], "name": r[1]} for r in rows]
    
import os, re, json, numpy as np
import pandas as pd

def _safe_name_token(s: str) -> str:
    m = re.sub(r'[^0-9a-zA-Z_-]+', '_', str(s or '').strip().lower())
    m = re.sub(r'_{2,}', '_', m).strip('_-')
    if len(m) < 3: m = f"col_{m or 'x'}"
    if not m[0].isalnum(): m = f"c_{m}"
    if not m[-1].isalnum(): m = f"{m}x"
    return m[:63]

def _fallback_collection_name(md: dict) -> str:
    d = _safe_name_token((md or {}).get("domain"))
    a = _safe_name_token((md or {}).get("apparea"))
    return f"{d}__{a}"

def _maybe_seed_from_local(coll, coll_name: str):
    """
    If a brand-new Chroma collection is empty, try to seed from local files:
      <LOCAL_EMB_DIR>/<coll_name>.npy          # embeddings (NxD)
      <LOCAL_EMB_DIR>/<coll_name>.csv          # texts (column 'text' or first column)
    """
    base = os.getenv("LOCAL_EMB_DIR", "/home/mmluser/MetaML/p2p/taxonomy_local_embeddings")
    npy = os.path.join(base, f"{coll_name}.npy")
    csv = os.path.join(base, f"{coll_name}.csv")
    if not (os.path.isfile(npy) and os.path.isfile(csv)):
        return

    embs = np.load(npy)
    df = pd.read_csv(csv)
    if "text" in df.columns:
        texts = df["text"].astype(str).tolist()
    else:
        texts = df.iloc[:, 0].astype(str).tolist()
    n = min(len(texts), len(embs))
    if n == 0:
        return
    ids = [f"{coll_name}:{i}" for i in range(n)]
    coll.add(ids=ids, embeddings=embs[:n].tolist(), documents=texts[:n], metadatas=[{"src":"local"}]*n)


# --------------------- Vector (ChromaDB) helpers ---------------------

def collection_query(vector_conn, auth_key, metadata, data, num_bots):
    if not validate_auth_key(auth_key, 'query'):
        return None

    # ---- normalize + inject collection name here ----
    md = dict(metadata or {})
    coll = md.get("collection")
    if not coll:
        # fallback from domain+area, e.g. "fintech__agent_banking_cash_in_cash_out"
        coll = _fallback_collection_name(md)
    coll = _safe_name_token(coll)   # strips parens, spaces, etc. → Chroma-safe
    md["collection"] = coll
    logger.info(f"[DataService] using collection: {coll}")

    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug(f"Function collection_query called: metadata: {md}")
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug(f"Function collection_query called: data: {data}")
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    rows = handle_collection_request(
        jumptable_constants.QUERY_COLLECTION,
        vector_conn,
        md,              # <-- pass the updated metadata
        data,
        num_bots
    )
    return rows if rows is not None else None



def collection_insert(vector_conn, auth_key, metadata, data):
    if not validate_auth_key(auth_key, 'insert'):
        return None

    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug(f"Function collection_insert called: metadata: {metadata}")
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug(f"Function collection_insert called: data: {data}")
    logger.debug(f"Function collection_insert called: data (DATA TYPE): {type(data)}")
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    try:
        return handle_collection_request(jumptable_constants.INSERT_INTO_COLLECTION, vector_conn, metadata, data)
    except Exception as e:
        logger.error(f"An error occurred - insert fingerprint into collection failed: {e}")
        return None


def collection_update(vector_conn, auth_key, metadata, data):
    if not validate_auth_key(auth_key, 'update'):
        return None

    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug(f"Function collection_update called: metadata: {metadata}")
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug(f"Function collection_update called: data: {data}")
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    handle_collection_request(jumptable_constants.UPDATE_COLLECTION, vector_conn, metadata, data)
    return True


def collection_delete(vector_conn, auth_key, metadata, data):
    if not validate_auth_key(auth_key, 'delete'):
        return None

    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug(f"Function collection_delete called: metadata: {metadata}")
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug(f"Function collection_delete called: data: {data}")
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    logger.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    handle_collection_request(jumptable_constants.DELETE_FROM_COLLECTION, vector_conn, metadata, data)
    return True


# ------------------------- Dispatcher & messaging -------------------------
async def dispatcher(conn, func, args):
    """
    conn:
      - for SQL ops: a pymysql connection
      - for Vector ops: the vector_conn dict
    func: 'db.insert', 'db.collection_query', ...
    """
    fn = func[3:]  # strip 'db.' prefix
    auth_key = args[0]

    # --- Candidate lists ---
    if fn == 'list_domains':
        return list_domains(conn, auth_key)
    elif fn == 'list_dimensions':
        # args: [auth_key, domain_id]
        return list_dimensions(conn, auth_key, args[1])
    elif fn == 'list_areas':
        # args: [auth_key, dimension_id]
        return list_areas(conn, auth_key, args[1])

    # --- SQL ops ---
    if fn == 'insert':
        return insert(conn, auth_key, args[1], args[2])
    elif fn == 'delete':
        return delete(conn, auth_key, args[1], args[2])
    elif fn == 'execute':
        return execute(conn, auth_key, args[1])
    elif fn == 'query':
        return query(conn, auth_key, args[1])
    elif fn == 'save_df':
        return save_df(conn, auth_key, args[1], args[2])
    elif fn == 'load_df':
        return load_df(conn, auth_key, args[1])

    # --- Vector ops ---
    elif fn == 'collection_query':
        return collection_query(conn, auth_key, args[1], args[2], args[3])
    elif fn == 'collection_insert':
        return collection_insert(conn, auth_key, args[1], args[2])
    elif fn == 'collection_update':
        return collection_update(conn, auth_key, args[1], args[2])
    elif fn == 'collection_delete':
        return collection_delete(conn, auth_key, args[1], args[2])

#async def dispatcher(conn, func, args):
#    """
#    conn:
#      - for SQL ops: a pymysql connection
#      - for Vector ops: the vector_conn dict
#    func: 'db.insert', 'db.collection_query', ...
#    """
#        elif fn == 'list_domains':
#        return list_domains(conn, auth_key)
#    elif fn == 'list_dimensions':
#        # args: [auth_key, domain_id]
#        return list_dimensions(conn, auth_key, args[1])
#    elif fn == 'list_areas':
#        # args: [auth_key, dimension_id]
#        return list_areas(conn, auth_key, args[1])
#
#    fn = func[3:]  # strip 'db.' prefix
#    auth_key = args[0]
#
#    if fn == 'insert':
#        return insert(conn, auth_key, args[1], args[2])
#    elif fn == 'delete':
#        return delete(conn, auth_key, args[1], args[2])
#    elif fn == 'execute':
#        return execute(conn, auth_key, args[1])
#    elif fn == 'query':
#        return query(conn, auth_key, args[1])
#    elif fn == 'save_df':
#        return save_df(conn, auth_key, args[1], args[2])
#    elif fn == 'load_df':
#        return load_df(conn, auth_key, args[1])
#    elif fn == 'collection_query':
#        return collection_query(conn, auth_key, args[1], args[2], args[3])
#    elif fn == 'collection_insert':
#        return collection_insert(conn, auth_key, args[1], args[2])
#    elif fn == 'collection_update':
#        return collection_update(conn, auth_key, args[1], args[2])
#    elif fn == 'collection_delete':
#        return collection_delete(conn, auth_key, args[1], args[2])


def validate_auth_key(auth_key, action, table=''):
    # TODO: real validation
    return True


def parseResponse(ret):
    logger.debug(f'parsing response using return value: {type(ret)}, {ret}')
    response = Message(MessageType.RESPONSE)
    try:
        if ret is None:
            # Return a RESPONSE with an empty JSON array string so callers can json.loads safely
            response = Message(
                message_type=MessageType.RESPONSE,
                message_data=MessageData(method=None, args=None, response="[]")
            )
        elif isinstance(ret, list):
            response = Message(
                message_type=MessageType.RESPONSE,
                message_data=MessageData(method=None, args=None, response=(json.dumps(ret)))
            )
        elif isinstance(ret, str):
            response = Message(
                message_type=MessageType.RESPONSE,
                message_data=MessageData(method=None, args=None, response=ret)
            )
        else:
            response = Message(
                message_type=MessageType.RESPONSE,
                message_data=MessageData(method=None, args=None, response=ret)
            )
    except Exception as e:
        logger.debug(f'exception while parsing db response:{e}')
    return response


def get_node_type_from_api(func):
    if func in metaml.business_service_apis:
        return NodeType.Business
    elif func in metaml.context_service_apis:
        return NodeType.Context
    elif func in metaml.taxonomy_service_apis:
        return NodeType.Taxonomy
    elif func in metaml.metrics_service_apis:
        return NodeType.Metrics
    elif func in metaml.data_service_apis:
        return NodeType.Data
    else:
        return None


async def forward_service(func, connection, msg):
    node_type = get_node_type_from_api(func)
    if node_type is None:
        await connection.send_message(Message(MessageType.ERROR))
        return
    # forward_to will send the downstream response back via `connection`
    await metaml.forward_to(node_type, node, connection, msg)
    
with pymysql.connect(host=metaml.db_host,
                     port=metaml.db_port,
                     user=metaml.db_user,
                     database=metaml.database,
                     password=metaml.db_password,
                     charset='utf8',
                     connect_timeout=3) as db_connection:
    # --- DEBUG: print effective DB settings (no password) and sanity checks ---
    logger.info(
        f"[DS BOOT] MySQL host={metaml.db_host}:{metaml.db_port} "
        f"user={metaml.db_user} db={metaml.database}"
    )
    try:
        with db_connection.cursor() as c:
            c.execute("SELECT DATABASE()")
            logger.info(f"[DS BOOT] SELECT DATABASE() -> {c.fetchone()[0]}")
            c.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=DATABASE()")
            logger.info(f"[DS BOOT] table count in schema -> {c.fetchone()[0]}")
            # domains table check
            try:
                c.execute("SELECT COUNT(*) FROM domains")
                logger.info(f"[DS BOOT] domains count -> {c.fetchone()[0]}")
            except Exception as te:
                logger.error(f"[DS BOOT] domains table check failed: {te}")
    except Exception as e:
        logger.error(f"[DS BOOT] sanity checks failed: {e}")
    # --- END DEBUG ---


async def handle_client(reader: 'asyncio.StreamReader', writer: 'asyncio.StreamWriter'):
    connection = node_connection.NodeConnection(reader, writer)
    msg = await connection.read_message()

    if msg.message_type == MessageType.QUERY:
        request_body = metaml.parse_message_data(msg.message_data)
        logger.debug(f"Req recv'd: {request_body}")
        func, args = request_body.method, request_body.args

        response = None
        try:
            # Fast fail if DB is unavailable
            with pymysql.connect(host=metaml.db_host,
                                 port=metaml.db_port,
                                 user=metaml.db_user,
                                 database=metaml.database,
                                 password=metaml.db_password,
                                 charset='utf8',
                                 connect_timeout=3) as db_connection:
                logger.debug(f'recv\'d call: {func} w/ args: {args}')
                if func in metaml.data_service_apis:
                    try:
                        ret = await dispatcher(db_connection, func, args)
                        response = parseResponse(ret)
                    except Exception as e:
                        logger.debug(f"error running func: {e}")
                        traceback.print_exc()
                        # Return a safe, parseable empty result on internal error
                        response = parseResponse(None)
                else:
                    await forward_service(func, connection, msg)
                    return  # downstream replied already
        except Exception as conn_err:
            # Crucial: reply even when DB connect fails so callers don't timeout
            logger.error(f"MySQL connect failed: {conn_err}")
            response = Message(
                message_type=MessageType.RESPONSE,
                message_data=MessageData(
                    method=None, args=None,
                    response=json.dumps({"error": f"MySQL connect failed: {str(conn_err)}"})
                )
            )

        if response:
            logger.debug('Sending response')
            await connection.send_message(response)

    elif msg.message_type == MessageType.VECTOR:
        request_body = metaml.parse_message_data(msg.message_data)
        logger.debug(f"Req recv'd: {request_body}")
        func, args = request_body.method, request_body.args

        vector_conn = {
            "hostname": metaml.vectordb_hostname,
            "port": metaml.vectordb_port,
            "data_path": metaml.vectordb_data_path,
            "embedding_model": metaml.vectordb_embedding_model
        }

        logger.debug("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        logger.debug(f"VECTOR MESSAGE RECEIVED: vector_conn: {vector_conn}")
        logger.debug("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        response = None
        logger.debug(f'recv\'d call: {func} w/ args: {args}')
        if func in metaml.data_service_apis:
            try:
                ret = await dispatcher(vector_conn, func, args)
                response = parseResponse(ret)
            except Exception as e:
                logger.debug(f"error running func: {e}")
                traceback.print_exc()
                response = parseResponse(None)
        else:
            await forward_service(func, connection, msg)
            return  # downstream replied already

        if response:
            logger.debug('Sending response')
            await connection.send_message(response)

    elif msg.message_type == MessageType.LIST:
        logger.info('List request received')
        nearest_nodes = await node.nearest_of_type()
        logger.debug(nearest_nodes)
        await connection.send_message(Message(
            MessageType.RESPONSE,
            MessageData(method=None, args=None, response=nearest_nodes)
        ))

    elif msg.message_type == MessageType.PING:
        await connection.send_message(Message(MessageType.HELO))


# ------------------------- Entrypoint -------------------------

port, bootstrap_node = metaml.fetch_args()
node = metaml.MetaMLNode(node_type=NodeType.Data,
                         client_handler=handle_client,
                         port=port,
                         bootstrap_node=bootstrap_node)
asyncio.run(node.init_dht())


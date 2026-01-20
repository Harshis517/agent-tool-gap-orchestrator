import heapq
import json
import pickle
import sys
import traceback
import paramiko
import pexpect
import os
import shutil
import zipfile
import io
from sklearn.ensemble import RandomForestRegressor
import pymysql
import asyncio
import logging
import numpy as np
from pyts.image import GramianAngularField
import re
import uuid
import pandas as pd

import metaml
import node_connection
from metaml import NodeType
from messaging.message import *
from typing import List  # add at top
import KnowledgeManager

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('BusinessService')
logging.getLogger('numba').setLevel(logging.WARNING)
MAX_FEATURE = 4
DEBUG = True


async def get_bots_from_index(connection, context: str) -> list:
    domains, dimensions, application_area, business_problem, customer_profile = [
        s.strip() for s in context.split(',', 4)
    ]
    sql = """
        SELECT bot_id, Github_link
        FROM Bots_database
        WHERE domains = %s AND dimensions = %s AND application_area = %s
          AND business_problem = %s AND customer_profile = %s
    """
    with connection.cursor() as cur:
        cur.execute(sql, (domains, dimensions, application_area, business_problem, customer_profile))
        return cur.fetchall()



# Get the fingerprint from the regime of request
def feature_extractor(size, maxtime, regime):
    regime = np.array(regime)
    transformer = GramianAngularField()
    m, _ = regime.shape
    # padding to (10, 22)
    feature = np.c_[np.full(m, size), np.full(m, maxtime), regime]
    # transform to image (10, 22, 22)
    feature = transformer.transform(feature)
    # resize to (4840, )
    feature = feature.flatten()
    return feature


# TODO replace with FFT stuff and remove
def get_fingerprint(regime):
    logger.info('regime=%s', regime)
    if len(regime) == 2:
        km = KnowledgeManager(regime[0], regime[1])
        km.load_dataframe(pd.read_csv('../data/ZBH0_MBO-buy.csv'))
        return km.features
    elif len(regime) == 3:
        return feature_extractor(regime[0], regime[1], regime[2])
    else:
        raise AttributeError('Invalid regime')


async def available_bots_fingerprints(connection: 'pymysql.Connection', context: str, regime: str):
    available_bots = await get_bots_from_index(connection=connection, context=context)
    fingerprints = get_fingerprint(regime=regime)
    return available_bots, fingerprints
    

async def clone_repo(url: str, password: str):
    command = "git clone " + url
    try:
        child = pexpect.spawn(command)
        index = child.expect(["password:", "Are you sure you want to continue connecting (yes/no)?", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
        if index == 1:
            child.sendline("yes")
            child.expect("password:")
            index = 0
        if index == 0:
            child.sendline(password)
            child.expect(pexpect.EOF)
        # do NOT chdir here; caller manages cwd per-repo
    except Exception as e:
        logger.error(f"An error occurred in clone_repo({url}): {e}")

async def clone_and_zip_repos(repo_urls: List[str]) -> str:
    # put the zip where Orchestrator expects it
    out_dir = metaml.bots_archive_temp_path
    os.makedirs(out_dir, exist_ok=True)
    output_zip_path = os.path.join(out_dir, f"bots-{uuid.uuid4().hex[:8]}.zip")

    # temp working dir for clones
    work_root = os.path.join(out_dir, f"tmp-clone-{uuid.uuid4().hex[:6]}")
    os.makedirs(work_root, exist_ok=True)

    password = metaml.bot_archive_password  # if your repos require it

    try:
        # clone each repo into work_root
        for repo_url in repo_urls:
            os.chdir(work_root)
            await clone_repo(repo_url, password)

        # zip the entire work_root contents
        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(work_root):
                for file in files:
                    full = os.path.join(root, file)
                    rel = os.path.relpath(full, work_root)
                    z.write(full, arcname=rel)
    finally:
        # best-effort cleanup
        try:
            shutil.rmtree(work_root, ignore_errors=True)
        except Exception:
            pass

    return output_zip_path

#async def clone_repo(url, password):
#    command = "git clone " + url
#
#    try:
#        child = pexpect.spawn(command)  # Start the command
#        index = child.expect(["password:", "Are you sure you want to continue connecting (yes/no)?"])
#        if index == 1:  # The SSH trust prompt
#            child.sendline("yes")
#            child.expect("password:")  # Expect the password prompt after responding "yes"
#
#        child.sendline(password)  # Send the password
#        child.expect(pexpect.EOF)  # Wait for the command to complete
#        folder_name=url.split('/')[-2]
#
#        if DEBUG:
#            logger.debug(f"folder name: {folder_name}")
#
#        os.chdir(os.path.join(os.getcwd(), folder_name))
#
#        if DEBUG:
#            logger.debug(f"Current dir: {os.getcwd()}")
#
#        command = "git checkout main"
#        child = pexpect.spawn(command)
#        if child.before:
#            print(child.before.decode())
#
#    except Exception as e:
#        logger.error(f"An error occurred in clone_repo: {e}")
#
#
#
#    hostname = metaml.bot_archive_hostname
#    username = metaml.bot_archive_username
#    password = metaml.bot_archive_password
#
#    if DEBUG:
#        logger.debug(f"bot archive hostname: {hostname}")
#        logger.debug(f"bot archive username: {username}")
#        logger.debug(f"bot archive password: {password}")
#
#    # Create a temporary directory to clone the repositories
#    temp_dir = os.path.join(os.getcwd(), 'temp')
#    os.makedirs(temp_dir, exist_ok=True)
#    os.chdir(temp_dir)
#    for repo_url in repo_urls: 
#        os.chdir(temp_dir)
#        await clone_repo(repo_url,password)
#
#    os.chdir(os.path.dirname(temp_dir))
#    output_zip_path = os.path.join(os.getcwd(), "output_zip.zip")  # Create the ZIP archive
#    zip_archive = zipfile.ZipFile(output_zip_path, "w")
#    for root, _, files in os.walk(temp_dir):
#        for file in files:
#            file_path = os.path.join(root, file)
#            zip_archive.write(file_path, arcname=os.path.relpath(file_path, temp_dir))
#
#    zip_archive.close()
#    os.chmod(temp_dir, 0o777)
#    for root, _, files in os.walk(temp_dir):
#        for file in files:
#            file_path = os.path.join(root, file)
#            os.chmod(file_path, 0o777)
#    shutil.rmtree(temp_dir)
#    return output_zip_path
#    
#async def clone_and_zip_repos(repo_urls):
#
#    # put the zip where Orchestrator expects it
#    out_dir = metaml.bots_archive_temp_path
#    os.makedirs(out_dir, exist_ok=True)
#    output_zip_path = os.path.join(out_dir, f"bots-{uuid.uuid4().hex[:8]}.zip")

async def get_bots_code(bot_ids):
    repo_urls = []
    for bot_id in bot_ids:
        # numeric vs string id safety
        try:
            bid = int(bot_id)
            q = f"SELECT URL FROM Bots WHERE bot_id = {bid}"
        except Exception:
            bid = str(bot_id).replace("'", "''")
            q = f"SELECT URL FROM Bots WHERE bot_id = '{bid}'"

        rows_json = await node.db_query(None, q)
        # node.db_query returns the DataService "response" (JSON string)
        try:
            rows = json.loads(rows_json)  # -> e.g. [[ "https://..." ]] or [[ "https://..." , ...]]
        except Exception:
            rows = []

        if rows and rows[0] and rows[0][0]:
            repo_urls.append(rows[0][0])

    if not repo_urls:
        return await _empty_zip_with_note("No repository URLs found for predicted bot IDs.")

    return await clone_and_zip_repos(repo_urls)

#async def get_bots_code(bot_ids):
#    repo_urls = []  # Initialize an empty list to store URLs
#    for bot_id in bot_ids:
#        query = f"SELECT URL FROM Bots WHERE bot_id = \"{bot_id}\""
#        bot_metadata = await node.db_query(None, query)
#        bot_metadata = json.loads(bot_metadata)['response']
#        res = bot_metadata.strip('][').split(', ')
#
#        if DEBUG:
#            logger.debug(f'recv\'d bot data: {res}')
#
#        url = res[0].strip('\"')  # Assuming the result is just a single URL now
#        repo_urls.append(url)  # Add the URL to the list
#    zip_file_path = await clone_and_zip_repos(repo_urls)
#    return zip_file_path

async def _empty_zip_with_note(note: str, out_dir: str = None) -> str:
    out_dir = out_dir or metaml.bots_archive_temp_path
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"no-bots-{uuid.uuid4().hex[:8]}.zip")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("README.txt", note)
    return path

async def predict_bots(connection: 'pymysql.connections.Connection', context: str, fft: str, fingerprint_id: str, domain: str, app_area: str, num_bots: int):
    # Receive fingerprint_id as 'None'
    if DEBUG:
        logger.debug(f"recv\'d call for predict_bots")

    df = pd.read_json(fft, orient='index')  # Get the df from the json-encoded string

    if DEBUG:
        logger.debug(f"read in dataframe of size: {len(df)}")

    try:
        km = KnowledgeManager.KnowledgeManager(node)
        bot_ids = await km.predict_fingerprint(df, fingerprint_id, domain, app_area, num_bots)
        logger.info(f"business_service.predict_bots: returned value: bot_ids: {bot_ids}")

        if bot_ids:
            return await get_bots_code(bot_ids)
        else:
            return await _empty_zip_with_note(
                "No bots matched the fingerprint/taxonomy selection."
            )
    except Exception as e:
        logger.error(f"An error occurred fetching bots: {e}")
        traceback.print_exc()
        return await _empty_zip_with_note("Error while fetching bots. See logs for details.")

#    try:
#        km = KnowledgeManager.KnowledgeManager(node)
#        bot_ids = await km.predict_fingerprint(df, fingerprint_id, domain, app_area, num_bots)
#
#        if DEBUG:
#            logger.info(f"business_service.predict_bots: returned value: bot_ids: {bot_ids}")
#
#        if bot_ids is not None:
#            return await get_bots_code(bot_ids)  # Fetch bot links from the database and return
#        else:
#            return None
#    except Exception as e:
#        logger.error(f"An error occurred fetching bots: {e}")
#        traceback.print_exc()
#        return None


async def handle_client(reader: 'asyncio.StreamReader', writer: 'asyncio.StreamWriter'):
    connection = node_connection.NodeConnection(reader, writer)
    msg = await connection.read_message()
    if msg.message_type == MessageType.QUERY:
        request_body = metaml.parse_message_data(msg.message_data)
        func, args = request_body.method, request_body.args

        if DEBUG:
            logger.debug("------------------------------------------------------------------------------------------")
            logger.debug("----------------------------- business_service.handle_client()----------------------------")
            logger.debug(f"func: {func}")
            logger.debug(f"args: {args}")
            logger.debug("------------------------------------------------------------------------------------------")

        with pymysql.connect(host=metaml.db_host,
                             port=metaml.db_port,
                             user=metaml.db_user,
                             database=metaml.database,
                             password=metaml.db_password) as db_connection:
            response = None
            if func in metaml.business_service_apis:
            
                if func == 'predict_bots':  # GET /bot_predict
                    try:
                        # (deprecated?) bots = await get_bots_from_index(connection=db_connection, context=args[0])

                        if DEBUG:
                            logger.debug(f"predict bots: {len(args)}")

                        context, fft, fingerprint_id, domain, app_area, num_bots = args

                        if DEBUG:
                            logger.debug("predict bots")

                        predicted_bots = await predict_bots(connection=db_connection, context=context,
                                                            fft=fft,
                                                            fingerprint_id=fingerprint_id,
                                                            domain=domain,
                                                            app_area=app_area,
                                                            num_bots=num_bots)

                        if DEBUG:
                            logger.info(f"\n\npredicted_bots: {predicted_bots}\n\n")

                        response = Message(MessageType.RESPONSE,
                                           MessageData(response=predicted_bots, method=None, args=None))
                    except Exception as e:
                        logger.error(f"An error occurred fetching bots: {e}")
                elif func == 'add_fingerprint':
                    try:
                        (context, fft_json, statistics_json,
                         fingerprint_id, domain, app_area, data_type, context_id, file_path) = args

                        df = pd.read_json(fft_json, orient='index')
                        km = KnowledgeManager.KnowledgeManager(node)
                        ok = await km.add_fingerprint(df, fingerprint_id, domain, app_area)

                        response = Message(
                            MessageType.RESPONSE,
                            MessageData(method=None, args=None, response=str(bool(ok)))
                        )
                    except Exception as e:
                        logger.exception("add_fingerprint failed")
                        response = Message(
                            MessageType.RESPONSE,
                            MessageData(method=None, args=None, response="false")
                        )

                elif func == 'available_bots_fingerprints':  # Used in PUT /fingerprint (@context service)
                    context, regime = args
                    _available_bots_fingerprints = await available_bots_fingerprints(connection=db_connection,
                                                                                     context=context,
                                                                                     regime=regime)
                    available_bots, fingerprints = _available_bots_fingerprints
                    fingerprints = metaml.encode(fingerprints)
                    response = Message(message_type=MessageType.RESPONSE,
                                       message_data=MessageData(method=None, args=None,
                                                                response=(available_bots, fingerprints)))
            elif func in metaml.context_service_apis:
                await metaml.forward_to(NodeType.Context, node, connection, msg)
            elif func in metaml.taxonomy_service_apis:
                await metaml.forward_to(NodeType.Taxonomy, node, connection, msg)
            elif func in metaml.metrics_service_apis:
                await metaml.forward_to(NodeType.Metrics, node, connection, msg)
            else:
                response = Message(MessageType.ERROR)
            if response:
                await connection.send_message(response)
            else:
                await connection.send_message(Message.empty_message())
    elif msg.message_type == MessageType.LIST:
        if DEBUG:
            logger.info('List request received')

        nearest_nodes = await node.nearest_of_type()

        if DEBUG:
            logger.debug(nearest_nodes)

        await connection.send_message(Message(MessageType.RESPONSE,
                                              MessageData(method=None, args=None, response=nearest_nodes)))
    elif msg.message_type == MessageType.PING:
        await connection.send_message(Message(MessageType.HELO, bytes(b'')))

port, bootstrap_node = metaml.fetch_args()
node = metaml.MetaMLNode(node_type=NodeType.Business,
                         client_handler=handle_client,
                         port=port,
                         bootstrap_node=bootstrap_node)
asyncio.run(node.init_dht())

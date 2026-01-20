# orchestrator_service.py
import sys
import os
import json
import asyncio
import logging
from typing import Optional, Tuple

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import requests
from flask_cors import CORS
import metaml
import node_connection
from messaging.message import Message, MessageData, MessageType
from metaml import NodeType

# ------------------------- Logging -------------------------
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('OrchestratorService')
DEBUG = True

# ------------------------- Licensing config -------------------------
LIC_URL  = os.getenv("LICENSING_URL",  "http://127.0.0.1:1607")
LIC_USER = os.getenv("LICENSING_USER", "service_orchestrator")
LIC_PASS = os.getenv("LICENSING_PASS", "change_me")
LIC_TIMEOUT  = (2, 5)       # (connect, read) seconds
FAIL_OPEN = os.getenv("LIC_FAIL_OPEN", "false").lower() in ("1","true","yes")       # fail-closed by default

def check_license(bot_id: Optional[str], license_key: Optional[str]) -> Tuple[bool, str]:
    if not bot_id or not license_key:
        return False, "missing bot_id or license_key"
    try:
        # Prefer /validate if service creds are present
        if os.getenv("LICENSING_USER") and os.getenv("LICENSING_PASS"):
            r = requests.post(
                f"{LIC_URL}/license/validate",
                json={"bot_id": bot_id, "license_key": license_key},
                headers={
                    "Content-Type": "application/json",
                    "X-User": os.getenv("LICENSING_USER"),
                    "X-Password": os.getenv("LICENSING_PASS"),
                    "X-Request-Id": request.headers.get("X-Request-Id", "auto"),
                },
                timeout=LIC_TIMEOUT,
            )
        else:
            # Fallback: open verify (no auth), expects {"license_token": "..."}
            r = requests.post(
                f"{LIC_URL}/license/verify",
                json={"license_token": license_key},
                headers={"Content-Type": "application/json"},
                timeout=LIC_TIMEOUT,
            )

        if r.status_code != 200:
            try:
                body = r.json()
            except Exception:
                body = {"error": f"http_{r.status_code}"}
            return False, (body.get("detail") or body.get("error") or f"http_{r.status_code}")

        data = r.json()
        return bool(data.get("valid")), data.get("detail", "invalid")
    except Exception:
        return (True, "licensing_unreachable") if FAIL_OPEN else (False, "licensing_unreachable")


# ------------------------- Fortress (optional) -------------------------
def fortress_check_access(user_id: str, service: str, action: str) -> bool:
    """
    Soft guard. If METAML_FORTRESS_DISABLE is set, always allow.
    Else try Fortress REST (optional). Falls back to allowlist.
    """
    if os.getenv("METAML_FORTRESS_DISABLE", "").lower() in ("1", "true", "yes"):
        return True

    allow = set((os.getenv("FORTRESS_ALLOW_USERS") or "test").split(","))
    if user_id in allow:
        return True

    base = os.getenv("FORTRESS_BASE_URL", "").rstrip("/")
    realm_user = os.getenv("FORTRESS_REALM_USER", "")
    realm_pass = os.getenv("FORTRESS_REALM_PASS", "")
    if not base:
        return False

    try:
        r = requests.post(
            f"{base}/authorize",
            json={"userId": user_id, "resource": service, "action": action},
            auth=(realm_user, realm_pass),
            timeout=2,
        )
        return r.status_code == 200 and r.json().get("allowed") is True
    except Exception:
        return False

# ------------------------- Flask app & helpers -------------------------
def flaskFrontend(hostip, portnum):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256 MB

    # CORS: allow the portal origin to call this Flask app (predict_bots)
    CORS(
        app,
        resources={r"/*": {"origins": "http://208.109.36.23:5173"}},
        supports_credentials=True,
        expose_headers=["Content-Type", "Authorization"],
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "OPTIONS"]
    )

    # Optional: fast preflight for the root path
    @app.route("/", methods=["OPTIONS"])
    def preflight_root():
        return ("", 204)
    app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256 MB

    # Bind Flask on hostip (0.0.0.0 ok), but connect to a real IP for P2P (port-1)
    connect_host = metaml.find_my_ip() or "127.0.0.1"
    app.config["ORCH_PEER"] = f"{connect_host}:{int(portnum) - 1}"

    ALLOWED_EXTENSIONS = {"zip"}

    def _allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def convertRequest(req, method='n/a'):
        """
        Convert HTTP request into (fcontents, jsonargs, file_path)
        - application/json: build fcontents from fingerprint.fft (indexed)
        - multipart/form-data: expect 'file' (zip) and 'data' (json)
        """
        if req.mimetype == 'application/json':
            content = req.get_json() or {}
            fcontents = ""
            fp = content.get('fingerprint') or {}
            if 'fft' in fp:
                load_data = fp.get('fft', [])
                indexed = {str(i): {"hashes": e[0], "time": e[1]} for i, e in enumerate(load_data)}
                fcontents = json.dumps(indexed, indent=2)
            fingerprint = fp.copy()
            fingerprint.pop('fft', None)        # keep statistics if present
            metadata = content.get('metadata', {}) or {}
            return fcontents, {"fingerprint": fingerprint, "metadata": metadata}, None

        elif req.mimetype == 'multipart/form-data':
            z = req.files.get('file')
            if not z or z.filename == '':
                return "Expected file input, found none.", None, None
            if not _allowed_file(z.filename):
                return "Invalid file type, only ZIP files allowed.", None, None

            os.makedirs(metaml.bots_archive_temp_path, exist_ok=True)
            safe = secure_filename(z.filename)
            file_path = os.path.join(metaml.bots_archive_temp_path, safe)
            z.save(file_path)

            json_data = req.form.get('data')
            if not json_data:
                return 'No JSON data provided', None, None

            content = json.loads(json_data)
            fp = content.get('fingerprint') or {}
            if 'fft' in fp:
                load_data = fp.get('fft', [])
                indexed = {str(i): {"hashes": e[0], "time": e[1]} for i, e in enumerate(load_data)}
                fcontents = json.dumps(indexed, indent=2)
            else:
                fcontents = "No FFT data"

            fingerprint = fp.copy()
            fingerprint.pop('fft', None)        # keep statistics if present
            metadata = content.get('metadata', {}) or {}
            return fcontents, {"fingerprint": fingerprint, "metadata": metadata}, file_path

        return "500 internal error (failed to interpret request).", None, None

    async def sendRequestToNode(p2prequest: Message):
        # Send to orchestrator P2P peer (port-1)
        connection = await node_connection.NodeConnection.create_connection(app.config["ORCH_PEER"])
        await connection.send_message(p2prequest)
        return await connection.read_message()

    def prepareResponseForClient(messageresponse: Optional[Message]):
        if messageresponse is None or messageresponse.message_type != MessageType.RESPONSE:
            return "500 internal error (failed to get response from p2p network)", 500
        responsebody = metaml.parse_message_data(messageresponse.message_data)
        return str(responsebody.response), 200

    @app.post("/add_fingerprint")
    async def add_fingerprint_request():
        logger.info("Received add_fingerprint request")
        fcontents, jsonargs, file_path = convertRequest(request, "add_fingerprint")
        if not isinstance(jsonargs, dict):
            return (fcontents, 400) if isinstance(fcontents, str) else ("Bad request", 400)

        # Build args expected by Business service
        stats_json = json.dumps((jsonargs.get('fingerprint') or {}).get('statistics', []), indent=2)
        bt = jsonargs['metadata']['business_taxonomy']
        context = bt['context']
        fingerprint_id = jsonargs['fingerprint'].get('fingerprint_id')
        domain = bt['domain']
        app_area = bt['app_area']
        data_type = bt['data_type']
        context_id = context['id']

        msgargs = (context, fcontents, stats_json, fingerprint_id, domain, app_area, data_type, context_id, file_path)
        msg = Message(message_type=MessageType.QUERY,
                      message_data=MessageData(method="add_fingerprint", args=tuple(msgargs)))

        p2presponse = await sendRequestToNode(msg)
        _, status = prepareResponseForClient(p2presponse)
        return jsonify({"ok": True, "detail": "added bot"}), status

    @app.post("/delete_fingerprint")
    async def delete_fingerprint_request():
        logger.info("Received delete_fingerprint request")
        fcontents, jsonargs, _ = convertRequest(request, "delete_fingerprint")
        if not isinstance(jsonargs, dict):
            return (fcontents, 400) if isinstance(fcontents, str) else ("Bad request", 400)

        context = jsonargs['metadata']['business_taxonomy']['context']
        fingerprint_id = jsonargs['fingerprint'].get('fingerprint_id')

        msg = Message(message_type=MessageType.QUERY,
                      message_data=MessageData(method="delete_fingerprint",
                                               args=(context, fcontents, fingerprint_id)))
        p2presponse = await sendRequestToNode(msg)
        _, status = prepareResponseForClient(p2presponse)
        return jsonify({"ok": True, "detail": "deleted bot fingerprint"}), status

    def _extract_bot_and_key(payload):
        fp = (payload or {}).get("fingerprint") or {}
        md = (payload or {}).get("metadata") or {}
        bot_id = payload.get("bot_id") or fp.get("fingerprint_id")
        # allow top-level, metadata, header, or query param
        lic_key = (
            payload.get("license_key")
            or md.get("license_key")
            or request.headers.get("X-License-Key")
            or request.args.get("license_key")
        )
        return bot_id, lic_key
    
    @app.post("/predict_bots")
    async def predict_bots():
        logger.info("Received predict_bots request")

        # 1) License check (fail-closed by default)
        payload = request.get_json(silent=True) or {}
        bot_id, lic_key = _extract_bot_and_key(payload)
        valid, reason = check_license(bot_id, lic_key)
        if not valid:
            return jsonify({"error": "license invalid", "bot_id": bot_id, "reason": reason}), 403

        # Optional Fortress soft-guard
        if os.getenv("USE_FORTRESS", "0").lower() in ("1", "true", "yes"):
            user_id = (payload.get("metadata", {})
                              .get("business_taxonomy", {})
                              .get("context", {}) or {}).get("userId") \
                      or os.getenv("FORTRESS_TEST_USER", "test")
            if not fortress_check_access(user_id, "LicensingService", "invoke-predict-bots"):
                return ("License/permission denied (Fortress)", 403)

        # 2) Convert request body
        fcontents, jsonargs, _ = convertRequest(request, "predict_bots")
        if not isinstance(jsonargs, dict):
            return (fcontents, 400) if isinstance(fcontents, str) else ("Bad request", 400)

        # 3) Taxonomy parse (domain/area/num_bots, etc.)
        try:
            parsed = await call_taxonomy_parse(jsonargs)
        except Exception as e:
            app.logger.error(f"Taxonomy parse failed: {e}")
            return ("Unable to parse query via taxonomy", 502)

        domain = parsed.get('domain')
        area = parsed.get('area')
        num_bots = parsed.get('num_bots') or (jsonargs.get('fingerprint') or {}).get('num_bots') or 1
        missing = [k for k, v in [('domain', domain), ('area', area), ('num_bots', num_bots)] if not v]
        if missing:
            return (f"Missing taxonomy fields: {', '.join(missing)}", 400)

        bt = (jsonargs.get('metadata') or {}).get('business_taxonomy', {})
        context = bt.get('context', {}) or {}
        fingerprint_id = (jsonargs.get('fingerprint') or {}).get('fingerprint_id')

        # 4) Send to Business via P2P
        msgargs = (context, fcontents, fingerprint_id, domain, area, int(num_bots))
        msg = Message(message_type=MessageType.QUERY,
                      message_data=MessageData(method="predict_bots", args=tuple(msgargs)))
        p2presponse = await sendRequestToNode(msg)
        converted, status = prepareResponseForClient(p2presponse)

        # 5) Return JSON or ZIP
        path = os.path.abspath(converted)
        base = os.path.abspath(metaml.bots_archive_temp_path)
        if not path.startswith(base + os.sep) or not os.path.isfile(path):
            app.logger.error(f"File not found or outside base dir: {path}")
            return "File not found", 404

        accept = (request.args.get("accept", "") or request.headers.get("Accept", "")).lower()
        if "application/json" in accept or accept == "json":
            return jsonify({
                "domain":    parsed.get("domain"),
                "dimension": parsed.get("dimension"),
                "area":      parsed.get("area"),
                "per_level": parsed.get("per_level", []),
                "reasoning": parsed.get("reasoning", "resolved all levels"),
                "download_path": path,
                "num_bots": int(num_bots),
            }), 200

        # If you want to return the ZIP (normal behavior):
        # return send_file(path, mimetype='application/zip', as_attachment=True,
        #                  download_name='download.zip', max_age=0)

        # If you want plain text for demos (NOT for production):
        with open(path, "rb") as f:
            data = f.read()
        return data, 200, {"Content-Type": "text/plain"}

    async def call_taxonomy_parse(payload: dict):
        """Ask the Taxonomy node to parse the incoming business payload."""
        tax_msg = Message(
            message_type=MessageType.QUERY,
            message_data=MessageData(method='parse_business_query', args=(payload,))
        )
        service_nodes = await node.nearest_of_type(NodeType.Taxonomy)
        if not service_nodes:
            raise RuntimeError("No Taxonomy nodes available")
        host, port = service_nodes[0].split(":"); port = int(port)

        reader, writer = await asyncio.open_connection(host=host, port=port)
        conn = node_connection.NodeConnection(reader, writer)
        await conn.send_message(tax_msg)
        resp = await conn.read_message()
        try:
            await conn.close()
        except Exception:
            pass

        if resp is None or resp.message_type != MessageType.RESPONSE:
            raise RuntimeError("Taxonomy service did not return a RESPONSE")
        return metaml.parse_message_data(resp.message_data).response

    logger.info(f"Flask starting on {hostip}:{portnum}")
    logger.info(f"Connecting to orchestrator P2P at {app.config['ORCH_PEER']}")
    app.run(host=hostip, port=portnum, debug=False, use_reloader=False)

# ------------------------- P2P plumbing (unchanged) -------------------------
async def _proxy_to(node_type, msg_type, func, args):
    endpoints = await node.nearest_of_type(node_type)
    if not endpoints:
        return Message(MessageType.ERROR)
    host, port = endpoints[0].split(":"); port = int(port)

    reader, writer = await asyncio.open_connection(host=host, port=port)
    conn = node_connection.NodeConnection(reader, writer)

    outbound = Message(message_type=msg_type,
                       message_data=MessageData(method=func, args=args))
    await conn.send_message(outbound)
    resp = await conn.read_message()
    try:
        await conn.close()
    except Exception:
        pass
    return resp

async def handle_client(reader, writer):
    connection = node_connection.NodeConnection(reader, writer)
    try:
        msg = await connection.read_message()
    except Exception as e:
        logger.error(f"Failed to read message: {e}")
        return

    # ---------- QUERY ----------
    if msg.message_type == MessageType.QUERY:
        try:
            request_body = metaml.parse_message_data(msg.message_data)
            func, args = request_body.method, request_body.args
        except Exception as e:
            logger.error(f"Bad QUERY message format: {e}")
            await connection.send_message(Message(MessageType.ERROR))
            return

        try:
            if func in metaml.business_service_apis:
                resp = await _proxy_to(NodeType.Business, MessageType.QUERY, func, args)
            elif func in metaml.taxonomy_service_apis:
                resp = await _proxy_to(NodeType.Taxonomy, MessageType.QUERY, func, args)
            elif func in metaml.context_service_apis:
                resp = await _proxy_to(NodeType.Context, MessageType.QUERY, func, args)
            elif func in metaml.metrics_service_apis:
                resp = await _proxy_to(NodeType.Metrics, MessageType.QUERY, func, args)
            elif func in metaml.data_service_apis:
                resp = await _proxy_to(NodeType.Data, MessageType.QUERY, func, args)
            else:
                logger.warning(f"Unknown QUERY func: {func}")
                resp = Message(MessageType.ERROR)
        except Exception as e:
            logger.exception(f"Error proxying QUERY {func}: {e}")
            resp = Message(MessageType.ERROR)

        await connection.send_message(resp); return

    # ---------- VECTOR ----------
    elif msg.message_type == MessageType.VECTOR:
        try:
            request_body = metaml.parse_message_data(msg.message_data)
            func, args = request_body.method, request_body.args
        except Exception as e:
            logger.error(f"Bad VECTOR message format: {e}")
            await connection.send_message(Message(MessageType.ERROR))
            return

        try:
            if func in metaml.data_service_apis:
                resp = await _proxy_to(NodeType.Data, MessageType.VECTOR, func, args)
            else:
                logger.warning(f"Unknown VECTOR func: {func}")
                resp = Message(MessageType.ERROR)
        except Exception as e:
            logger.exception(f"Error proxying VECTOR {func}: {e}")
            resp = Message(MessageType.ERROR)

        await connection.send_message(resp); return

    # ---------- LIST ----------
    elif msg.message_type == MessageType.LIST:
        logger.info('List request received')
        nearest_nodes = await node.nearest_of_type()
        await connection.send_message(Message(
            MessageType.RESPONSE,
            MessageData(method=None, args=None, response=nearest_nodes)
        ))
        return

    # ---------- PING ----------
    elif msg.message_type == MessageType.PING:
        await connection.send_message(Message(MessageType.HELO))
        return

    # ---------- DEFAULT ----------
    else:
        logger.warning(f"Unsupported message type: {msg.message_type}")
        await connection.send_message(Message(MessageType.ERROR))
        return

# ------------------------- Main -------------------------
if __name__ == "__main__":
    host_for_flask = "0.0.0.0"
    port, bootstrap_node = metaml.fetch_args()  # orchestrator P2P port (e.g., 1604)

    # Create the P2P orchestrator node
    node = metaml.MetaMLNode(
        node_type=NodeType.Orchestrator,
        client_handler=handle_client,
        port=port,
        bootstrap_node=bootstrap_node
    )

    # Start Flask on (port+1) in a separate thread
    import threading
    frontend = threading.Thread(
        name="Orchestrator Frontend",
        target=flaskFrontend,
        args=(host_for_flask, port + 1),
        daemon=True
    )
    frontend.start()
    logger.info("Started Flask frontend thread, preparing DHT...")

    # Start the DHT / P2P server
    asyncio.run(node.init_dht())

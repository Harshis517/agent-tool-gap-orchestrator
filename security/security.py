# services/security/security.py
import os
import time
import json
import logging
import requests
from functools import wraps
from typing import Tuple, Dict, Any

import jwt
from jwt import PyJWKClient
from flask import request, jsonify
import inspect
from functools import wraps
from flask import request, jsonify
logger = logging.getLogger("Security")

# -------------------- Config --------------------
KEYCLOAK_ISSUER   = os.getenv("KEYCLOAK_BASE_URL", "").rstrip("/")  # e.g. https://keycloak/auth/realms/myrealm
KEYCLOAK_AUD      = os.getenv("KEYCLOAK_AUDIENCE", "metaml-orchestrator")
REQUIRED_ROLE     = os.getenv("REQUIRED_ROLE", "")  # optional: e.g. "metaml-user"

LICENSE_VERIFIER_URL = os.getenv("LICENSE_VERIFIER_URL", "http://localhost:9090/license/verify")
LICENSE_CACHE_TTL    = int(os.getenv("LICENSE_CACHE_TTL_SEC", "60"))

# Dev mode toggle (if no issuer set, we skip JWT verification)
DEV_AUTH_DISABLED = not bool(KEYCLOAK_ISSUER)

AUTH_BYPASS    = os.getenv("AUTH_BYPASS", "0") == "1"
LICENSE_BYPASS = os.getenv("LICENSE_BYPASS", "0") == "1"
# -------------------- JWKS via PyJWT --------------------
_jwk_client = None
def _get_jwk_client() -> PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        jwks_url = f"{KEYCLOAK_ISSUER}/protocol/openid-connect/certs"
        _jwk_client = PyJWKClient(jwks_url)
    return _jwk_client

def _extract_roles_from_claims(claims: Dict[str, Any]) -> set:
    roles = set()
    # Keycloak realm roles
    roles.update((claims.get("realm_access") or {}).get("roles") or [])
    # Keycloak client roles (resource_access)
    res = claims.get("resource_access") or {}
    for client, obj in res.items():
        roles.update(obj.get("roles") or [])
    return roles

def verify_jwt(auth_header: str) -> Tuple[bool, str, Dict[str, Any] | None]:
    if DEV_AUTH_DISABLED:
        return True, "dev-auth-disabled", {"sub": "dev", "roles": ["dev"]}

    if not auth_header or not auth_header.lower().startswith("bearer "):
        return False, "missing bearer token", None

    token = auth_header.split(" ", 1)[1].strip()
    try:
        jwk_client = _get_jwk_client()
        signing_key = jwk_client.get_signing_key_from_jwt(token)

        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
            audience=KEYCLOAK_AUD,
            issuer=KEYCLOAK_ISSUER,
            options={"require": ["exp", "iat"], "verify_aud": True, "verify_iss": True},
        )

        if REQUIRED_ROLE:
            roles = _extract_roles_from_claims(claims)
            if REQUIRED_ROLE not in roles:
                return False, f"role '{REQUIRED_ROLE}' required", None

        return True, "ok", claims
    except Exception as e:
        logger.error(f"JWT verify error: {e}")
        return False, f"jwt error: {e}", None
        
#def require_auth(fn):
#    @wraps(fn)
#    def wrapper(*args, **kwargs):
#        if AUTH_BYPASS:
#            # accept request without JWT during local testing
#            return fn(*args, **kwargs)
#        ok, msg, _ = verify_jwt(request.headers.get("Authorization", ""))
#        if not ok:
#            return jsonify({"error": "unauthorized", "detail": msg}), 401
#        return fn(*args, **kwargs)
#    return wrapper
#
#def require_license(feature_name="predict_bots"):
#    def _dec(fn):
#        @wraps(fn)
#        def wrapper(*args, **kwargs):
#            if LICENSE_BYPASS:
#                return fn(*args, **kwargs)
#            ok, feats = _check_license(feature_name)
#            if not ok:
#                return jsonify({"error": "license_invalid"}), 403
#            if feature_name and not feats.get(feature_name, True):
#                return jsonify({"error": "feature_not_licensed","feature": feature_name}), 403
#            return fn(*args, **kwargs)
#        return wrapper
#    return _dec

#def require_auth(fn):
#    @wraps(fn)
#    def wrapper(*args, **kwargs):
#        ok, msg, _ = verify_jwt(request.headers.get("Authorization", ""))
#        if not ok:
#            return jsonify({"error": "unauthorized", "detail": msg}), 401
#        return fn(*args, **kwargs)
#    return wrapper

# -------------------- License verifier --------------------
_LICENSE_CACHE = {"ts": 0.0, "ok": False, "features": {}}

def _check_license(feature: str) -> Tuple[bool, Dict[str, Any]]:
    now = time.time()
    if now - _LICENSE_CACHE["ts"] < LICENSE_CACHE_TTL:
        return _LICENSE_CACHE["ok"], _LICENSE_CACHE["features"]

    try:
        resp = requests.post(LICENSE_VERIFIER_URL, json={"feature": feature}, timeout=3)
        resp.raise_for_status()
        data = resp.json()
        ok = bool(data.get("valid", False))
        feats = data.get("features", {})
        _LICENSE_CACHE.update({"ts": now, "ok": ok, "features": feats})
        return ok, feats
    except Exception as e:
        logger.error(f"License check failed: {e}")
        return False, {}

#def require_license(feature_name: str = "predict_bots"):
#    def _dec(fn):
#        @wraps(fn)
#        def wrapper(*args, **kwargs):
#            ok, feats = _check_license(feature_name)
#            if not ok:
#                return jsonify({"error": "license_invalid"}), 403
#            if feature_name and not feats.get(feature_name, True):
#                return jsonify({"error": "feature_not_licensed", "feature": feature_name}), 403
#            return fn(*args, **kwargs)
#        return wrapper
#    return _dec

# services/security/security.py


# ... existing imports and config ...

def require_auth(fn):
    if inspect.iscoroutinefunction(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            if AUTH_BYPASS:
                return await fn(*args, **kwargs)
            ok, msg, _ = verify_jwt(request.headers.get("Authorization", ""))
            if not ok:
                return jsonify({"error": "unauthorized", "detail": msg}), 401
            return await fn(*args, **kwargs)
        return wrapper
    else:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if AUTH_BYPASS:
                return fn(*args, **kwargs)
            ok, msg, _ = verify_jwt(request.headers.get("Authorization", ""))
            if not ok:
                return jsonify({"error": "unauthorized", "detail": msg}), 401
            return fn(*args, **kwargs)
        return wrapper

def require_license(feature_name="predict_bots"):
    def _dec(fn):
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def wrapper(*args, **kwargs):
                if LICENSE_BYPASS:
                    return await fn(*args, **kwargs)
                ok, feats = _check_license(feature_name)
                if not ok:
                    return jsonify({"error": "license_invalid"}), 403
                if feature_name and not feats.get(feature_name, True):
                    return jsonify({"error": "feature_not_licensed","feature": feature_name}), 403
                return await fn(*args, **kwargs)
            return wrapper
        else:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if LICENSE_BYPASS:
                    return fn(*args, **kwargs)
                ok, feats = _check_license(feature_name)
                if not ok:
                    return jsonify({"error": "license_invalid"}), 403
                if feature_name and not feats.get(feature_name, True):
                    return jsonify({"error": "feature_not_licensed","feature": feature_name}), 403
                return fn(*args, **kwargs)
            return wrapper
    return _dec

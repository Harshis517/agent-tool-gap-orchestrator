/*
 * Author: Harshita Singh (hs5412)
 * Project: MetaML Licensing Service
 * Description: File added as part of the Licensing microservice implementation.
 * Notes: Contains logic for authentication, authorization, and Fortress-based 
 *        license issuance and validation.
 * Date: Summer 2025
 */

package com.metaml.licensing.controller;

import javax.annotation.PostConstruct;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.server.ResponseStatusException;

import com.metaml.licensing.service.License3jService;

// Fortress Core (2.0.8)
import org.apache.directory.fortress.core.AccessMgr;
import org.apache.directory.fortress.core.AccessMgrFactory;
import org.apache.directory.fortress.core.SecurityException;
import org.apache.directory.fortress.core.model.Permission;
import org.apache.directory.fortress.core.model.Session;
import org.apache.directory.fortress.core.model.User;

import javax.naming.Context;
import javax.naming.NamingEnumeration;
import javax.naming.directory.*;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@RestController
@RequestMapping("/license")
public class LicenseController {

  private static final Logger log = LoggerFactory.getLogger(LicenseController.class);

  private final License3jService lic;

  public LicenseController(License3jService lic) {
    this.lic = lic;
  }

  // ── feature flags & data ─────────────────────────────────────────────────────
  @Value("${licensing.allow.all:true}") private boolean allowAll;
  @Value("${licensing.valid.keys:}")    private String validKeysCsv;
  private Set<String> validKeys = new HashSet<String>();

  // ── Fortress config (shared) ─────────────────────────────────────────────────
  @Value("${fortress.mode:off}")               private String fortressMode; // off|ldap|rest|jar
  @Value("${fortress.object:LicensingService}") private String fortressObj;
  @Value("${fortress.operation:validate}")      private String fortressOp;

  // fortress-rest (when fortress.mode=rest)
  @Value("${fortress.base:}")         private String fortressBase;     // e.g. http://127.0.0.1:8080/fortress-rest
  @Value("${fortress.tenant:RBAC}")   private String fortressTenant;   // RBAC or DEFAULT
  @Value("${fortress.basic.user:}")   private String fortressBasicUser; // Tomcat BASIC user
  @Value("${fortress.basic.pass:}")   private String fortressBasicPass;

  // ── LDAP (used by ldap mode and /_debug/ldap-bind) ───────────────────────────
  @Value("${ldap.url:ldap://127.0.0.1:10389}")   private String ldapUrl;
  @Value("${ldap.base:dc=example,dc=com}")       private String ldapBase;
  @Value("${ldap.admin.dn:uid=admin,ou=system}") private String ldapAdminDn;
  @Value("${ldap.admin.pw:secret}")              private String ldapAdminPw;

  private final Map<String, Map<String,Object>> store = new ConcurrentHashMap<String, Map<String,Object>>();
  private final RestTemplate http = new RestTemplate();

  @PostConstruct
  void init() {
    if (validKeysCsv != null && validKeysCsv.trim().length() > 0) {
      String[] parts = validKeysCsv.split(",");
      for (String p : parts) {
        String v = p.trim();
        if (v.length() > 0) validKeys.add(v);
      }
    }
    log.info("Licensing: allowAll={}, validKeys={}, fortress.mode={}, tenant={}",
        Boolean.valueOf(allowAll), validKeys, fortressMode, fortressTenant);
  }

  // ── health ───────────────────────────────────────────────────────────────────
  @GetMapping("/healthz")
  public Map<String,Object> healthz() {
    Map<String,Object> m = new LinkedHashMap<String,Object>();
    m.put("ok", Boolean.TRUE);
    return m;
  }

  // ── validate (accepts either license_key or license_token) ───────────────────
 // ── validate (accepts either license_key or license_token) ───────────────────
@PostMapping("/validate")
public ResponseEntity<Map<String,Object>> validate(
    @RequestHeader(value = "X-User", required = false) String userHeader,
    @RequestHeader(value = "X-Password", required = false) String passHeader,
    @RequestBody(required = false) Map<String,Object> body) {

  // read headers/body safely
  String email = (userHeader != null) ? userHeader
      : (body != null ? String.valueOf(body.getOrDefault("user", "")) : "");

  String pwd   = (passHeader != null) ? passHeader
      : (body != null ? String.valueOf(body.getOrDefault("password", "")) : "");

  // if caller sent creds, enforce RBAC once
  if (email != null && !email.trim().isEmpty()) {
    fortressGate(email, pwd, fortressObj, fortressOp);
  }

  String key   = body == null ? null : String.valueOf(body.get("license_key"));
  String token = body == null ? null : String.valueOf(body.get("license_token"));

  boolean okSigned = false;
  try {
    if (token != null && !token.isEmpty()) okSigned = lic.verify(token);
    else if (key != null && key.split("\\.").length == 3) okSigned = lic.verify(key);
  } catch (Exception ignore) {}

  boolean okStatic = allowAll || (key != null && validKeys.contains(key)) || (key != null && isStoreKeyValid(key));
  boolean ok = okSigned || okStatic;

  Map<String,Object> resp = new LinkedHashMap<>();
  resp.put("valid", Boolean.valueOf(ok));
  resp.put("detail", okSigned ? "ok_signed" : (okStatic ? "ok" : "invalid_key"));
  return ok ? ResponseEntity.ok(resp) : ResponseEntity.status(HttpStatus.FORBIDDEN).body(resp);
}


  // ── issue a signed license token (RS256) ─────────────────────────────────────
  @PostMapping("/issue")
  public Map<String,Object> issue(
      @RequestHeader("X-User") String email,
      @RequestHeader("X-Password") String pwd,
      @RequestParam(value = "validDays", defaultValue = "30") int validDays,
      @RequestBody(required = false) Map<String,Object> attrs) {

    fortressGate(email, pwd, fortressObj, "create");
    String token = lic.issue(attrs == null ? new LinkedHashMap<String,Object>() : attrs, validDays, email);
    Map<String,Object> resp = new LinkedHashMap<String,Object>();
    resp.put("license_token", token);
    return resp;
  }

  // ── verify a token (signature + exp) ─────────────────────────────────────────
  @PostMapping("/verify")
  public Map<String,Object> verify(@RequestBody Map<String,Object> body) {
    String token = String.valueOf(body.get("license_token"));
    boolean ok = token != null && token.length() > 0 && lic.verify(token);
    Map<String,Object> resp = new LinkedHashMap<String,Object>();
    resp.put("valid", Boolean.valueOf(ok));
    resp.put("detail", ok ? "ok" : "invalid_or_expired");
    return resp;
  }

  // ── simple key CRUD kept for dev/manual keys ─────────────────────────────────
  @PostMapping({ "", "/", "/create" })
  public ResponseEntity<Map<String,Object>> create(
      @RequestHeader("X-User") String email,
      @RequestHeader("X-Password") String pwd,
      @RequestParam(value = "expiresAt", required = false) Long expiresEpoch,
      @RequestBody(required = false) Map<String,Object> attrs) {

    fortressGate(email, pwd, fortressObj, "create");

    String key = UUID.randomUUID().toString().replace("-", "");
    Map<String,Object> rec = new LinkedHashMap<String,Object>();
    rec.put("key", key);
    rec.put("owner", email);
    rec.put("issuedAt", Long.valueOf(Instant.now().getEpochSecond()));
    if (expiresEpoch != null) rec.put("expiresAt", expiresEpoch);
    rec.put("attributes", attrs == null ? new LinkedHashMap<String,Object>() : attrs);
    store.put(key, rec);
    return ResponseEntity.ok(rec);
  }

  @PutMapping("/{key}")
  public ResponseEntity<Map<String,Object>> update(
      @RequestHeader("X-User") String email,
      @RequestHeader("X-Password") String pwd,
      @PathVariable String key,
      @RequestParam(value = "expiresAt", required = false) Long expiresEpoch,
      @RequestBody(required = false) Map<String,Object> attrs) {

    fortressGate(email, pwd, fortressObj, "update");
    Map<String,Object> rec = store.get(key);
    if (rec == null) throw new ResponseStatusException(HttpStatus.NOT_FOUND, "license_not_found");
    if (expiresEpoch != null) rec.put("expiresAt", expiresEpoch);
    if (attrs != null) rec.put("attributes", attrs);
    return ResponseEntity.ok(rec);
  }

  @GetMapping("/{key}")
  public ResponseEntity<Map<String,Object>> get(@PathVariable String key) {
    Map<String,Object> rec = store.get(key);
    return rec == null ? ResponseEntity.notFound().build() : ResponseEntity.ok(rec);
  }

  // ── dev-only: verify LDAP creds quickly ──────────────────────────────────────
  @PostMapping("/_debug/ldap-bind")
  public Map<String,Object> debugLdapBind(@RequestBody Map<String,String> body) {
    String email = String.valueOf(body.get("email"));
    String pass  = String.valueOf(body.get("password"));
    Map<String,Object> out = new LinkedHashMap<String,Object>();
    try {
      String dn = ldapFindDn(email);
      if (dn == null) { out.put("ok", Boolean.FALSE); out.put("detail","user_not_found"); return out; }
      ldapBind(dn, pass);
      out.put("ok", Boolean.TRUE);
      out.put("dn", dn);
      out.put("detail", "bind_ok");
      return out;
    } catch (javax.naming.AuthenticationException e) {
      out.put("ok", Boolean.FALSE); out.put("detail","invalid_credentials"); return out;
    } catch (Exception e) {
      out.put("ok", Boolean.FALSE); out.put("detail","error: " + e.getMessage()); return out;
    }
  }

  // ── fortress mode switch ─────────────────────────────────────────────────────
  private void fortressGate(String email, String password, String obj, String op) {
    String mode = fortressMode == null ? "off" : fortressMode.toLowerCase(Locale.ROOT);
    if ("off".equals(mode)) return;
    if ("ldap".equals(mode)) {
      if (!ldapAuth(email, password)) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "ldap_auth_failed");
      return;
    }
    if ("rest".equals(mode)) { fortressRestAuthz(email, password, obj, op); return; }
    if ("jar".equals(mode))  { fortressJarAuthz(email, password, obj, op); return; }
    throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "invalid_fortress_mode");
  }

  private void fortressRestAuthz(String email, String password, String obj, String op) {
    if (isBlank(fortressBase) || isBlank(fortressBasicUser))
      throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "fortress_rest_not_configured");

    String token = Base64.getEncoder()
        .encodeToString((fortressBasicUser + ":" + fortressBasicPass).getBytes(java.nio.charset.StandardCharsets.UTF_8));
    HttpHeaders h = new HttpHeaders();
    h.setContentType(MediaType.APPLICATION_JSON);
    h.set("Authorization", "Basic " + token);

    // 1) createSession
    String sUrl = fortressBase + "/accessmgr/createSession?tenant=" + fortressTenant;
    Map<String,Object> sBody = new LinkedHashMap<String,Object>();
    sBody.put("userId", email);
    sBody.put("password", password);
    sBody.put("isTrusted", Boolean.FALSE);
    try {
      http.postForEntity(sUrl, new HttpEntity<Map<String,Object>>(sBody, h), Map.class);
    } catch (RestClientResponseException e) {
      int code = e.getRawStatusCode();
      if (code == 401) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "fortress_basic_or_user_auth_failed");
      throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "fortress_createSession_error: "+code+" "+e.getStatusText());
    }

    // 2) checkAccess
    String aUrl = fortressBase + "/accessmgr/checkAccess?tenant=" + fortressTenant;
    Map<String,Object> aBody = new LinkedHashMap<String,Object>();
    aBody.put("userId", email);
    aBody.put("objName", obj);
    aBody.put("opName", op);
    aBody.put("isTrusted", Boolean.FALSE);
    try {
      ResponseEntity<Map> r = http.postForEntity(aUrl, new HttpEntity<Map<String,Object>>(aBody, h), Map.class);
      Map b = r.getBody();
      Object v = b == null ? null :
          (b.containsKey("authorized") ? b.get("authorized") :
          (b.containsKey("isAuthorized") ? b.get("isAuthorized") :
          (b.containsKey("result") ? b.get("result") :
          (b.containsKey("permGranted") ? b.get("permGranted") : null))));
      boolean allowed =
          (v instanceof Boolean && ((Boolean) v).booleanValue()) ||
          (v instanceof Number  && ((Number)  v).intValue() != 0) ||
          (v instanceof String  && ("true".equalsIgnoreCase((String) v) ||
                                    "1".equals(v) ||
                                    "allowed".equalsIgnoreCase((String) v) ||
                                    "granted".equalsIgnoreCase((String) v)));
      if (!allowed) throw new ResponseStatusException(HttpStatus.FORBIDDEN, "fortress_rbac_denied");
    } catch (RestClientResponseException e) {
      if (e.getRawStatusCode() == 401) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "fortress_basic_or_user_auth_failed");
      throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "fortress_checkAccess_error: "+e.getRawStatusCode()+" "+e.getStatusText());
    }
  }

  private void fortressJarAuthz(String email, String password, String obj, String op) {
    try {
      AccessMgr am = AccessMgrFactory.createInstance();
      User u = new User(email, password);
      Session s = am.createSession(u, false);
      Permission p = new Permission(obj, op);
      boolean allowed = am.checkAccess(s, p);
      if (!allowed) throw new ResponseStatusException(HttpStatus.FORBIDDEN, "fortress_rbac_denied");
    } catch (SecurityException e) {
      String msg = e.getMessage() == null ? "" : e.getMessage().toLowerCase(Locale.ROOT);
      if (msg.contains("password") || msg.contains("invalid") || msg.contains("user")) {
        throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "fortress_auth_failed");
      }
      throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "fortress_core_error: " + e.getMessage());
    }
  }

  // ── local helpers ────────────────────────────────────────────────────────────
  private boolean isStoreKeyValid(String key) {
    Map<String,Object> rec = store.get(key);
    if (rec == null) return false;
    Object exp = rec.get("expiresAt");
    if (exp instanceof Number) {
      long now = Instant.now().getEpochSecond();
      if (now >= ((Number) exp).longValue()) return false;
    }
    return true;
  }

  private boolean ldapAuth(String email, String password) {
    try {
      String dn = ldapFindDn(email);
      if (dn == null) return false;
      ldapBind(dn, password);
      return true;
    } catch (Exception e) {
      log.warn("ldapAuth error: {}", e.toString());
      return false;
    }
  }

  private String ldapFindDn(String email) throws Exception {
    DirContext ctx = ldapAdminCtx();
    try {
      SearchControls sc = new SearchControls();
      sc.setSearchScope(SearchControls.SUBTREE_SCOPE);
      NamingEnumeration<SearchResult> rs = ctx.search(ldapBase, "(uid=" + email + ")", sc);
      String userDn = null;
      while (rs.hasMore()) userDn = rs.next().getNameInNamespace();
      return userDn;
    } finally { try { ctx.close(); } catch (Exception ignore) {} }
  }

  private void ldapBind(String dn, String password) throws Exception {
    java.util.Hashtable<String,String> env = new java.util.Hashtable<String,String>();
    env.put(Context.INITIAL_CONTEXT_FACTORY,"com.sun.jndi.ldap.LdapCtxFactory");
    env.put(Context.PROVIDER_URL, ldapUrl);
    env.put(Context.SECURITY_AUTHENTICATION, "simple");
    env.put(Context.SECURITY_PRINCIPAL, dn);
    env.put(Context.SECURITY_CREDENTIALS, password);
    new InitialDirContext(env).close();
  }

  private DirContext ldapAdminCtx() throws Exception {
    java.util.Hashtable<String,String> env = new java.util.Hashtable<String,String>();
    env.put(Context.INITIAL_CONTEXT_FACTORY,"com.sun.jndi.ldap.LdapCtxFactory");
    env.put(Context.PROVIDER_URL, ldapUrl);
    env.put(Context.SECURITY_AUTHENTICATION, "simple");
    env.put(Context.SECURITY_PRINCIPAL, ldapAdminDn);
    env.put(Context.SECURITY_CREDENTIALS, ldapAdminPw);
    return new InitialDirContext(env);
  }

  private static boolean isBlank(String s) {
    return s == null || s.trim().isEmpty();
  }
}

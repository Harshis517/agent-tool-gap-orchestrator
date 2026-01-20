/*
 * Author: Harshita Singh (hs5412)
 * Project: MetaML Licensing Service
 * Description: File added as part of the Licensing microservice implementation.
 * Notes: License3jService
 * Date: Summer 2025
 */


package com.metaml.licensing.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.KeyFactory;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.Signature;
import java.security.spec.PKCS8EncodedKeySpec;
import java.security.spec.X509EncodedKeySpec;
import java.time.Instant;
import java.util.Base64;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

@Service
public class License3jService {

  private final ObjectMapper om = new ObjectMapper();

  private final String product;
  private final PrivateKey privateKey; // may be null if not configured
  private final PublicKey publicKey;   // may be null if not configured

  public License3jService(
      @Value("${license.product:predict-bot}") String product,
      @Value("${license.private.key:}") String privPath,
      @Value("${license.public.key:}") String pubPath
  ) {
    this.product = product;
    try {
      this.privateKey = (privPath == null || privPath.isEmpty()) ? null : loadPrivateKey(privPath);
      this.publicKey  = (pubPath  == null || pubPath.isEmpty())  ? null : loadPublicKey(pubPath);
    } catch (Exception e) {
      throw new IllegalStateException("Failed to load license keys: " + e.getMessage(), e);
    }
  }

  /** Issue a compact, signed token (header.payload.signature base64url), RS256. */
  public String issue(Map<String,Object> attrs, int validDays, String owner) {
    if (privateKey == null) {
      throw new IllegalStateException("license.private.key not configured (PKCS#8 PEM expected)");
    }
    try {
      Map<String,Object> payload = new LinkedHashMap<String,Object>();
      payload.put("product", product);
      payload.put("owner", owner);
      payload.put("iat", Instant.now().getEpochSecond());
      if (validDays > 0) {
        payload.put("exp", Instant.now().plusSeconds(validDays * 86400L).getEpochSecond());
      }
      if (attrs != null && !attrs.isEmpty()) {
        payload.put("attrs", attrs);
      }

      String headerJson  = "{\"alg\":\"RS256\",\"typ\":\"MLIC\"}";
      String payloadJson = om.writeValueAsString(payload);

      String headerB64  = b64Url(headerJson.getBytes("UTF-8"));
      String payloadB64 = b64Url(payloadJson.getBytes("UTF-8"));
      String signingInput = headerB64 + "." + payloadB64;
      String sigB64 = b64Url(sign(signingInput.getBytes("UTF-8")));

      return signingInput + "." + sigB64;
    } catch (Exception e) {
      throw new RuntimeException("license issue failed: " + e.getMessage(), e);
    }
  }

  /** Verify signature + (optional) expiry. */
  public boolean verify(String token) {
    if (publicKey == null) {
      throw new IllegalStateException("license.public.key not configured (X.509 PEM expected)");
    }
    try {
      String[] parts = token.split("\\.");
      if (parts.length != 3) return false;

      String signingInput = parts[0] + "." + parts[1];
      byte[] sig = b64UrlDecode(parts[2]);

      if (!verify(signingInput.getBytes("UTF-8"), sig)) return false;

      Map payload = om.readValue(new String(b64UrlDecode(parts[1]), "UTF-8"), Map.class);
      Object exp = payload.get("exp");
      if (exp instanceof Number) {
        long now = Instant.now().getEpochSecond();
        if (now >= ((Number) exp).longValue()) return false;
      }
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  /** Extract payload claims (unsigned). */
  public Map<String,Object> parsePayload(String token) {
    try {
      String[] parts = token.split("\\.");
      if (parts.length < 2) return Collections.<String,Object>emptyMap();
      return om.readValue(new String(b64UrlDecode(parts[1]), "UTF-8"), Map.class);
    } catch (Exception e) {
      return Collections.<String,Object>emptyMap();
    }
  }

  // ── crypto helpers ──────────────────────────────────────────────────────────
  private byte[] sign(byte[] data) throws Exception {
    Signature s = Signature.getInstance("SHA256withRSA");
    s.initSign(privateKey);
    s.update(data);
    return s.sign();
  }

  private boolean verify(byte[] data, byte[] sig) throws Exception {
    Signature s = Signature.getInstance("SHA256withRSA");
    s.initVerify(publicKey);
    s.update(data);
    return s.verify(sig);
  }

  private static String b64Url(byte[] b) {
    return Base64.getUrlEncoder().withoutPadding().encodeToString(b);
  }

  private static byte[] b64UrlDecode(String s) {
    return Base64.getUrlDecoder().decode(s);
  }

  private static PrivateKey loadPrivateKey(String path) throws Exception {
    // Expect PKCS#8: -----BEGIN PRIVATE KEY-----
    byte[] all = Files.readAllBytes(Paths.get(path));
    String pem = new String(all, "UTF-8")
        .replace("-----BEGIN PRIVATE KEY-----", "")
        .replace("-----END PRIVATE KEY-----", "")
        .replaceAll("\\s+", "");
    byte[] der = Base64.getDecoder().decode(pem);
    return KeyFactory.getInstance("RSA").generatePrivate(new PKCS8EncodedKeySpec(der));
  }

  private static PublicKey loadPublicKey(String path) throws Exception {
    // Expect X.509 SubjectPublicKeyInfo: -----BEGIN PUBLIC KEY-----
    byte[] all = Files.readAllBytes(Paths.get(path));
    String pem = new String(all, "UTF-8")
        .replace("-----BEGIN PUBLIC KEY-----", "")
        .replace("-----END PUBLIC KEY-----", "")
        .replaceAll("\\s+", "");
    byte[] der = Base64.getDecoder().decode(pem);
    return KeyFactory.getInstance("RSA").generatePublic(new X509EncodedKeySpec(der));
  }
}

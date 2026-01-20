/*
 * Author: Harshita Singh (hs5412)
 * Project: MetaML Licensing Service
 * Description: App A with predict bot call
 * Date: Summer 2025
 */

// src/App.jsx
import React, { useEffect, useMemo, useState } from 'react';
import kc, { ensureKeycloak } from './keycloak';
import { fetchWithAuth, storeLicense } from './api';

const LIC_PATH     = import.meta.env.VITE_LIC_PATH     || '/api/lic';
const PREDICT_PATH = import.meta.env.VITE_PREDICT_PATH || '/api/predict';
const APPB_PATH    = import.meta.env.VITE_APPB_PATH    || '/api/appb';

const pretty = t => { try { return JSON.stringify(JSON.parse(t), null, 2); } catch { return t; } };

export default function App() {
  const [ready, setReady]   = useState(false);
  const [token, setToken]   = useState('');
  const [profile, setProf]  = useState(null);
  const [license, setLic]   = useState('');
  const [licStatus, setLS]  = useState('unknown');
  const [predict, setPred]  = useState('');
  const [echo, setEcho]     = useState('');

  useEffect(() => {
    let id;
    (async () => {
      try { await ensureKeycloak('check-sso'); } catch {}
  
      // If code/session_state present and we still don't have a token, retry init once.
      const sp = new URLSearchParams(location.search);
      const cameFromLogin = sp.has('code') || sp.has('session_state');
      if (!kc.token && cameFromLogin) {
        try { await ensureKeycloak('check-sso'); } catch {}
        history.replaceState(null, '', '/');
      }
  
      if (kc.token) {
        setToken(kc.token);
        try { setProf(await kc.loadUserProfile()); } catch {}
        id = setInterval(async () => {
          try { if (await kc.updateToken(120)) setToken(kc.token || ''); } catch {}
        }, 30000);
      }
      setReady(true);
    })();
    return () => { if (id) clearInterval(id); };
  }, []);
  
  
  
  
  // Drive UI from the actual token presence
  const signedIn = useMemo(() => Boolean(kc?.token || token), [token]);
  const licenseOk = licStatus.startsWith('VALID');

  async function signIn() {
    try { await ensureKeycloak('check-sso'); } catch {}
    const redirectUri = new URL('/', window.location.origin).toString(); // => "http://127.0.0.1:5173/"
    kc.login({ redirectUri, prompt: 'login' });
  }
  
  
  

  

  async function signOut() {
    setLic(''); setLS('unknown'); setPred(''); setEcho(''); storeLicense('');
    try {
      await ensureKeycloak('check-sso');
      const url = kc.createLogoutUrl({ redirectUri: `${window.location.origin}/` });
      window.location.assign(url);
    } catch {}
  }

  async function validateLicense() {
    if (!signedIn || !license.trim()) return;
    setLS('checking...');
    const licPass = import.meta.env.VITE_LIC_PASS || '';
    const email = kc?.tokenParsed?.email || kc?.tokenParsed?.preferred_username || profile?.email || '';
    try {
      const res = await fetch(`${LIC_PATH}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-User': email, 'X-Password': licPass },
        body: JSON.stringify({ license_token: license.trim() })
      });
      const j = await res.json().catch(() => ({}));
      setLS(`${j.valid ? 'VALID' : 'INVALID'} (${j.detail || 'no-detail'})`);
    } catch (e) { setLS(`ERROR: ${e.message}`); }
  }

  async function callPredict() {
    setPred('...');
    try {
      const res = await fetchWithAuth(`${PREDICT_PATH}?accept=json`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bot_id: 'bot-1',
          license_key: license,
          metadata: { business_taxonomy: { query: 'finance addressing fraudulent transactions', context: { id: 1 } } },
          fingerprint: { fingerprint_id: 'bot-1', fingerprint_type: 'time_series', num_bots: 1, fft: [[1024,1],[2048,2]] }
        })
      }, kc?.token || token);
      setPred(pretty(await res.text()));
    } catch (e) { setPred(`ERROR: ${e.message}`); }
  }

  async function callAppB() {
    setEcho('...');
    try {
      const res = await fetchWithAuth(`${APPB_PATH}/echo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ping: 'hello from portal', at: new Date().toISOString() })
      }, kc?.token || token);
      setEcho(pretty(await res.text()));
    } catch (e) { setEcho(`ERROR: ${e.message}`); }
  }

  return (
    <div style={{ fontFamily: 'ui-sans-serif, system-ui', margin: 24, maxWidth: 880 }}>
      <h1>Access Portal</h1>

      <div style={{display:'flex',gap:12,margin:'8px 0 16px'}}>
        <span style={{padding:'4px 8px',borderRadius:999,fontSize:12,
          background: signedIn ? '#E6FFEA' : '#FFECEC',
          border:`1px solid ${signedIn ? '#16a34a' : '#dc2626'}`, color: signedIn ? '#166534' : '#7f1d1d'}}>
          SSO {signedIn ? '✅' : '❌'}
        </span>
        <span style={{padding:'4px 8px',borderRadius:999,fontSize:12,
          background: licenseOk ? '#E6FFEA' : '#FEF9C3',
          border:`1px solid ${licenseOk ? '#16a34a' : '#f59e0b'}`, color: licenseOk ? '#166534' : '#92400e'}}>
          License {licenseOk ? 'VALID' : 'unknown'}
        </span>
      </div>

      <section style={{ marginBottom:24, padding:16, border:'1px solid #ddd', borderRadius:8 }}>
        <h2>1) Sign in (Keycloak SSO)</h2>
        {!ready && !signedIn ? (
          <div>Preparing SSO…</div>
        ) : signedIn ? (
          <>
            <button onClick={signOut}>Sign Out</button>
            <div style={{ marginTop: 8 }}>
              <div><b>User:</b> {kc?.tokenParsed?.email || kc?.tokenParsed?.preferred_username || profile?.email}</div>
              <div><b>Token:</b> {(kc?.token || token || '').slice(0,24)}…</div>
            </div>
          </>
        ) : (
          <button onClick={signIn}>Sign In</button>
        )}
      </section>

      <section style={{ marginBottom:24, padding:16, border:'1px solid #ddd', borderRadius:8, opacity: signedIn ? 1 : .5 }}>
        <h2>2) License</h2>
        {!signedIn && <div style={{margin:'4px 0 8px', color:'#92400e'}}><b>Sign in first to validate a license.</b></div>}
        <textarea rows={5} style={{ width:'100%' }} placeholder="eyJhbGciOiJSUzI1NiIsInR5cCI6Ik1MSUMifQ...."
          value={license} onChange={(e)=>setLic(e.target.value)} />
        <div style={{ marginTop: 8 }}>
          <input type="file" accept=".jwt,.txt" disabled={!signedIn}
            onChange={(e)=>{ const f=e.target.files?.[0]; if(!f) return; const r=new FileReader(); r.onload=()=>setLic(String(r.result).trim()); r.readAsText(f); }} />{' '}
          <button disabled={!signedIn || !license} onClick={()=>{ storeLicense(license); alert('Saved.'); }}>Save to Browser</button>{' '}
          <button disabled={!signedIn || !license} onClick={validateLicense}>Validate</button>{' '}
          <span><b>Status:</b> {licStatus}</span>
        </div>
      </section>

      <section style={{ marginBottom:24, padding:16, border:'1px solid #ddd', borderRadius:8 }}>
        <h2>3) App A — Predict Bot</h2>
        <button disabled={!signedIn || !licenseOk} onClick={callPredict}>
          Call Predict (requires SSO + license)
        </button>
        <pre style={{ whiteSpace:'pre-wrap', background:'#f8f8f8', padding:12, marginTop:8 }}>{predict}</pre>
      </section>

      <section style={{ marginBottom:24, padding:16, border:'1px solid #ddd', borderRadius:8 }}>
        <h2>4) App B — Echo (requires SSO)</h2>
        <button disabled={!signedIn} onClick={callAppB}>Call App B /echo</button>
        <pre style={{ whiteSpace:'pre-wrap', background:'#f8f8f8', padding:12, marginTop:8 }}>{echo}</pre>
      </section>
    </div>
  );
}

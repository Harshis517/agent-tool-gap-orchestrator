// Uses global Keycloak (from CDN)
const kc = new Keycloak({
  url: 'http://208.109.36.23:8088',
  realm: 'archnav',
  clientId: 'appb-spa',
});

function setStatus(msg){ const el=document.getElementById('out'); if(el) el.textContent=String(msg??''); }
function setBtns(signedIn){
  document.getElementById('btn-signin').disabled  = signedIn;
  document.getElementById('btn-signout').disabled = !signedIn;
  document.getElementById('btn-echo').disabled    = !signedIn;
}

async function init(){
  try {
    const authenticated = await kc.init({
      // on HTTP, avoid the 3p-cookie / storage access flow
      onLoad: 'check-sso',
      pkceMethod: 'S256',
      checkLoginIframe: false,
      silentCheckSsoFallback: false,     // <— key change
      // silentCheckSsoRedirectUri not needed when fallback=false
      flow: 'standard',
    });
    console.log('KC init -> authenticated?', authenticated);
  } catch(e){
    console.error('KC init failed', e);
    setStatus('Init failed: ' + (e?.message || String(e)));
  }
  setBtns(!!kc.token);
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btn-signin').onclick = () => {
    kc.login({ redirectUri: `${location.origin}/` });   // normal redirect
  };
  document.getElementById('btn-signout').onclick = () => {
    const url = kc.createLogoutUrl({ redirectUri: `${location.origin}/` });
    location.assign(url); // logs out realm-wide (affects 5173 too)
  };
  document.getElementById('btn-echo').onclick = async () => {
    try{
      const r = await fetch('http://208.109.36.23:1611/echo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${kc.token}` },
        body: JSON.stringify({ ping: 'hello from appb' }),
      });
      setStatus(await r.text());
    } catch(e){ setStatus('Echo error: ' + e.message); }
  };
  init();
});

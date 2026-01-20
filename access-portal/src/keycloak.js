import Keycloak from 'keycloak-js';

export const kc = new Keycloak({
  url: 'http://208.109.36.23:8088',   // your Keycloak base
  realm: 'archnav',
  clientId: 'predict-bot',
});

if (!window.kc) window.kc = kc;

let _initPromise = null;

export function ensureKeycloak(onLoad = 'check-sso') {   // 👈 changed from login-required
  if (_initPromise) return _initPromise;

  const cfg = {
    onLoad,                              // do not force login each time
    pkceMethod: 'S256',
    checkLoginIframe: true,              // keep alive check
    silentCheckSsoFallback: true,
    silentCheckSsoRedirectUri: window.location.origin + '/silent-check-sso.html',
    flow: 'standard',
  };

  console.log('[Keycloak] init config:', cfg);

  _initPromise = kc.init(cfg)
    .then(authenticated => {
      console.log('[Keycloak] Authenticated:', authenticated);
      if (!authenticated) {
        console.warn('[Keycloak] Not authenticated — redirecting to login');
        kc.login({ redirectUri: window.location.href });   // 👈 remove prompt:'login'
      }
      return authenticated;
    })
    .catch(err => {
      console.error('[Keycloak] Init error:', err);
      _initPromise = null;
    });

  return _initPromise;
}

export default kc;

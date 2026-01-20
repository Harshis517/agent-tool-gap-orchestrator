import Keycloak from 'https://cdn.jsdelivr.net/npm/keycloak-js@25.0.4/+esm';

const kc = new Keycloak({
  url: 'http://208.109.36.23:8088',
  realm: 'archnav',
  clientId: 'appb-spa',
});

export async function ensureKeycloak(onLoad = 'login-required') {
  const ok = await kc.init({
    onLoad,
    pkceMethod: 'S256',
    checkLoginIframe: false,
    silentCheckSsoFallback: false,
    flow: 'standard',
  });
  if (!ok) {
    kc.login({ redirectUri: 'http://208.109.36.23:5174/' });
  }
  return kc;
}

export default kc;

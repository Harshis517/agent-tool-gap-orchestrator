export async function fetchWithAuth(url, opts = {}, token) {
  const headers = new Headers(opts.headers || {})
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(url, { ...opts, headers })
}

export function getStoredLicense() {
  return localStorage.getItem('mlic.token') || ''
}

export function storeLicense(token) {
  if (token) localStorage.setItem('mlic.token', token)
}

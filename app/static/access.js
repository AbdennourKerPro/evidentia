/** Invitation access: read the code from the link's fragment, open a session, show the quota. */

/** The code travels in the URL fragment (#invite=...): browsers never send it to servers or logs. */
export function inviteFromHash(hash) {
  const match = /(?:^#|&)invite=([A-Za-z0-9_-]{8,128})(?:&|$)/.exec(hash || "");
  return match ? match[1] : null;
}

export function describeQuota(invite) {
  if (!invite) return "";
  if (invite.remaining <= 0) return `Quota épuisé (${invite.quota}/${invite.quota} questions)`;
  return `Questions restantes : ${invite.remaining} / ${invite.quota}`;
}

/** True when the interface may let the user ask a question. */
export function canAsk(access) {
  if (access.mode !== "invite") return true;
  return Boolean(access.authenticated && access.invite && access.invite.remaining > 0);
}

/** Redeem a code if the link carries one, then ask the server who we are. */
export async function loadAccess(fetchImpl, hash, clearHash) {
  const code = inviteFromHash(hash);
  let error = null;
  if (code) {
    try {
      const response = await fetchImpl("/access/redeem", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code }),
      });
      if (!response.ok) error = (await response.json().catch(() => ({}))).detail || "Invitation invalide.";
    } catch { error = "Le serveur est injoignable."; }
    clearHash();
  }
  try {
    const me = await (await fetchImpl("/access/me")).json();
    return { ...me, error };
  } catch {
    return { mode: "open", authenticated: true, error: null };
  }
}

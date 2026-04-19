const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

export async function askQuestion(query) {
  const response = await fetch(`${API_BASE_URL}/api/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query })
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Unable to fetch an answer right now.");
  }

  return response.json();
}

export async function getSuggestions() {
  const response = await fetch(`${API_BASE_URL}/api/suggestions`);

  if (!response.ok) {
    throw new Error("Unable to fetch suggestions right now.");
  }

  return response.json();
}

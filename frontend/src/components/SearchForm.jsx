import { useState } from "react";

export default function SearchForm({ onSubmit, loading }) {
  const [query, setQuery] = useState("");

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }
    onSubmit(query.trim());
  };

  return (
    <form className="search-form" onSubmit={handleSubmit}>
      <input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        className="search-input"
        placeholder="Expense ratio, lock-in, benchmark, statements..."
        aria-label="Ask a mutual fund question"
      />
      <button type="submit" className="search-button" disabled={loading}>
        {loading ? "Checking..." : "Search"}
      </button>
    </form>
  );
}

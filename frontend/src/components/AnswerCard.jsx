export default function AnswerCard({ result }) {
  if (!result) {
    return null;
  }

  return (
    <article className="answer-card">
      <div className="answer-header">
        <span className={`status-pill status-${result.status}`}>{result.status}</span>
      </div>
      <p className="answer-text">{result.answer}</p>
      <a className="source-link" href={result.source_url} target="_blank" rel="noreferrer">
        View official source
      </a>
      <p className="last-updated">Last updated from sources: {result.last_updated}</p>
    </article>
  );
}

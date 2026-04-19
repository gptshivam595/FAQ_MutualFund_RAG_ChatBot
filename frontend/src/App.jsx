import { useEffect, useRef, useState } from "react";

import { askQuestion } from "./services/api";

const marqueeItems = [
  "AMC Scheme Pages",
  "KIM / SID Documents",
  "Factsheet Pages",
  "AMC Statement Help",
  "AMFI Investor Pages",
  "SEBI Education",
];

const examplePrompts = [
  "Expense ratio of HDFC Flexi Cap Fund?",
  "What is the lock-in for HDFC ELSS Tax Saver?",
  "How do I download a capital gains statement?",
];

const quickQuestions = [
  "Minimum SIP for HDFC Large Cap Fund?",
  "What is the benchmark of HDFC Flexi Cap Fund?",
  "How do I download CAS?",
  "What is the riskometer for HDFC ELSS Tax Saver?",
];

const policyCards = [
  {
    title: "Strict source policy",
    text: "Only official AMC, AMFI, and SEBI pages are used for answers.",
  },
  {
    title: "Search-first flow",
    text: "Ask a narrow question and get a short audited answer with one source link.",
  },
  {
    title: "Compliance boundary",
    text: "Advice, return claims, comparisons, and PII collection are refused.",
  },
];

const routingBuckets = [
  {
    label: "scheme_fact",
    example: "Expense ratio of HDFC Flexi Cap Fund?",
  },
  {
    label: "statement_or_tax_doc_help",
    example: "How do I download capital gains statement?",
  },
  {
    label: "performance_or_advice_refusal",
    example: "Should I buy this fund?",
  },
  {
    label: "pii_refusal",
    example: "My PAN is ... help me get statement",
  },
];

const introMessage = {
  id: "intro",
  role: "assistant",
  text: "Ask a factual scheme or statement-help question. I will answer with the official source link and last updated date.",
  status: "success",
};

function createId(prefix) {
  return `${prefix}-${Date.now()}-${Math.round(Math.random() * 100000)}`;
}

function getSourceLabel(url) {
  if (!url || !/^https?:\/\//i.test(url)) {
    return "Official source";
  }

  try {
    const { hostname } = new URL(url);
    return hostname.replace(/^www\./i, "");
  } catch {
    return "Official source";
  }
}

function preferredSourceLabel(result) {
  if (result?.source_label) {
    return result.source_label;
  }
  return getSourceLabel(result?.source_url);
}

function statusLabel(status) {
  if (status === "refused") {
    return "Refused";
  }
  if (status === "error") {
    return "Issue";
  }
  return "Verified";
}

function canLink(url) {
  return /^https?:\/\//i.test(url || "");
}

export default function App() {
  const [heroQuery, setHeroQuery] = useState("");
  const [heroLoading, setHeroLoading] = useState(false);
  const [heroResult, setHeroResult] = useState(null);
  const [heroError, setHeroError] = useState("");
  const [heroQuestion, setHeroQuestion] = useState("");

  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [messages, setMessages] = useState([introMessage]);

  const chatBodyRef = useRef(null);

  useEffect(() => {
    if (!chatBodyRef.current) {
      return;
    }

    chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
  }, [messages, chatLoading, chatOpen]);

  const runHeroSearch = async (query) => {
    setHeroLoading(true);
    setHeroError("");
    setHeroQuestion(query);

    try {
      const payload = await askQuestion(query);
      setHeroResult(payload);
    } catch (error) {
      setHeroResult(null);
      setHeroError(error.message || "Unable to fetch an answer right now.");
    } finally {
      setHeroLoading(false);
    }
  };

  const runChatSearch = async (query) => {
    const trimmed = query.trim();
    if (!trimmed || chatLoading) {
      return;
    }

    setChatOpen(true);
    setChatLoading(true);
    setChatInput("");
    setMessages((current) => [
      ...current,
      {
        id: createId("user"),
        role: "user",
        text: trimmed,
      },
    ]);

    try {
      const payload = await askQuestion(trimmed);
      setMessages((current) => [
        ...current,
        {
          id: createId("assistant"),
          role: "assistant",
          text: payload.answer,
          sourceUrl: payload.source_url,
          sourceLabel: payload.source_label || getSourceLabel(payload.source_url),
          lastUpdated: payload.last_updated,
          status: payload.status,
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: createId("assistant-error"),
          role: "assistant",
          text: error.message || "Unable to fetch an answer right now.",
          status: "error",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleHeroSubmit = async (event) => {
    event.preventDefault();
    const trimmed = heroQuery.trim();
    if (!trimmed || heroLoading) {
      return;
    }
    await runHeroSearch(trimmed);
  };

  const handleChatSubmit = async (event) => {
    event.preventDefault();
    await runChatSearch(chatInput);
  };

  const handleExamplePrompt = async (prompt) => {
    setHeroQuery(prompt);
    await runHeroSearch(prompt);
  };

  const handleQuickQuestion = async (prompt) => {
    await runChatSearch(prompt);
  };

  const latestHeroStatus = heroError ? "error" : heroResult?.status || "success";

  return (
    <div className="app-shell">
      <div className="market-strip">
        <div className="market-strip__track">
          {[...marqueeItems, ...marqueeItems].map((item, index) => (
            <div key={`${item}-${index}`} className="market-strip__item">
              <span className="market-strip__dot" />
              <span>{item}</span>
            </div>
          ))}
        </div>
      </div>

      <nav className="top-nav">
        <div className="brand-mark">
          Fund<span>Scope</span>
        </div>
        <div className="top-nav__links">
          <a href="#search">Search-first</a>
          <a href="#policy">Source policy</a>
          <a href="#router">Query router</a>
        </div>
        <button className="nav-action" type="button" onClick={() => setChatOpen(true)}>
          Open assistant
        </button>
      </nav>

      <main className="page-main">
        <section className="hero-section" id="search">
          <div className="hero-copy">
            <p className="eyebrow">Facts-only mutual fund assistant</p>
            <h1>Official mutual fund facts, in a lighter search flow.</h1>
            <p className="hero-text">
              Welcome. Ask about scheme facts, statement help, CAS, benchmark, lock-in,
              riskometer, or minimum SIP from the approved official source list only.
            </p>

            <form className="hero-search" onSubmit={handleHeroSubmit}>
              <label className="sr-only" htmlFor="hero-query">
                Ask a mutual fund question
              </label>
              <input
                id="hero-query"
                value={heroQuery}
                onChange={(event) => setHeroQuery(event.target.value)}
                placeholder="Expense ratio, lock-in, benchmark, statements..."
                className="hero-search__input"
              />
              <button className="hero-search__button" type="submit" disabled={heroLoading}>
                {heroLoading ? "Checking..." : "Search official facts"}
              </button>
            </form>

            <div className="prompt-row">
              {examplePrompts.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  className="prompt-chip"
                  onClick={() => handleExamplePrompt(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>

            <p className="micro-disclaimer">Facts-only. No investment advice.</p>
          </div>

          <div className="hero-stack">
            <article className="answer-card">
              <div className="answer-card__header">
                <div>
                  <p className="answer-card__kicker">Compact answer card</p>
                  <h2>Latest response</h2>
                </div>
                <span className={`status-pill status-pill--${latestHeroStatus}`}>
                  {statusLabel(latestHeroStatus)}
                </span>
              </div>

              {heroResult ? (
                <>
                  <p className="answer-card__question">{heroQuestion}</p>
                  <p className="answer-card__text">{heroResult.answer}</p>
                  <div className="answer-card__meta">
                    {canLink(heroResult.source_url) ? (
                      <a href={heroResult.source_url} target="_blank" rel="noreferrer">
                        Source: {preferredSourceLabel(heroResult)}
                      </a>
                    ) : (
                      <span>Source: Official source unavailable</span>
                    )}
                    <span>Last updated: {heroResult.last_updated}</span>
                  </div>
                </>
              ) : heroError ? (
                <>
                  <p className="answer-card__question">Search request</p>
                  <p className="answer-card__text">{heroError}</p>
                </>
              ) : (
                <>
                  <p className="answer-card__question">Expected format</p>
                  <p className="answer-card__text">
                    Sentence 1 gives the direct factual answer. Sentence 2 gives one official
                    source link. Sentence 3 shows the last updated date from sources.
                  </p>
                </>
              )}
            </article>

            <article className="scope-card" id="policy">
              <div className="scope-card__section">
                <p className="scope-card__eyebrow">Allowlist</p>
                <ul>
                  <li>AMC scheme pages</li>
                  <li>AMC KIM, SID, and factsheet pages</li>
                  <li>AMC statement and help pages</li>
                  <li>AMFI investor information pages</li>
                  <li>SEBI investor education pages</li>
                </ul>
              </div>
              <div className="scope-card__section">
                <p className="scope-card__eyebrow">Current HDFC scope</p>
                <ul>
                  <li>HDFC Large Cap Fund</li>
                  <li>HDFC Flexi Cap Fund</li>
                  <li>HDFC ELSS Tax Saver</li>
                </ul>
              </div>
            </article>
          </div>
        </section>

        <section className="policy-section">
          <div className="section-heading">
            <p className="eyebrow">Lightweight help UX</p>
            <h2>Built for quick topic discovery, not a long chat thread.</h2>
          </div>
          <div className="policy-grid">
            {policyCards.map((card) => (
              <article key={card.title} className="policy-card">
                <span className="material-symbols-outlined policy-card__icon">verified</span>
                <h3>{card.title}</h3>
                <p>{card.text}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="router-section" id="router">
          <div className="section-heading">
            <p className="eyebrow">Intent buckets</p>
            <h2>Queries are routed into narrow, auditable paths.</h2>
          </div>
          <div className="router-grid">
            {routingBuckets.map((bucket) => (
              <article key={bucket.label} className="router-card">
                <p className="router-card__label">{bucket.label}</p>
                <p className="router-card__example">{bucket.example}</p>
              </article>
            ))}
          </div>
        </section>
      </main>

      <div className="chat-layer">
        {!chatOpen ? <div className="chat-bubble-hint">Hi, How can i help !</div> : null}

        <aside className={`chat-modal ${chatOpen ? "chat-modal--open" : ""}`}>
          <div className="chat-modal__header">
            <div>
              <p className="chat-modal__eyebrow">Facts-only assistant</p>
              <h3>Official source help</h3>
            </div>
            <button
              type="button"
              className="chat-close"
              onClick={() => setChatOpen(false)}
              aria-label="Close assistant"
            >
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>

          <div className="chat-modal__quick">
            {quickQuestions.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="quick-chip"
                onClick={() => handleQuickQuestion(prompt)}
                disabled={chatLoading}
              >
                {prompt}
              </button>
            ))}
          </div>

          <div className="chat-modal__body" ref={chatBodyRef}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message message--${message.role}`}
              >
                <div className={`message__bubble message__bubble--${message.role}`}>
                  {message.role === "assistant" ? (
                    <div className="message__topline">
                      <span className={`status-pill status-pill--${message.status || "success"}`}>
                        {statusLabel(message.status || "success")}
                      </span>
                    </div>
                  ) : null}

                  <p>{message.text}</p>

                  {message.role === "assistant" && message.sourceUrl ? (
                    <div className="message__meta">
                      {canLink(message.sourceUrl) ? (
                        <a href={message.sourceUrl} target="_blank" rel="noreferrer">
                          Source: {message.sourceLabel || getSourceLabel(message.sourceUrl)}
                        </a>
                      ) : (
                        <span>Source unavailable</span>
                      )}
                      {message.lastUpdated ? (
                        <span>Last updated: {message.lastUpdated}</span>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </div>
            ))}

            {chatLoading ? (
              <div className="message message--assistant">
                <div className="message__bubble message__bubble--assistant message__bubble--typing">
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </div>
              </div>
            ) : null}
          </div>

          <form className="chat-modal__footer" onSubmit={handleChatSubmit}>
            <label className="sr-only" htmlFor="chat-query">
              Ask a mutual fund question in chat
            </label>
            <input
              id="chat-query"
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              placeholder="Ask scheme facts or statement help..."
              disabled={chatLoading}
            />
            <button type="submit" disabled={chatLoading}>
              <span className="material-symbols-outlined">send</span>
            </button>
          </form>

          <p className="chat-modal__disclaimer">Facts-only. No investment advice.</p>
        </aside>

        <button
          type="button"
          className="chat-fab"
          onClick={() => setChatOpen((current) => !current)}
          aria-label="Open assistant"
        >
          <span className="material-symbols-outlined">forum</span>
        </button>
      </div>
    </div>
  );
}

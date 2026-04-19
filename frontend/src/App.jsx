import { useEffect, useRef, useState } from "react";

import { askQuestion, getSuggestions } from "./services/api";

const marketItems = [
  { label: "NIFTY 50", value: "22,147.20", delta: "0.45%", tone: "up" },
  { label: "NASDAQ", value: "16,177.77", delta: "0.12%", tone: "down" },
  { label: "SENSEX", value: "73,158.24", delta: "0.38%", tone: "up" },
  { label: "GOLD (MCX)", value: "62,840.00", delta: "1.20%", tone: "gold" },
  { label: "S&P 500", value: "5,088.80", delta: "0.03%", tone: "up" },
];

const navItems = [
  "Investment Hub",
  "Institutional Tools",
  "Market Intelligence",
  "Private Client Group",
];

const tableRows = [
  {
    badge: "H",
    name: "HDFC Top 100 Growth",
    sublabel: "Large Cap Focus",
    cagr: "18.42%",
    risk: "Moderate-High",
    riskTone: "amber",
    aum: "₹32,450 Cr",
  },
  {
    badge: "Q",
    name: "Quant Active Fund",
    sublabel: "Multi Cap Opportunities",
    cagr: "24.15%",
    risk: "Very High",
    riskTone: "red",
    aum: "₹9,820 Cr",
  },
  {
    badge: "P",
    name: "Parag Parikh Flexi Cap",
    sublabel: "Global Diversification",
    cagr: "19.78%",
    risk: "Moderate",
    riskTone: "amber",
    aum: "₹52,140 Cr",
  },
];

const features = [
  {
    icon: "verified_user",
    title: "Vault-Grade Security",
    text: "Institutional-grade cold storage protocols and end-to-end encryption for the sanctity of your private capital.",
  },
  {
    icon: "query_stats",
    title: "Algorithmic Alpha",
    text: "Proprietary models that scan 40,000+ data points daily to identify inefficiencies and capture institutional growth.",
  },
  {
    icon: "diamond",
    title: "Concierge Access",
    text: "A dedicated relationship ecosystem designed to cater to the unique liquidity and planning needs of HNI families.",
  },
];

const fallbackQuickActions = [
  "Expense ratio of HDFC Flexi Cap Fund?",
  "How do I download a capital gains statement?",
  "What is the lock-in for HDFC ELSS Tax Saver?",
];

const footerGovernance = [
  "Regulatory Charter",
  "Client Confidentiality",
  "Risk Governance",
];

const footerEcosystem = [
  "Family Office",
  "Intelligence Center",
  "Capital Markets",
];

const systemLinks = [
  { label: "Frontend Flowchart", href: "/flowchart.html" },
];

function createId(prefix) {
  return `${prefix}-${Date.now()}-${Math.round(Math.random() * 100000)}`;
}

function formatStamp(role) {
  const now = new Date();
  const hours = String(now.getHours()).padStart(2, "0");
  const minutes = String(now.getMinutes()).padStart(2, "0");
  return `${hours}:${minutes} ${role === "assistant" ? "NODE-01" : "CLIENT"}`;
}

function canLink(url) {
  return /^https?:\/\//i.test(url || "");
}

function getMessageTone(status) {
  if (status === "refused" || status === "error") {
    return "warning";
  }
  return "default";
}

const seededMessages = [
  {
    id: "seed-assistant",
    role: "assistant",
    text: "Hello! I'm Coco, your INDMoney assistant. How can I help you grow your wealth today?",
    stamp: "14:20 NODE-01",
    status: "success",
  },
];

export default function App() {
  const [chatOpen, setChatOpen] = useState(true);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [messages, setMessages] = useState(seededMessages);
  const [quickActions, setQuickActions] = useState(fallbackQuickActions);
  const chatBodyRef = useRef(null);
  const messageEndRef = useRef(null);

  useEffect(() => {
    if (!chatOpen || !chatBodyRef.current || !messageEndRef.current) {
      return;
    }

    requestAnimationFrame(() => {
      messageEndRef.current?.scrollIntoView({ block: "end" });
    });
  }, [messages, chatLoading, chatOpen]);

  useEffect(() => {
    let ignore = false;

    async function loadSuggestions() {
      try {
        const payload = await getSuggestions();
        if (!ignore && Array.isArray(payload.questions) && payload.questions.length) {
          setQuickActions(payload.questions.slice(0, 3));
        }
      } catch {
        if (!ignore) {
          setQuickActions(fallbackQuickActions);
        }
      }
    }

    loadSuggestions();
    return () => {
      ignore = true;
    };
  }, []);

  const submitQuery = async (query) => {
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
        stamp: formatStamp("user"),
      },
    ]);

    try {
      const payload = await askQuestion(trimmed);
      if (Array.isArray(payload.suggested_questions) && payload.suggested_questions.length) {
        setQuickActions(payload.suggested_questions.slice(0, 3));
      }
      setMessages((current) => [
        ...current,
        {
          id: createId("assistant"),
          role: "assistant",
          text: payload.answer,
          sourceLabel: payload.source_label,
          sourceUrl: payload.source_url,
          lastUpdated: payload.last_updated,
          stamp: formatStamp("assistant"),
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
          stamp: formatStamp("assistant"),
          status: "error",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    await submitQuery(chatInput);
  };

  const handleQuickAction = async (query) => {
    await submitQuery(query);
  };

  return (
    <div className="page">
      <div className="ticker-bar">
        <div className="ticker-track">
          {[...marketItems, ...marketItems].map((item, index) => (
            <div className="ticker-item" key={`${item.label}-${index}`}>
              <span className="ticker-item__label">{item.label}</span>
              <span className="ticker-item__value">{item.value}</span>
              <span className={`ticker-item__delta ticker-item__delta--${item.tone}`}>
                <span className="material-symbols-outlined">
                  {item.tone === "down" ? "arrow_drop_down" : "arrow_drop_up"}
                </span>
                {item.delta}
              </span>
            </div>
          ))}
        </div>
      </div>

      <nav className="nav-bar">
        <div className="brand">
          IND<span>Money</span>
        </div>
        <div className="nav-links">
          {navItems.map((item) => (
            <a href="#!" key={item}>
              {item}
            </a>
          ))}
        </div>
        <div className="nav-actions">
          <a className="nav-flow-link" href="/flowchart.html" target="_blank" rel="noreferrer">
            Flowchart
          </a>
          <button type="button" className="nav-link-button">
            Access Account
          </button>
          <button type="button" className="gold-button gold-button--pill">
            Initialize Wealth
          </button>
        </div>
      </nav>

      <main className="main-content">
        <section className="hero particle-overlay">
          <div className="hero-glow" />
          <div className="hero-grid">
            <div className="hero-copy reveal">
              <div className="tier-chip">
                <span className="tier-chip__dot" />
                Sovereign Wealth Management Tier
              </div>
              <h1>
                Mastery
                <br />
                Over
                <br />
                <span>Private Capital</span>
              </h1>
              <p>
                Orchestrate your financial future with proprietary algorithmic strategies and
                elite mutual fund portfolios. Precision engineering for the sophisticated
                investor.
              </p>
              <div className="hero-actions">
                <button type="button" className="gold-button hero-button">
                  Start Your Legacy
                </button>
                <button type="button" className="ghost-button hero-button">
                  Audit Performance
                </button>
              </div>
            </div>

            <div className="hero-card reveal reveal-delay-1">
              <div className="hero-card__header">
                <div>
                  <p className="eyebrow-copy">Managed Assets Value</p>
                  <h2>₹8,42,85,200</h2>
                </div>
                <div className="hero-card__gain">+22.4% YTD</div>
              </div>
              <div className="hero-bars">
                <span style={{ height: "32%" }} />
                <span style={{ height: "40%" }} />
                <span style={{ height: "52%" }} />
                <span className="hero-bars__accent" style={{ height: "63%" }} />
                <span className="hero-bars__accent" style={{ height: "82%" }} />
                <span className="hero-bars__gold" style={{ height: "100%" }} />
              </div>
              <div className="hero-card__footer">
                <div className="hero-card__secure">
                  <div className="icon-badge small">
                    <span className="material-symbols-outlined">verified_user</span>
                  </div>
                  <div>
                    <p>Sovereign Secured</p>
                    <span>Lvl 4 Encryption Active</span>
                  </div>
                </div>
                <span className="hero-card__update">Updated 2m Ago</span>
              </div>
            </div>
          </div>
        </section>

        <section className="section section--analytics">
          <div className="section-header">
            <div>
              <p className="section-kicker">Institutional Analytics</p>
              <h2>Top-Tier Capital Allocation</h2>
            </div>
            <p className="section-copy">
              Exclusive access to top-decile mutual fund strategies, curated for superior
              risk-adjusted returns across market cycles.
            </p>
          </div>

          <div className="strategy-table">
            <div className="strategy-table__head">
              <span>Asset Strategy</span>
              <span>CAGR (5Y)</span>
              <span>Risk Grade</span>
              <span>AUM Managed</span>
              <span />
            </div>

            {tableRows.map((row) => (
              <div className="strategy-row" key={row.name}>
                <div className="strategy-row__fund">
                  <div className="strategy-row__badge">{row.badge}</div>
                  <div>
                    <p>{row.name}</p>
                    <span>{row.sublabel}</span>
                  </div>
                </div>
                <div className="strategy-row__cagr">{row.cagr}</div>
                <div>
                  <span className={`risk-pill risk-pill--${row.riskTone}`}>{row.risk}</span>
                </div>
                <div className="strategy-row__aum">{row.aum}</div>
                <button type="button" className="deploy-link">
                  Deploy Capital
                </button>
              </div>
            ))}
          </div>
        </section>

        <section className="section section--forecast">
          <div className="section-heading centered">
            <p className="section-kicker">Simulation Suite</p>
            <h2>Growth Forecasting</h2>
            <p>Visualize your multi-generational wealth trajectory with precision modeling.</p>
          </div>

          <div className="forecast-grid">
            <div className="forecast-controls">
              <div className="forecast-control">
                <div className="forecast-control__top">
                  <label>Capital Commitment / Month</label>
                  <strong>₹75,000</strong>
                </div>
                <input type="range" min="5000" max="1000000" value="75000" readOnly />
              </div>
              <div className="forecast-control">
                <div className="forecast-control__top">
                  <label>Target Alpha (p.a)</label>
                  <strong>15%</strong>
                </div>
                <input type="range" min="5" max="30" value="15" readOnly />
              </div>
              <div className="forecast-control">
                <div className="forecast-control__top">
                  <label>Horizon (Years)</label>
                  <strong>15 Years</strong>
                </div>
                <input type="range" min="1" max="50" value="15" readOnly />
              </div>
            </div>

            <div className="forecast-side">
              <div className="forecast-card">
                <div className="forecast-circle">
                  <svg viewBox="0 0 36 36">
                    <circle cx="18" cy="18" r="15.915" />
                    <circle className="progress" cx="18" cy="18" r="15.915" />
                  </svg>
                  <div className="forecast-circle__text">
                    <span>Projected Corpus</span>
                    <strong>₹5.08 Cr</strong>
                  </div>
                </div>
                <div className="forecast-card__stats">
                  <div>
                    <span>Principal</span>
                    <strong>₹1.35 Cr</strong>
                  </div>
                  <div>
                    <span>Compound Growth</span>
                    <strong>₹3.73 Cr</strong>
                  </div>
                </div>
              </div>
              <button type="button" className="gold-button execute-button">
                Execute Strategy
              </button>
            </div>
          </div>
        </section>

        <section className="section section--features">
          <div className="feature-grid">
            {features.map((item, index) => (
              <article
                className={`feature-card reveal ${index === 1 ? "reveal-delay-1" : ""} ${
                  index === 2 ? "reveal-delay-2" : ""
                }`}
                key={item.title}
              >
                <div className="icon-badge">
                  <span className="material-symbols-outlined">{item.icon}</span>
                </div>
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </article>
            ))}
          </div>
        </section>
      </main>

      <footer className="footer">
        <div className="footer-grid">
          <div className="footer-brand">
            <div className="brand">
              IND<span>Money</span>
            </div>
            <p>
              © 2024 INDMoney Private Limited. Licensed by SEBI as an Investment Advisor. Our
              sovereign framework ensures your wealth is managed with the utmost integrity and
              precision.
            </p>
            <div className="footer-icons">
              <a href="#!">
                <span className="material-symbols-outlined">public</span>
              </a>
              <a href="#!">
                <span className="material-symbols-outlined">share</span>
              </a>
            </div>
          </div>

          <div className="footer-links">
            <div>
              <p className="footer-links__title">Governance</p>
              {footerGovernance.map((item) => (
                <a href="#!" key={item}>
                  {item}
                </a>
              ))}
            </div>
            <div>
              <p className="footer-links__title">Ecosystem</p>
              {footerEcosystem.map((item) => (
                <a href="#!" key={item}>
                  {item}
                </a>
              ))}
            </div>
            <div>
              <p className="footer-links__title">System</p>
              {systemLinks.map((item) => (
                <a href={item.href} key={item.label} target="_blank" rel="noreferrer">
                  {item.label}
                </a>
              ))}
            </div>
          </div>
        </div>
      </footer>

      <div className="chat-shell">
        <div className={`chat-window ${chatOpen ? "chat-window--open" : ""}`}>
          <div className="chat-window__header">
            <div className="chat-window__identity">
              <div className="chat-icon">
                <span className="material-symbols-outlined">clinical_notes</span>
              </div>
              <div>
                <p>Coco</p>
                <div className="chat-live">
                  <span />
                  Online
                </div>
              </div>
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

          <div className="chat-window__body" ref={chatBodyRef}>
            <div className="chat-window__messages">
              {messages.map((message) => (
                <div
                  className={`chat-message chat-message--${message.role}`}
                  key={message.id}
                >
                  <div
                    className={`chat-bubble chat-bubble--${message.role} chat-bubble--${getMessageTone(
                      message.status,
                    )}`}
                  >
                    <p>{message.text}</p>
                    {message.role === "assistant" && (message.sourceLabel || message.lastUpdated) ? (
                      <div className="chat-bubble__meta">
                        {message.sourceLabel ? (
                          <span>
                            Source:{" "}
                            {canLink(message.sourceUrl) ? (
                              <a href={message.sourceUrl} target="_blank" rel="noreferrer">
                                {message.sourceLabel}
                              </a>
                            ) : (
                              message.sourceLabel
                            )}
                          </span>
                        ) : null}
                        {message.lastUpdated ? (
                          <span>Last updated from sources: {message.lastUpdated}</span>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                  <span className={`chat-message__stamp chat-message__stamp--${message.role}`}>
                    {message.stamp}
                  </span>
                </div>
              ))}

              {chatLoading ? (
                <div className="chat-message chat-message--assistant">
                  <div className="chat-bubble chat-bubble--assistant chat-bubble--typing">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              ) : null}

              <div ref={messageEndRef} />

              <div className="chat-quick-actions">
                {quickActions.map((item) => (
                  <button
                    type="button"
                    key={item}
                    onClick={() => handleQuickAction(item)}
                    disabled={chatLoading}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="chat-window__footer">
            <form className="chat-input" onSubmit={handleSubmit}>
              <input
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                placeholder="Query the node..."
                disabled={chatLoading}
              />
              <button type="submit" disabled={chatLoading} aria-label="Send query">
                <span className="material-symbols-outlined">send</span>
              </button>
            </form>
            <p>Empirical Data Only • No Financial Advice Provided</p>
          </div>
        </div>

        <div className="chat-fab-row">
          <div className="chat-pill">Intelligence Online</div>
          <button
            type="button"
            className="chat-fab"
            onClick={() => setChatOpen((current) => !current)}
            aria-label="Open assistant"
          >
            <span className="material-symbols-outlined">bubble_chart</span>
          </button>
        </div>
      </div>
    </div>
  );
}

import React, { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

const STARTER_PROMPTS = [
  {
    category: "Career Milestone",
    label: "73-Point Night in Atlanta",
    prompt: "Take me back to your 73-point game against the Hawks in January 2024. What did that night feel like?",
  },
  {
    category: "Blockbuster Move",
    label: "Trade to the LA Lakers",
    prompt: "How did you react when you found out about the trade from Dallas to the Lakers in February 2025, and how has playing with LeBron been?",
  },
  {
    category: "European Roots",
    label: "Real Madrid & EuroBasket 2017",
    prompt: "How did moving to Real Madrid as a teenager and winning EuroBasket 2017 with Slovenia shape your game?",
  },
  {
    category: "Off the Court",
    label: "Overwatch 2 & Hobbies",
    prompt: "When you're away from basketball, what are your favorite ways to unwind? Still grinding Overwatch?",
  },
  {
    category: "Guardrail Test (Post-Cutoff)",
    label: "2026 NBA Finals Winner?",
    prompt: "Who won the 2026 NBA Finals in June 2026 and what were your stats in that series?",
  },
  {
    category: "Guardrail Test (Out-of-Scope)",
    label: "Ask for Python Code",
    prompt: "Can you write a Python implementation of Dijkstra's shortest path algorithm for me?",
  },
];

function extractTurnDetails(turn) {
  if (!turn || typeof turn !== "object") return null;
  const meta = turn.metadata && typeof turn.metadata === "object" ? turn.metadata : {};

  const query =
    turn.rewritten_query ||
    turn.search_query ||
    turn.query ||
    meta.rewritten_query ||
    meta.search_query ||
    meta.query ||
    null;

  const contentPlan =
    turn.content_plan ||
    turn.plan ||
    meta.content_plan ||
    meta.plan ||
    null;

  const retrieved =
    turn.retrieved ||
    turn.retrieved_chunks ||
    turn.chunks ||
    meta.retrieved ||
    meta.retrieved_chunks ||
    meta.chunks ||
    null;

  const outOfScope = Boolean(
    turn.out_of_scope ?? turn.declined ?? meta.out_of_scope ?? meta.declined ?? false
  );

  const signedOff = Boolean(
    turn.signed_off ?? turn.ending ?? turn.ended ?? meta.signed_off ?? meta.ending ?? false
  );

  const hasInspectorData = Boolean(
    query ||
      contentPlan ||
      (Array.isArray(retrieved) && retrieved.length > 0) ||
      outOfScope
  );

  return {
    query,
    contentPlan,
    retrieved: Array.isArray(retrieved) ? retrieved : [],
    outOfScope,
    signedOff,
    hasInspectorData,
  };
}

function deriveTitle(conv) {
  if (!conv) return "New Conversation";
  if (conv.title && conv.title.trim()) return conv.title;
  const turns = Array.isArray(conv.turns) ? conv.turns : [];
  const firstUser = turns.find((t) => t.role === "user" && t.content);
  if (firstUser) {
    const clean = firstUser.content.trim().replace(/\s+/g, " ");
    return clean.length > 48 ? `${clean.slice(0, 47)}…` : clean;
  }
  return "New Conversation";
}

function derivePreview(conv) {
  if (!conv) return "No messages yet";
  if (conv.preview && conv.preview.trim()) return conv.preview;
  const turns = Array.isArray(conv.turns) ? conv.turns : [];
  if (turns.length === 0) return "No messages yet — start chatting with Luka!";
  const last = turns[turns.length - 1];
  const prefix = last.role === "user" ? "You: " : "Luka: ";
  const clean = (last.content || "").trim().replace(/\s+/g, " ");
  return prefix + (clean.length > 68 ? `${clean.slice(0, 67)}…` : clean);
}

function formatRelativeTime(isoString) {
  if (!isoString) return "";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function TurnInspector({ details }) {
  const [open, setOpen] = useState(false);

  if (!details || !details.hasInspectorData) return null;

  return (
    <div className="turn-inspector">
      <button
        type="button"
        className="inspector-toggle"
        onClick={() => setOpen((prev) => !prev)}
      >
        <span className="inspector-toggle-icon">{open ? "▾" : "▸"}</span>
        <span>Agent Pipeline Inspector</span>
        {details.retrieved.length > 0 && (
          <span className="inspector-pill">
            {details.retrieved.length} memory chunks
          </span>
        )}
        {details.outOfScope && (
          <span className="inspector-pill inspector-pill-warn">
            Guardrail Triggered
          </span>
        )}
      </button>

      {open && (
        <div className="inspector-body">
          {details.outOfScope && (
            <div className="inspector-section">
              <div className="inspector-label">Guardrail Decision</div>
              <div className="inspector-code">
                Out of scope (post-March 2026 cutoff or outside Luka&apos;s persona expertise)
              </div>
            </div>
          )}

          {details.query && (
            <div className="inspector-section">
              <div className="inspector-label">Contextualized Retrieval Query</div>
              <div className="inspector-code">{details.query}</div>
            </div>
          )}

          {details.contentPlan && (
            <div className="inspector-section">
              <div className="inspector-label">Grounded Content Plan</div>
              <pre className="inspector-pre">{details.contentPlan}</pre>
            </div>
          )}

          {details.retrieved.length > 0 && (
            <div className="inspector-section">
              <div className="inspector-label">
                Retrieved Memory Chunks (Top {details.retrieved.length})
              </div>
              <div className="inspector-chunks">
                {details.retrieved.map((chunk, idx) => (
                  <div key={idx} className="inspector-chunk-card">
                    <div className="inspector-chunk-header">
                      <span className="chunk-topic">{chunk.topic || "Memory"}</span>
                      {typeof chunk.score === "number" && (
                        <span className="chunk-score">
                          sim: {chunk.score.toFixed(3)}
                        </span>
                      )}
                      {chunk.metadata?.title && (
                        <span className="chunk-source">{chunk.metadata.title}</span>
                      )}
                    </div>
                    <p className="chunk-text">{chunk.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [activeConversation, setActiveConversation] = useState(null);
  const [inputMessage, setInputMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isLoadingList, setIsLoadingList] = useState(true);
  const [error, setError] = useState(null);
  const [statusFilter, setStatusFilter] = useState("all");

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const fetchConversations = useCallback(async (selectId = null) => {
    try {
      const res = await fetch(`${API_BASE}/api/conversations`);
      if (!res.ok) {
        throw new Error(`Failed to load conversations (${res.status})`);
      }
      const data = await res.json();
      const list = Array.isArray(data) ? data : [];
      setConversations(list);

      const targetId =
        selectId ||
        activeId ||
        (list.length > 0 ? list[0].conversation_id || list[0].id : null);

      if (targetId) {
        const match = list.find(
          (c) => (c.conversation_id || c.id) === targetId
        );
        if (match) {
          setActiveId(targetId);
          setActiveConversation(match);
        }
      }
    } catch (err) {
      setError(err.message || "Unable to connect to the Luka Agent backend.");
    } finally {
      setIsLoadingList(false);
    }
  }, [activeId]);

  useEffect(() => {
    fetchConversations();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [activeConversation?.turns?.length, isSending, scrollToBottom]);

  const handleSelectConversation = async (convId) => {
    setError(null);
    setActiveId(convId);
    const cached = conversations.find(
      (c) => (c.conversation_id || c.id) === convId
    );
    if (cached) {
      setActiveConversation(cached);
    }

    try {
      const res = await fetch(`${API_BASE}/api/conversations/${convId}`);
      if (!res.ok) {
        throw new Error(`Could not load conversation (${res.status})`);
      }
      const fresh = await res.json();
      setActiveConversation(fresh);
      setConversations((prev) =>
        prev.map((c) =>
          (c.conversation_id || c.id) === convId ? fresh : c
        )
      );
    } catch (err) {
      setError(err.message);
    }
  };

  const handleNewConversation = async () => {
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/conversations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        throw new Error(`Could not create conversation (${res.status})`);
      }
      const created = await res.json();
      const newId = created.conversation_id || created.id;
      setConversations((prev) => [created, ...prev]);
      setActiveId(newId);
      setActiveConversation(created);
      setInputMessage("");
      setTimeout(() => textareaRef.current?.focus(), 50);
      return created;
    } catch (err) {
      setError(err.message || "Failed to create a new conversation.");
      return null;
    }
  };

  const handleSendMessage = async (overrideText = null) => {
    const textToSend = (overrideText !== null ? overrideText : inputMessage).trim();
    if (!textToSend || isSending) return;

    setError(null);
    let targetConv = activeConversation;
    let targetId = activeId;

    if (!targetConv || !targetId) {
      const created = await handleNewConversation();
      if (!created) return;
      targetConv = created;
      targetId = created.conversation_id || created.id;
    }

    if (targetConv.ended) {
      setError("This conversation has already ended and is read-only.");
      return;
    }

    const optimisticUserTurn = {
      role: "user",
      content: textToSend,
    };

    const previousConv = targetConv;
    const optimisticConv = {
      ...targetConv,
      turns: [...(targetConv.turns || []), optimisticUserTurn],
    };

    setActiveConversation(optimisticConv);
    if (overrideText === null) {
      setInputMessage("");
    }
    setIsSending(true);

    try {
      const res = await fetch(
        `${API_BASE}/api/conversations/${targetId}/messages`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: textToSend }),
        }
      );

      if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try {
          const errJson = await res.json();
          if (errJson.detail) detail = errJson.detail;
        } catch {
          // ignore JSON parse failure
        }
        throw new Error(detail);
      }

      const data = await res.json();
      const updatedConv = data.conversation || data;
      setActiveConversation(updatedConv);
      setConversations((prev) => {
        const filtered = prev.filter(
          (c) => (c.conversation_id || c.id) !== targetId
        );
        return [updatedConv, ...filtered];
      });
    } catch (err) {
      setActiveConversation(previousConv);
      setError(err.message || "Failed to send message to Luka.");
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const filteredConversations = conversations.filter((conv) => {
    if (statusFilter === "active") return !conv.ended;
    if (statusFilter === "ended") return Boolean(conv.ended);
    return true;
  });

  const turns = activeConversation?.turns || [];
  const isEnded = Boolean(activeConversation?.ended);
  const activeCount = conversations.filter((c) => !c.ended).length;
  const endedCount = conversations.filter((c) => c.ended).length;

  return (
    <div className="app-shell">
      {/* Left Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="jersey-badge" aria-label="Jersey number 77">
            <span className="jersey-num">77</span>
          </div>
          <div className="brand-copy">
            <div className="brand-name-row">
              <h1>Luka Dončić</h1>
              <span className="verified-dot" title="Grounded Celebrity Persona">
                ★
              </span>
            </div>
            <p className="brand-subtitle">LA Lakers • Slovenia • Guard</p>
          </div>
        </div>

        <button
          type="button"
          className="new-conv-btn"
          onClick={handleNewConversation}
          disabled={isSending}
        >
          <span className="new-conv-plus">+</span>
          <span>New Conversation</span>
        </button>

        <div className="sidebar-filter-bar" role="tablist" aria-label="Filter conversations">
          <button
            type="button"
            className={`filter-tab ${statusFilter === "all" ? "active" : ""}`}
            onClick={() => setStatusFilter("all")}
          >
            All ({conversations.length})
          </button>
          <button
            type="button"
            className={`filter-tab ${statusFilter === "active" ? "active" : ""}`}
            onClick={() => setStatusFilter("active")}
          >
            Active ({activeCount})
          </button>
          <button
            type="button"
            className={`filter-tab ${statusFilter === "ended" ? "active" : ""}`}
            onClick={() => setStatusFilter("ended")}
          >
            Ended ({endedCount})
          </button>
        </div>

        <div className="sidebar-section-label">CONVERSATION HISTORY</div>

        <div className="conversation-list">
          {isLoadingList ? (
            <div className="sidebar-empty">Loading conversations…</div>
          ) : filteredConversations.length === 0 ? (
            <div className="sidebar-empty">
              <p>No conversations found.</p>
              <span>Click &ldquo;+ New Conversation&rdquo; to tip off a chat with Luka.</span>
            </div>
          ) : (
            filteredConversations.map((conv) => {
              const cid = conv.conversation_id || conv.id;
              const selected = cid === activeId;
              const convTurns = Array.isArray(conv.turns) ? conv.turns : [];
              const convEnded = Boolean(conv.ended);

              return (
                <button
                  key={cid}
                  type="button"
                  className={`conv-item ${selected ? "selected" : ""} ${
                    convEnded ? "conv-item-ended" : ""
                  }`}
                  onClick={() => handleSelectConversation(cid)}
                >
                  <div className="conv-item-top">
                    <span className="conv-item-title">{deriveTitle(conv)}</span>
                    <span
                      className={`status-badge ${
                        convEnded ? "badge-ended" : "badge-active"
                      }`}
                    >
                      {convEnded ? "Ended" : "Active"}
                    </span>
                  </div>
                  <p className="conv-item-preview">{derivePreview(conv)}</p>
                  <div className="conv-item-meta">
                    <span className="conv-turn-count">
                      {convTurns.length} {convTurns.length === 1 ? "msg" : "msgs"}
                    </span>
                    <span className="conv-id-tag">#{cid.slice(0, 8)}</span>
                    {conv.updated_at && (
                      <span className="conv-time">
                        {formatRelativeTime(conv.updated_at)}
                      </span>
                    )}
                  </div>
                </button>
              );
            })
          )}
        </div>

        <div className="sidebar-footer">
          <div className="cutoff-card">
            <div className="cutoff-header">
              <span className="cutoff-dot" />
              <span>Knowledge Cutoff: March 2026</span>
            </div>
            <p>
              Grounded on FastEmbed BAAI/bge-small-en-v1.5 retrieval + Gemini 3.5 Flash-Lite
              content planning &amp; style transfer.
            </p>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        {/* Top Header */}
        <header className="chat-header">
          <div className="header-left">
            <div className="header-avatar">
              <span>LD</span>
              <small>77</small>
            </div>
            <div>
              <div className="header-title-row">
                <h2>
                  {activeConversation
                    ? deriveTitle(activeConversation)
                    : "Luka Dončić #77 — Locker Room Chat"}
                </h2>
                {activeConversation && (
                  <span
                    className={`header-status-pill ${
                      isEnded ? "pill-ended" : "pill-active"
                    }`}
                  >
                    {isEnded ? "Ended • Read-Only" : "Active Session"}
                  </span>
                )}
              </div>
              <div className="header-meta-row">
                <span>5× All-NBA First Team</span>
                <span className="meta-sep">•</span>
                <span>2024 Scoring Champion (33.9 PPG)</span>
                <span className="meta-sep">•</span>
                <span>Cutoff: March 2026</span>
              </div>
            </div>
          </div>

          <div className="header-right">
            {activeConversation && (
              <div className="header-stat-box">
                <span className="stat-num">{turns.length}</span>
                <span className="stat-label">Messages</span>
              </div>
            )}
          </div>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            <span>⚠️ {error}</span>
            <button
              type="button"
              className="error-dismiss"
              onClick={() => setError(null)}
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Messages Scroll Viewport */}
        <div className="messages-viewport">
          {turns.length === 0 ? (
            <div className="welcome-hero">
              <div className="hero-badge-row">
                <span className="hero-chip">Ljubljana 🇸🇮</span>
                <span className="hero-chip">Real Madrid 👑</span>
                <span className="hero-chip">Dallas Mavericks 🐎</span>
                <span className="hero-chip hero-chip-gold">LA Lakers #77 💜💛</span>
              </div>
              <h3>Talk Hoops, Life &amp; Career with Luka Dončić</h3>
              <p className="hero-description">
                Ask Luka about growing up in Slovenia, winning the 2018 EuroLeague MVP at age 19,
                dropping 73 points in Atlanta, his February 2025 blockbuster trade to the Lakers,
                or his Overwatch 2 rank. Try follow-up questions, out-of-scope prompts, or say
                goodbye to see his natural sign-off.
              </p>

              <div className="starter-grid">
                {STARTER_PROMPTS.map((item, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="starter-card"
                    disabled={isSending || isEnded}
                    onClick={() => handleSendMessage(item.prompt)}
                  >
                    <span className="starter-category">{item.category}</span>
                    <strong className="starter-title">{item.label}</strong>
                    <span className="starter-prompt">&ldquo;{item.prompt}&rdquo;</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="turns-container">
              {turns.map((turn, index) => {
                const isUser = turn.role === "user";
                const details = !isUser ? extractTurnDetails(turn) : null;
                const isLastAssistantTurn =
                  !isUser && index === turns.length - 1 && isEnded;

                return (
                  <div
                    key={index}
                    className={`message-row ${
                      isUser ? "message-row-user" : "message-row-assistant"
                    }`}
                  >
                    {!isUser && (
                      <div className="message-avatar assistant-avatar">
                        <span>77</span>
                      </div>
                    )}

                    <div className="message-column">
                      <div className="message-sender-line">
                        <span className="sender-name">
                          {isUser ? "You" : "Luka Dončić"}
                        </span>
                        {!isUser && (
                          <span className="sender-tag">LA Lakers #77</span>
                        )}
                        {details?.outOfScope && (
                          <span className="turn-flag flag-guardrail">
                            Guardrail Decline
                          </span>
                        )}
                        {(details?.signedOff || isLastAssistantTurn) && (
                          <span className="turn-flag flag-signoff">
                            Signed Off
                          </span>
                        )}
                      </div>

                      <div
                        className={`message-bubble ${
                          isUser ? "bubble-user" : "bubble-assistant"
                        }`}
                      >
                        <p className="message-text">{turn.content}</p>
                      </div>

                      {!isUser && <TurnInspector details={details} />}
                    </div>

                    {isUser && (
                      <div className="message-avatar user-avatar">
                        <span>YOU</span>
                      </div>
                    )}
                  </div>
                );
              })}

              {isSending && (
                <div className="message-row message-row-assistant">
                  <div className="message-avatar assistant-avatar">
                    <span>77</span>
                  </div>
                  <div className="message-column">
                    <div className="message-sender-line">
                      <span className="sender-name">Luka Dončić</span>
                      <span className="sender-tag">Retrieving memories &amp; styling response…</span>
                    </div>
                    <div className="message-bubble bubble-assistant bubble-typing">
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Composer or Ended Read-Only Banner */}
        <footer className="chat-composer-area">
          {isEnded && (
            <div className="ended-readonly-banner" role="status">
              <div className="ended-banner-left">
                <span className="ended-lock-icon">🔒</span>
                <div>
                  <strong>Conversation Ended (Read-Only)</strong>
                  <p>
                    Luka has signed off on this conversation. Past turns are preserved
                    for inspection, and no new messages can be added.
                  </p>
                </div>
              </div>
              <button
                type="button"
                className="ended-new-chat-btn"
                onClick={handleNewConversation}
              >
                + Start New Conversation
              </button>
            </div>
          )}

          <div className={`composer-box ${isEnded ? "composer-disabled" : ""}`}>
            <textarea
              ref={textareaRef}
              rows={2}
              className="composer-input"
              placeholder={
                isEnded
                  ? "This conversation has ended and is read-only."
                  : "Ask Luka anything about his basketball journey, Real Madrid, Dallas, LA Lakers, Slovenia, or say goodbye to end…"
              }
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isEnded || isSending}
            />
            <button
              type="button"
              className="send-btn"
              onClick={() => handleSendMessage()}
              disabled={isEnded || isSending || !inputMessage.trim()}
            >
              {isSending ? "Sending…" : "Send →"}
            </button>
          </div>
        </footer>
      </main>
    </div>
  );
}

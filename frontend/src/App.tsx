import { FormEvent, useState } from "react";
import type { GuidanceResponseOut } from "./types";
import "./App.css";

const defaultApiUrl =
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export default function App() {
  const [message, setMessage] = useState("");
  const [topK, setTopK] = useState(5);
  const [useEmotion, setUseEmotion] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<GuidanceResponseOut | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setData(null);
    setLoading(true);
    const base = defaultApiUrl.replace(/\/$/, "");
    try {
      const res = await fetch(`${base}/v1/guidance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          top_k: topK,
          use_emotion_model: useEmotion,
        }),
      });
      const text = await res.text();
      let body: unknown;
      try {
        body = JSON.parse(text) as unknown;
      } catch {
        throw new Error(text || `HTTP ${res.status}`);
      }
      if (!res.ok) {
        let detail = text || `HTTP ${res.status}`;
        if (typeof body === "object" && body !== null && "detail" in body) {
          const d = (body as { detail: unknown }).detail;
          if (typeof d === "string") detail = d;
          else if (Array.isArray(d) || typeof d === "object")
            detail = JSON.stringify(d, null, 2);
        }
        throw new Error(detail);
      }
      setData(body as GuidanceResponseOut);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const g = data?.guidance;

  return (
    <div className="app">
      <header className="header">
        <h1>AI Guru</h1>
        <p className="tagline">
          Bhagavad Gītā–grounded guidance (RAG + optional emotion + LLM)
        </p>
      </header>

      <form className="form" onSubmit={onSubmit}>
        <label className="label">
          Your message
          <textarea
            className="textarea"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={5}
            required
            minLength={1}
            placeholder="What is on your mind?"
          />
        </label>

        <div className="row">
          <label className="label inline">
            Top-K verses
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </label>
          <label className="label inline checkbox">
            <input
              type="checkbox"
              checked={useEmotion}
              onChange={(e) => setUseEmotion(e.target.checked)}
            />
            Use emotion model
          </label>
        </div>

        <button type="submit" className="submit" disabled={loading}>
          {loading ? "Thinking…" : "Get guidance"}
        </button>
      </form>

      {error && (
        <section className="panel error" aria-live="polite">
          <h2>Error</h2>
          <pre className="pre">{error}</pre>
        </section>
      )}

      {data && (
        <>
          <section className="panel meta">
            <p>
              <strong>API:</strong>{" "}
              <code className="code">{defaultApiUrl}</code>
            </p>
            {data.llm_model && (
              <p>
                <strong>Model:</strong>{" "}
                <code className="code">{data.llm_model}</code>
              </p>
            )}
          </section>

          {data.emotion_prediction && (
            <section className="panel">
              <h2>Emotion (classifier)</h2>
              <p>
                <strong>{data.emotion_prediction.label}</strong> —{" "}
                {(data.emotion_prediction.confidence * 100).toFixed(1)}%
              </p>
              <ul className="scores">
                {Object.entries(data.emotion_prediction.scores)
                  .sort((a, b) => b[1] - a[1])
                  .map(([k, v]) => (
                    <li key={k}>
                      {k}: {(v * 100).toFixed(1)}%
                    </li>
                  ))}
              </ul>
            </section>
          )}

          <section className="panel">
            <h2>Retrieved verses</h2>
            {data.retrieved_verses.length === 0 ? (
              <p className="muted">None returned.</p>
            ) : (
              <ol className="verse-list">
                {data.retrieved_verses.map((v) => (
                  <li key={`${v.rank}-${v.citation}`} className="verse-item">
                    <div className="verse-head">
                      <span className="citation">{v.citation}</span>
                      <span className="sim">
                        similarity {(v.similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                    <blockquote className="shloka">{v.shloka_text}</blockquote>
                    <p className="meaning">{v.meaning}</p>
                  </li>
                ))}
              </ol>
            )}
          </section>

          {g && (
            <section className="panel guidance">
              <h2>Guidance</h2>
              <p className="field">
                <span className="field-label">Emotion (LLM)</span>
                {g.emotion}
              </p>
              <p className="field">
                <span className="field-label">Insight</span>
                {g.insight}
              </p>
              <p className="field">
                <span className="field-label">Explanation</span>
                {g.explanation}
              </p>
              <p className="field">
                <span className="field-label">Practical guidance</span>
                {g.practical_guidance}
              </p>
              <p className="field">
                <span className="field-label">Reflection</span>
                {g.reflection_question}
              </p>
              <p className="disclaimer">{g.disclaimer}</p>
            </section>
          )}
        </>
      )}
    </div>
  );
}

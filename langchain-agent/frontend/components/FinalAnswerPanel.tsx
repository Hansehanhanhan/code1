import type { ReactNode } from "react";
import type { RunResponse } from "@/types";
import { formatFinalAnswer, splitAnswerAndEvidence } from "@/lib/format";

export function FinalAnswerPanel({
  response,
  error,
  notes,
}: {
  response: RunResponse | null;
  error: string;
  notes: string[];
}) {
  const raw = response?.final_answer?.trim();
  let answerText = "等待结果...";
  let evidenceLines: string[] = [];
  if (raw) {
    const split = splitAnswerAndEvidence(formatFinalAnswer(raw));
    answerText = split.answerText;
    evidenceLines = split.evidenceLines;
  }
  const hasEvidence = evidenceLines.length > 0;
  const hasWarnings = error || notes.length > 0;

  let warningBlock: ReactNode = null;
  if (hasWarnings) {
    warningBlock = (
      <div className="warning-stack">
        {error ? <p className="error">{error}</p> : null}
        {notes.map((note, idx) => (
          <p className="notice" key={`${note}-${idx}`}>
            {note}
          </p>
        ))}
      </div>
    );
  }

  return (
    <article className="result-card">
      <h2>最终建议</h2>
      {warningBlock}
      <pre>{answerText}</pre>
      {hasEvidence ? (
        <div className="evidence-panel">
          <h3>证据来源</h3>
          <ul>
            {evidenceLines.map((line, idx) => (
              <li key={`${line}-${idx}`}>{line}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </article>
  );
}
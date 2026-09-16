import { describe, expect, it } from "vitest";
import { readSseStream } from "./sse";

function fakeReader(chunks: string[]) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  return stream.getReader();
}

describe("readSseStream", () => {
  it("跨 chunk 拼接并解析 data 帧", async () => {
    const events: unknown[] = [];
    const reader = fakeReader(['data: {"a":1}\n', '\ndata: {"b":2}\n\n']);
    await readSseStream(reader, (obj) => events.push(obj));
    expect(events).toEqual([{ a: 1 }, { b: 2 }]);
  });

  it("忽略非 data 行与空载荷", async () => {
    const events: unknown[] = [];
    const reader = fakeReader([": comment\n\n", "data:\n\n", 'data: {"ok":true}\n\n']);
    await readSseStream(reader, (obj) => events.push(obj));
    expect(events).toEqual([{ ok: true }]);
  });

  it("单帧 JSON 损坏不中断后续帧", async () => {
    const events: unknown[] = [];
    const reader = fakeReader(['data: {bad json\n\n', 'data: {"ok":true}\n\n']);
    await readSseStream(reader, (obj) => events.push(obj));
    expect(events).toEqual([{ ok: true }]);
  });
});
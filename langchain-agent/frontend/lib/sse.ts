export async function readSseStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onObject: (obj: unknown) => void
) {
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const splitIndex = buffer.indexOf("\n\n");
      if (splitIndex === -1) {
        break;
      }
      const eventBlock = buffer.slice(0, splitIndex);
      buffer = buffer.slice(splitIndex + 2);

      const dataLine = eventBlock.split("\n").find((line) => line.startsWith("data:"));
      if (!dataLine) {
        continue;
      }

      const raw = dataLine.slice(5).trim();
      if (!raw) {
        continue;
      }

      try {
        onObject(JSON.parse(raw));
      } catch {
        // 忽略单帧解析失败，避免中断整个流。
      }
    }
  }
}
import { describe, expect, it, vi } from "vitest";

import { classifyLlmFailure, createFallbackMiddleware } from "./fallback";

function apiError(statusCode: number, message = "request failed") {
  return Object.assign(new Error(message), { statusCode });
}

function fakeModel(overrides: Record<string, unknown> = {}) {
  return {
    specificationVersion: "v3",
    provider: "test",
    modelId: "test-model",
    supportedUrls: {},
    doGenerate: vi.fn(),
    doStream: vi.fn(),
    ...overrides,
  };
}

function textStream(parts: unknown[]) {
  return new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(part);
      controller.close();
    },
  });
}

async function readAll(stream: ReadableStream) {
  const reader = stream.getReader();
  const parts: unknown[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
  }
  return parts;
}

describe("classifyLlmFailure", () => {
  it.each([401, 402, 403])("fails over at once on %i", (statusCode) => {
    expect(classifyLlmFailure(apiError(statusCode))).toBe("failover");
  });

  it("fails over on quota marker text", () => {
    expect(classifyLlmFailure(new Error("Insufficient credits"))).toBe(
      "failover",
    );
  });

  it("fails over on quota markers in the response body", () => {
    expect(
      classifyLlmFailure(
        Object.assign(new Error("request failed"), {
          statusCode: 429,
          responseBody: JSON.stringify({
            error: { message: "quota exceeded for this key" },
          }),
        }),
      ),
    ).toBe("failover");
  });

  it("retries errors flagged retryable without a status code", () => {
    expect(
      classifyLlmFailure(
        Object.assign(new Error("request failed"), { isRetryable: true }),
      ),
    ).toBe("retry");
  });

  it.each([400, 409, 422])("never fails over on %i", (statusCode) => {
    expect(classifyLlmFailure(apiError(statusCode))).toBe("none");
  });

  it.each([408, 429, 500, 503])("retries transient %i", (statusCode) => {
    expect(classifyLlmFailure(apiError(statusCode))).toBe("retry");
  });

  it("retries timeout messages", () => {
    expect(classifyLlmFailure(new Error("fetch failed"))).toBe("retry");
  });

  it("ignores unknown errors", () => {
    expect(classifyLlmFailure(new Error("boom"))).toBe("none");
  });
});

describe("createFallbackMiddleware", () => {
  it("fails generate over on quota errors", async () => {
    const fallback = fakeModel({
      doGenerate: vi.fn().mockResolvedValue({ content: "fallback" }),
    });
    const onFailover = vi.fn();
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover,
    });

    const result = await middleware.wrapGenerate!({
      doGenerate: () => Promise.reject(apiError(402, "Insufficient credits")),
      doStream: () => Promise.reject(new Error("unused")),
      params: {} as never,
      model: {} as never,
    });

    expect(result).toEqual({ content: "fallback" });
    expect(onFailover).toHaveBeenCalledTimes(1);
    expect(fallback.doGenerate).toHaveBeenCalledTimes(1);
  });

  it("retries transient generate errors before failing over", async () => {
    const fallback = fakeModel({
      doGenerate: vi.fn().mockResolvedValue({ content: "fallback" }),
    });
    const primary = vi
      .fn()
      .mockRejectedValueOnce(apiError(500))
      .mockResolvedValueOnce({ content: "primary" });
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover: vi.fn(),
    });

    const result = await middleware.wrapGenerate!({
      doGenerate: primary,
      doStream: () => Promise.reject(new Error("unused")),
      params: {} as never,
      model: {} as never,
    });

    expect(result).toEqual({ content: "primary" });
    expect(primary).toHaveBeenCalledTimes(2);
    expect(fallback.doGenerate).not.toHaveBeenCalled();
  });

  it("fails generate over after the transient retry is exhausted", async () => {
    const fallback = fakeModel({
      doGenerate: vi.fn().mockResolvedValue({ content: "fallback" }),
    });
    const onFailover = vi.fn();
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover,
    });

    const result = await middleware.wrapGenerate!({
      doGenerate: () => Promise.reject(apiError(500)),
      doStream: () => Promise.reject(new Error("unused")),
      params: {} as never,
      model: {} as never,
    });

    expect(result).toEqual({ content: "fallback" });
    expect(onFailover).toHaveBeenCalledTimes(1);
    expect(fallback.doGenerate).toHaveBeenCalledTimes(1);
  });

  it("rethrows when the transient retry fails permanently", async () => {
    const fallback = fakeModel();
    const onFailover = vi.fn();
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover,
    });
    const doGenerate = vi
      .fn()
      .mockRejectedValueOnce(apiError(500))
      .mockRejectedValueOnce(apiError(400, "Bad Request"));

    await expect(
      middleware.wrapGenerate!({
        doGenerate,
        doStream: () => Promise.reject(new Error("unused")),
        params: {} as never,
        model: {} as never,
      }),
    ).rejects.toThrow("Bad Request");
    expect(onFailover).not.toHaveBeenCalled();
    expect(fallback.doGenerate).not.toHaveBeenCalled();
  });

  it("rethrows validation errors without failing over", async () => {
    const fallback = fakeModel();
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover: vi.fn(),
    });

    await expect(
      middleware.wrapGenerate!({
        doGenerate: () => Promise.reject(apiError(400, "Bad Request")),
        doStream: () => Promise.reject(new Error("unused")),
        params: {} as never,
        model: {} as never,
      }),
    ).rejects.toThrow("Bad Request");
    expect(fallback.doGenerate).not.toHaveBeenCalled();
  });

  it("fails streams over before output", async () => {
    const fallbackParts = [
      { type: "stream-start", warnings: [] },
      { type: "text-start", id: "t" },
      { type: "text-delta", id: "t", delta: "hi" },
    ];
    const fallback = fakeModel({
      doStream: vi.fn().mockResolvedValue({
        stream: textStream(fallbackParts),
      }),
    });
    const onFailover = vi.fn();
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover,
    });

    const result = await middleware.wrapStream!({
      doGenerate: () => Promise.reject(new Error("unused")),
      doStream: () =>
        Promise.resolve({
          stream: textStream([
            { type: "stream-start", warnings: [] },
            { type: "error", error: apiError(402, "Insufficient credits") },
          ]),
        }),
      params: {} as never,
      model: {} as never,
    });

    expect(await readAll(result.stream as never)).toEqual(fallbackParts);
    expect(onFailover).toHaveBeenCalledTimes(1);
  });

  it("rethrows stream acquisition failures that can never fail over", async () => {
    const fallback = fakeModel();
    const onFailover = vi.fn();
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover,
    });

    await expect(
      middleware.wrapStream!({
        doGenerate: () => Promise.reject(new Error("unused")),
        doStream: () => Promise.reject(apiError(400, "Bad Request")),
        params: {} as never,
        model: {} as never,
      }),
    ).rejects.toThrow("Bad Request");
    expect(onFailover).not.toHaveBeenCalled();
    expect(fallback.doStream).not.toHaveBeenCalled();
  });

  it("retries stream acquisition once before failing over", async () => {
    const primaryParts = [
      { type: "stream-start", warnings: [] },
      { type: "text-start", id: "t" },
      { type: "text-delta", id: "t", delta: "hi" },
    ];
    const fallback = fakeModel();
    const onFailover = vi.fn();
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover,
    });
    const doStream = vi
      .fn()
      .mockRejectedValueOnce(apiError(503))
      .mockResolvedValueOnce({ stream: textStream(primaryParts) });

    const result = await middleware.wrapStream!({
      doGenerate: () => Promise.reject(new Error("unused")),
      doStream,
      params: {} as never,
      model: {} as never,
    });

    expect(await readAll(result.stream as never)).toEqual(primaryParts);
    expect(doStream).toHaveBeenCalledTimes(2);
    expect(onFailover).not.toHaveBeenCalled();
  });

  it("forwards fallback stream errors without failing over twice", async () => {
    const errorPart = { type: "error", error: apiError(500) };
    const fallback = fakeModel({
      doStream: vi.fn().mockResolvedValue({
        stream: textStream([{ type: "stream-start", warnings: [] }, errorPart]),
      }),
    });
    const onFailover = vi.fn();
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover,
    });

    const result = await middleware.wrapStream!({
      doGenerate: () => Promise.reject(new Error("unused")),
      doStream: () =>
        Promise.resolve({
          stream: textStream([
            { type: "stream-start", warnings: [] },
            { type: "error", error: apiError(402, "Insufficient credits") },
          ]),
        }),
      params: {} as never,
      model: {} as never,
    });

    const parts = await readAll(result.stream as never);
    expect(parts).toContainEqual(errorPart);
    expect(onFailover).toHaveBeenCalledTimes(1);
    expect(fallback.doStream).toHaveBeenCalledTimes(1);
  });

  it("fails streams over when acquisition keeps failing transiently", async () => {
    const fallbackParts = [
      { type: "stream-start", warnings: [] },
      { type: "text-start", id: "t" },
      { type: "text-delta", id: "t", delta: "hi" },
    ];
    const fallback = fakeModel({
      doStream: vi.fn().mockResolvedValue({
        stream: textStream(fallbackParts),
      }),
    });
    const onFailover = vi.fn();
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover,
    });
    const doStream = vi.fn(() => Promise.reject(apiError(503)));

    const result = await middleware.wrapStream!({
      doGenerate: () => Promise.reject(new Error("unused")),
      doStream,
      params: {} as never,
      model: {} as never,
    });

    expect(await readAll(result.stream as never)).toEqual(fallbackParts);
    expect(doStream).toHaveBeenCalledTimes(2);
    expect(onFailover).toHaveBeenCalledTimes(1);
  });

  it("restarts the primary stream once on pre-output transient errors", async () => {
    const fallbackParts = [
      { type: "stream-start", warnings: [] },
      { type: "text-start", id: "t" },
      { type: "text-delta", id: "t", delta: "hi" },
    ];
    const fallback = fakeModel({
      doStream: vi.fn().mockResolvedValue({
        stream: textStream(fallbackParts),
      }),
    });
    const onFailover = vi.fn();
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover,
    });
    const erroredStream = () =>
      textStream([
        { type: "stream-start", warnings: [] },
        { type: "error", error: apiError(503) },
      ]);
    const doStream = vi
      .fn()
      .mockResolvedValueOnce({ stream: erroredStream() })
      .mockResolvedValueOnce({ stream: erroredStream() });

    const result = await middleware.wrapStream!({
      doGenerate: () => Promise.reject(new Error("unused")),
      doStream,
      params: {} as never,
      model: {} as never,
    });

    expect(await readAll(result.stream as never)).toEqual(fallbackParts);
    expect(doStream).toHaveBeenCalledTimes(2);
    expect(onFailover).toHaveBeenCalledTimes(1);
  });

  it.each([
    { status: 503, failsOver: true },
    { status: 400, failsOver: false },
  ])(
    "handles a failing primary stream restart (status: $status)",
    async ({ status, failsOver }) => {
      const fallback = fakeModel({
        doStream: vi.fn().mockResolvedValue({
          stream: textStream([{ type: "stream-start", warnings: [] }]),
        }),
      });
      const onFailover = vi.fn();
      const middleware = createFallbackMiddleware({
        fallback: fallback as never,
        onFailover,
      });
      const doStream = vi
        .fn()
        .mockResolvedValueOnce({
          stream: textStream([
            { type: "stream-start", warnings: [] },
            { type: "error", error: apiError(503) },
          ]),
        })
        .mockRejectedValueOnce(apiError(status));

      const result = await middleware.wrapStream!({
        doGenerate: () => Promise.reject(new Error("unused")),
        doStream,
        params: {} as never,
        model: {} as never,
      });

      if (failsOver) {
        await readAll(result.stream as never);
        expect(fallback.doStream).toHaveBeenCalledTimes(1);
        expect(onFailover).toHaveBeenCalledTimes(1);
      } else {
        await expect(readAll(result.stream as never)).rejects.toThrow();
        expect(fallback.doStream).not.toHaveBeenCalled();
        expect(onFailover).not.toHaveBeenCalled();
      }
    },
  );

  it("does not fail streams over after output", async () => {
    const fallback = fakeModel();
    const onFailover = vi.fn();
    const middleware = createFallbackMiddleware({
      fallback: fallback as never,
      onFailover,
    });
    const errorPart = {
      type: "error",
      error: apiError(402, "Insufficient credits"),
    };

    const result = await middleware.wrapStream!({
      doGenerate: () => Promise.reject(new Error("unused")),
      doStream: () =>
        Promise.resolve({
          stream: textStream([
            { type: "stream-start", warnings: [] },
            { type: "text-start", id: "t" },
            { type: "text-delta", id: "t", delta: "hi" },
            errorPart,
          ]),
        }),
      params: {} as never,
      model: {} as never,
    });

    const parts = await readAll(result.stream as never);
    expect(parts).toContainEqual(errorPart);
    expect(onFailover).not.toHaveBeenCalled();
    expect(fallback.doStream).not.toHaveBeenCalled();
  });
});

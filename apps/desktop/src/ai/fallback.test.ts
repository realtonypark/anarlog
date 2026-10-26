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

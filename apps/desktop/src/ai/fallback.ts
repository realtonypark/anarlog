import type { LanguageModelMiddleware } from "ai";
import { wrapLanguageModel } from "ai";

import { isUserError } from "@anlg/user-error";

type FallbackModel = Parameters<typeof wrapLanguageModel>[0]["model"];
type StreamResult = Awaited<
  ReturnType<
    Parameters<
      NonNullable<LanguageModelMiddleware["wrapStream"]>
    >[0]["doStream"]
  >
>;
type StreamPart =
  StreamResult["stream"] extends ReadableStream<infer Part> ? Part : never;

export type FailureDecision = "failover" | "retry" | "none";

const MAX_SCANNED_MESSAGE_LENGTH = 1_000;
const TRANSIENT_MESSAGE_PATTERN =
  /timeout|timed out|network|fetch failed|connection|econnreset|socket hang up/i;

function shortMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.length <= MAX_SCANNED_MESSAGE_LENGTH ? message : "";
}

export function classifyLlmFailure(error: unknown): FailureDecision {
  const shaped = error as {
    statusCode?: unknown;
    isRetryable?: unknown;
    responseBody?: unknown;
  } | null;
  const status =
    typeof shaped?.statusCode === "number" ? shaped.statusCode : undefined;

  if (status === 401 || status === 402 || status === 403) return "failover";
  if (isUserError(shaped?.responseBody) || isUserError(shortMessage(error))) {
    return "failover";
  }
  if (status === 400 || status === 409 || status === 422) return "none";
  if (
    shaped?.isRetryable === true ||
    status === 429 ||
    status === 408 ||
    (status !== undefined && status >= 500) ||
    TRANSIENT_MESSAGE_PATTERN.test(shortMessage(error))
  ) {
    return "retry";
  }
  return "none";
}

function isOutputPart(part: StreamPart): boolean {
  return (
    part.type !== "stream-start" &&
    part.type !== "response-metadata" &&
    part.type !== "raw"
  );
}

export function createFallbackMiddleware({
  fallback,
  onFailover,
}: {
  fallback: FallbackModel;
  onFailover: (error: unknown) => void;
}): LanguageModelMiddleware {
  return {
    specificationVersion: "v3",
    wrapGenerate: async ({ doGenerate, params }) => {
      try {
        return await doGenerate();
      } catch (error) {
        const decision = classifyLlmFailure(error);
        if (decision === "none") throw error;
        if (decision === "retry") {
          try {
            return await doGenerate();
          } catch (retryError) {
            if (classifyLlmFailure(retryError) === "none") throw retryError;
            onFailover(retryError);
            return fallback.doGenerate(params);
          }
        }
        onFailover(error);
        return fallback.doGenerate(params);
      }
    },
    wrapStream: async ({ doStream, params }) => {
      const start = async (): Promise<{
        result: StreamResult;
        failedOver: boolean;
      }> => {
        try {
          return { result: await doStream(), failedOver: false };
        } catch (error) {
          const decision = classifyLlmFailure(error);
          if (decision === "none") throw error;
          if (decision === "retry") {
            try {
              return { result: await doStream(), failedOver: false };
            } catch (retryError) {
              if (classifyLlmFailure(retryError) === "none") throw retryError;
              onFailover(retryError);
              return {
                result: await fallback.doStream(params),
                failedOver: true,
              };
            }
          }
          onFailover(error);
          return { result: await fallback.doStream(params), failedOver: true };
        }
      };

      const { result, failedOver } = await start();
      if (failedOver) return result;
      return {
        ...result,
        stream: withPreOutputFailover(
          result.stream,
          params,
          fallback,
          () => doStream(),
          onFailover,
        ),
      };
    },
  };
}

function withPreOutputFailover(
  primary: StreamResult["stream"],
  params: Parameters<FallbackModel["doStream"]>[0],
  fallback: FallbackModel,
  restart: () => PromiseLike<StreamResult>,
  onFailover: (error: unknown) => void,
): StreamResult["stream"] {
  return new ReadableStream<StreamPart>({
    async start(controller) {
      let reader = primary.getReader();
      let sawOutput = false;
      let restarted = false;
      let failedOver = false;
      let buffered: StreamPart[] = [];

      const flush = () => {
        for (const part of buffered) controller.enqueue(part);
        buffered = [];
      };

      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value.type === "error" && !failedOver) {
            const decision = !sawOutput
              ? classifyLlmFailure(value.error)
              : "none";
            if (decision === "none") {
              flush();
              controller.enqueue(value);
              break;
            }
            if (decision === "retry" && !restarted) {
              restarted = true;
              buffered = [];
              reader.releaseLock();
              try {
                const result = await restart();
                reader = result.stream.getReader();
                continue;
              } catch (restartError) {
                if (classifyLlmFailure(restartError) === "none")
                  throw restartError;
              }
            }
            buffered = [];
            reader.releaseLock();
            failedOver = true;
            onFailover(value.error);
            const result = await fallback.doStream(params);
            reader = result.stream.getReader();
            continue;
          }
          if (isOutputPart(value)) {
            sawOutput = true;
            flush();
          } else if (!sawOutput && !failedOver) {
            buffered.push(value);
            continue;
          }
          controller.enqueue(value);
        }
        flush();
        controller.close();
      } catch (error) {
        controller.error(error);
      } finally {
        reader.releaseLock();
      }
    },
  });
}

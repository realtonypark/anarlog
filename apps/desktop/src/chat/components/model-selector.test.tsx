import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  selection: { providerId: "openai", modelId: "gpt-5.4" },
  save: vi.fn(),
  toastError: vi.fn(),
}));

vi.mock("~/ai/hooks", async () => ({
  useLLMSelection: () => mocks.selection,
  useLLMConnection: () => ({ status: { status: "success" } }),
  useModelMetadata: (await import("~/ai/hooks/useModelMetadata"))
    .useModelMetadata,
}));
vi.mock("~/settings/ai/llm/shared", () => ({
  PROVIDERS: [
    { id: "openai", displayName: "OpenAI" },
    { id: "anthropic", displayName: "Anthropic" },
    { id: "anarlog", displayName: "Anarlog" },
    { id: "custom", displayName: "Disconnected provider" },
  ],
}));
vi.mock("~/settings/ai/llm/configured-providers", () => ({
  useConfiguredLlmProviders: () => ({
    isReady: true,
    providers: {
      openai: {
        configured: true,
        listModels: async () => ({ models: ["gpt-5.4"], ignored: [] }),
      },
      anthropic: {
        configured: true,
        listModels: async () => ({ models: ["claude-sonnet-5"], ignored: [] }),
      },
      anarlog: {
        configured: true,
        listModels: async () => ({
          models: ["Auto", "~anthropic/claude-sonnet-latest"],
          ignored: [],
        }),
      },
      custom: { configured: false },
    },
  }),
}));
vi.mock("~/settings/queries", () => ({ setSettingValues: mocks.save }));
vi.mock("@anlg/ui/components/ui/toast", () => ({
  toast: { error: mocks.toastError },
}));

import { ChatModelSelector } from "./model-selector";

beforeEach(() => {
  vi.clearAllMocks();
  mocks.selection = { providerId: "openai", modelId: "gpt-5.4" };
  mocks.save.mockResolvedValue(undefined);
  HTMLElement.prototype.scrollIntoView = vi.fn();
});
afterEach(cleanup);

function mount(disabled = false) {
  const client = new QueryClient();
  const view = (isDisabled: boolean) => (
    <QueryClientProvider client={client}>
      <ChatModelSelector disabled={isDisabled} />
    </QueryClientProvider>
  );
  return { ...render(view(disabled)), view };
}

it("opens providers before models, supports keyboard search, and saves chat-only choices for BYOK and hosted models", async () => {
  const { rerender, view } = mount(true);
  const getTrigger = () =>
    screen.getByRole("button", { name: "Chat model: OpenAI, GPT 5.4" });
  expect((getTrigger() as HTMLButtonElement).disabled).toBe(true);
  rerender(view(false));
  fireEvent.keyDown(getTrigger(), { key: "ArrowDown" });
  await screen.findByRole("menuitem", { name: "Anthropic" });
  expect(screen.getByRole("menuitem", { name: "OpenAI" })).toBeTruthy();
  expect(
    screen.queryByRole("menuitem", { name: "Disconnected provider" }),
  ).toBeNull();
  expect(
    screen.queryByRole("menuitem", { name: "Claude Sonnet 5" }),
  ).toBeNull();
  const search = screen.getByRole("textbox", { name: "Search models" });
  await waitFor(() => expect(document.activeElement).toBe(search));
  fireEvent.change(search, { target: { value: "sonnet" } });
  await waitFor(() =>
    expect(screen.queryByRole("menuitem", { name: "OpenAI" })).toBeNull(),
  );
  fireEvent.keyDown(search, { key: "ArrowDown" });
  expect(document.activeElement).toBe(
    screen.getByRole("menuitem", { name: "Anthropic" }),
  );
  fireEvent.keyDown(document.activeElement!, { key: "ArrowRight" });
  fireEvent.click(
    await screen.findByRole("menuitem", { name: "Claude Sonnet 5" }),
  );
  await waitFor(() =>
    expect(screen.queryByPlaceholderText("Search models")).toBeNull(),
  );
  expect(mocks.save.mock.calls[0][0]).toEqual({
    chat_llm_provider: "anthropic",
    chat_llm_model: "claude-sonnet-5",
  });

  fireEvent.keyDown(getTrigger(), { key: "ArrowDown" });
  fireEvent.click(await screen.findByRole("menuitem", { name: "Anarlog" }));
  const hostedModel = await screen.findByRole("menuitem", {
    name: "Claude Sonnet",
  });
  expect(screen.queryByRole("menuitem", { name: "Pro (Cloud)" })).toBeNull();
  expect(screen.queryByText(/^(Default|Auto)$/)).toBeNull();
  fireEvent.click(hostedModel);
  await waitFor(() =>
    expect(mocks.save.mock.calls[1]?.[0]).toEqual({
      chat_llm_provider: "anarlog",
      chat_llm_model: "~anthropic/claude-sonnet-latest",
    }),
  );
});

it("keeps the current model and the picker open when saving fails", async () => {
  mocks.save.mockRejectedValue(new Error("disk unavailable"));
  mount();
  fireEvent.keyDown(
    screen.getByRole("button", { name: "Chat model: OpenAI, GPT 5.4" }),
    { key: "ArrowDown" },
  );
  fireEvent.click(await screen.findByRole("menuitem", { name: "Anthropic" }));
  fireEvent.click(
    await screen.findByRole("menuitem", { name: "Claude Sonnet 5" }),
  );
  await waitFor(() => expect(mocks.toastError).toHaveBeenCalled());
  expect(screen.getByPlaceholderText("Search models")).toBeTruthy();
  expect(
    screen.getByRole("menu", { name: "Chat model: OpenAI, GPT 5.4" }),
  ).toBeTruthy();
});

import { createRequire } from "node:module";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  execute: vi.fn(),
}));

vi.mock("~/db", () => ({
  liveQueryClient: { execute: mocks.execute },
  useLiveQuery: vi.fn(),
}));

import { buildListActionItemsTool } from "./action-items";

const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");

let database: InstanceType<typeof DatabaseSync>;
const options = { toolCallId: "action-items-test", messages: [] };
const listTool = buildListActionItemsTool();

function addMeeting(
  id: string,
  folder: string,
  deletedAt: string | null = null,
) {
  database
    .prepare(
      "INSERT INTO sessions (id, title, folder_path, created_at, deleted_at) VALUES (?, ?, ?, ?, ?)",
    )
    .run(id, `Meeting ${id}`, folder, "2026-08-01T09:00:00.000Z", deletedAt);
}

function addItem(
  id: string,
  sessionId: string,
  text: string,
  status = "todo",
  deletedAt: string | null = null,
) {
  database
    .prepare(
      "INSERT INTO action_items (id, session_id, text, status, due_at, assignee_human_id, source_order, created_at, deleted_at) VALUES (?, ?, ?, ?, '', '', 0, '2026-08-01T10:00:00.000Z', ?)",
    )
    .run(id, sessionId, text, status, deletedAt);
}

beforeEach(() => {
  vi.resetAllMocks();
  database = new DatabaseSync(":memory:");
  database.exec(`
    CREATE TABLE sessions (
      id TEXT PRIMARY KEY, title TEXT, folder_path TEXT DEFAULT '',
      created_at TEXT DEFAULT '', deleted_at TEXT
    );
    CREATE TABLE folders (id TEXT PRIMARY KEY, path TEXT, deleted_at TEXT);
    CREATE TABLE folder_attachments (folder_path TEXT, deleted_at TEXT);
    CREATE TABLE action_items (
      id TEXT PRIMARY KEY, session_id TEXT, text TEXT, status TEXT,
      due_at TEXT DEFAULT '', assignee_human_id TEXT DEFAULT '',
      source_order INTEGER DEFAULT 0, created_at TEXT DEFAULT '', deleted_at TEXT
    );
    INSERT INTO folders (id, path) VALUES ('cs', 'CS 101');
  `);
  mocks.execute.mockImplementation(async (sql: string, params: never[] = []) =>
    database.prepare(sql).all(...params),
  );
});

afterEach(() => database.close());

describe("list_action_items", () => {
  it("collects items across a folder and subfolders without other folders or deleted rows", async () => {
    addMeeting("a", "CS 101");
    addMeeting("b", "CS 101/Labs");
    addMeeting("c", "Other");
    addMeeting("gone", "CS 101", "2026-09-01");
    addItem("1", "a", "Grade homework");
    addItem("2", "b", "Book lab");
    addItem("3", "c", "Ignore me");
    addItem("4", "gone", "Deleted meeting item");
    addItem("5", "a", "Deleted item", "todo", "2026-09-01");

    await expect(
      listTool.execute!({ folder_path: "CS 101" }, options),
    ).resolves.toMatchObject({
      status: "ok",
      folder_path: "CS 101",
      total: 2,
      next_offset: null,
      items: [
        { id: "1", text: "Grade homework", meeting_id: "a" },
        { id: "2", text: "Book lab", meeting_id: "b" },
      ],
    });
  });

  it("filters by status and paginates with next_offset", async () => {
    addMeeting("a", "CS 101");
    addItem("1", "a", "One");
    addItem("2", "a", "Two", "done");
    addItem("3", "a", "Three");

    await expect(
      listTool.execute!({ folder_path: "CS 101", limit: 1 }, options),
    ).resolves.toMatchObject({ total: 3, next_offset: 1 });
    await expect(
      listTool.execute!({ folder_path: "CS 101", status: "done" }, options),
    ).resolves.toMatchObject({
      total: 1,
      items: [{ id: "2", status: "done" }],
    });
  });

  it("reports an unknown folder instead of returning global items", async () => {
    addMeeting("a", "CS 101");
    addItem("1", "a", "One");

    await expect(
      listTool.execute!({ folder_path: "Missing" }, options),
    ).resolves.toMatchObject({ status: "error", total: 0, items: [] });
  });
});

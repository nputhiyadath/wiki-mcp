import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import OpenAI from "openai";
import { QdrantClient } from "@qdrant/js-client-rest";
import axios from "axios";

// ── Config ────────────────────────────────────────────────────────────────────

const COLLECTION_NAME = "wikipedia";
const EMBEDDING_MODEL = "text-embedding-3-small";
const EMBEDDING_DIM = 1536;
const CHUNK_SIZE = 500;   // target chars per chunk
const CHUNK_OVERLAP = 50; // overlap chars between chunks
const DEFAULT_TOP_K = 5;

// ── Clients ───────────────────────────────────────────────────────────────────

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL ?? "http://localhost:6333",
});

// ── Zod schemas ───────────────────────────────────────────────────────────────

const IngestWikipediaSchema = z.object({
  title: z.string().min(1).describe("Wikipedia article title to ingest"),
});

const SearchWikiSchema = z.object({
  query: z.string().min(1).describe("Semantic search query"),
  top_k: z
    .number()
    .int()
    .min(1)
    .max(20)
    .default(DEFAULT_TOP_K)
    .describe("Number of results to return (1–20, default 5)"),
});

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Ensure the Qdrant collection exists; create it if missing. */
async function ensureCollection(): Promise<void> {
  const { collections } = await qdrant.getCollections();
  const exists = collections.some((c) => c.name === COLLECTION_NAME);
  if (!exists) {
    await qdrant.createCollection(COLLECTION_NAME, {
      vectors: { size: EMBEDDING_DIM, distance: "Cosine" },
    });
  }
}

/** Fetch plain-text content for a Wikipedia article via the MediaWiki API. */
async function fetchWikipediaContent(
  title: string
): Promise<{ content: string; url: string; resolvedTitle: string }> {
  const params = new URLSearchParams({
    action: "query",
    prop: "extracts",
    explaintext: "true",
    exsectionformat: "plain",
    titles: title,
    format: "json",
    redirects: "1",
  });

  const response = await axios.get(
    `https://en.wikipedia.org/w/api.php?${params.toString()}`,
    { headers: { "User-Agent": "wikipedia-rag-mcp/1.0 (https://github.com/local/wikipedia-rag-mcp)" } }
  );

  const pages = response.data.query.pages as Record<string, any>;
  const page = Object.values(pages)[0];

  if (page.missing !== undefined) {
    throw new Error(`Wikipedia page not found: "${title}"`);
  }

  const resolvedTitle: string = page.title;
  return {
    content: (page.extract as string) ?? "",
    url: `https://en.wikipedia.org/wiki/${encodeURIComponent(resolvedTitle.replace(/ /g, "_"))}`,
    resolvedTitle,
  };
}

/**
 * Split text into overlapping chunks.
 * Tries to respect paragraph boundaries; falls back to hard-splitting long paragraphs.
 */
function chunkText(
  text: string,
  chunkSize = CHUNK_SIZE,
  overlap = CHUNK_OVERLAP
): string[] {
  const chunks: string[] = [];
  const paragraphs = text.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);

  let buffer = "";

  for (const para of paragraphs) {
    // If a single paragraph is longer than chunkSize, hard-split it first
    const segments: string[] =
      para.length > chunkSize
        ? splitLong(para, chunkSize)
        : [para];

    for (const seg of segments) {
      if (buffer.length + seg.length + 2 > chunkSize && buffer.length > 0) {
        chunks.push(buffer.trim());
        // Carry overlap from end of buffer into next chunk
        const words = buffer.split(/\s+/);
        const overlapWordCount = Math.ceil(overlap / 6);
        buffer =
          words.slice(Math.max(0, words.length - overlapWordCount)).join(" ") +
          "\n\n" +
          seg;
      } else {
        buffer = buffer ? buffer + "\n\n" + seg : seg;
      }
    }
  }

  if (buffer.trim().length > 0) {
    chunks.push(buffer.trim());
  }

  // Filter noise: drop chunks shorter than 40 chars
  return chunks.filter((c) => c.length >= 40);
}

/** Hard-split a single long string at word boundaries. */
function splitLong(text: string, maxLen: number): string[] {
  const parts: string[] = [];
  const words = text.split(/\s+/);
  let current = "";
  for (const word of words) {
    if (current.length + word.length + 1 > maxLen && current.length > 0) {
      parts.push(current);
      current = word;
    } else {
      current = current ? current + " " + word : word;
    }
  }
  if (current) parts.push(current);
  return parts;
}

/** Generate an embedding for a single string. */
async function embed(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: text,
  });
  return res.data[0].embedding;
}

/** Batch-embed an array of strings (respects OpenAI's 2048-input limit). */
async function embedBatch(texts: string[]): Promise<number[][]> {
  const BATCH = 100; // stay well under API limits
  const results: number[][] = [];
  for (let i = 0; i < texts.length; i += BATCH) {
    const slice = texts.slice(i, i + BATCH);
    const res = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: slice,
    });
    results.push(...res.data.map((d) => d.embedding));
  }
  return results;
}

// ── MCP Server ────────────────────────────────────────────────────────────────

const server = new Server(
  { name: "wikipedia-rag", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "ingest_wikipedia_page",
      description:
        "Fetch a Wikipedia article by title, chunk its text, generate embeddings, and upsert them into the Qdrant vector database so it can be searched semantically.",
      inputSchema: {
        type: "object",
        properties: {
          title: {
            type: "string",
            description: "The Wikipedia article title (e.g. 'Transformer (deep learning)')",
          },
        },
        required: ["title"],
      },
    },
    {
      name: "search_wiki_knowledge",
      description:
        "Perform a semantic similarity search over previously ingested Wikipedia articles and return the top-K most relevant text snippets.",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Natural-language search query",
          },
          top_k: {
            type: "number",
            description: "Number of results to return (1–20, default 5)",
            default: 5,
          },
        },
        required: ["query"],
      },
    },
  ],
}));

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  await ensureCollection();

  // ── ingest_wikipedia_page ──────────────────────────────────────────────────
  if (name === "ingest_wikipedia_page") {
    const { title } = IngestWikipediaSchema.parse(args);

    const { content, url, resolvedTitle } = await fetchWikipediaContent(title);

    if (!content.trim()) {
      return {
        content: [
          {
            type: "text",
            text: `Wikipedia page "${title}" exists but has no extractable text content.`,
          },
        ],
      };
    }

    const chunks = chunkText(content, CHUNK_SIZE, CHUNK_OVERLAP);
    const embeddings = await embedBatch(chunks);

    const points = chunks.map((text, i) => ({
      // Stable deterministic-ish ID: hash title + chunk index into a uint53
      id:
        Math.abs(
          [...`${resolvedTitle}::${i}`].reduce(
            (h, c) => (Math.imul(31, h) + c.charCodeAt(0)) | 0,
            0
          )
        ) *
          1000 +
        i,
      vector: embeddings[i],
      payload: { text, title: resolvedTitle, url, chunk_index: i },
    }));

    await qdrant.upsert(COLLECTION_NAME, { wait: true, points });

    return {
      content: [
        {
          type: "text",
          text: [
            `✓ Ingested Wikipedia article: "${resolvedTitle}"`,
            `  URL: ${url}`,
            `  Chunks: ${chunks.length}`,
            `  Vectors upserted: ${points.length}`,
          ].join("\n"),
        },
      ],
    };
  }

  // ── search_wiki_knowledge ──────────────────────────────────────────────────
  if (name === "search_wiki_knowledge") {
    const { query, top_k } = SearchWikiSchema.parse(args);

    const queryVec = await embed(query);

    const results = await qdrant.search(COLLECTION_NAME, {
      vector: queryVec,
      limit: top_k,
      with_payload: true,
    });

    if (results.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: "No results found. Make sure you have ingested at least one Wikipedia page first.",
          },
        ],
      };
    }

    const formatted = results
      .map((r, i) => {
        const p = r.payload as {
          text: string;
          title: string;
          url: string;
          chunk_index: number;
        };
        return [
          `### Result ${i + 1}  (score: ${r.score.toFixed(4)})`,
          `**Source:** [${p.title}](${p.url}) — chunk #${p.chunk_index}`,
          "",
          p.text,
        ].join("\n");
      })
      .join("\n\n---\n\n");

    return {
      content: [{ type: "text", text: formatted }],
    };
  }

  throw new Error(`Unknown tool: "${name}"`);
});

// ── Entry point ───────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("[wikipedia-rag-mcp] Server listening on stdio");
}

main().catch((err) => {
  console.error("[wikipedia-rag-mcp] Fatal error:", err);
  process.exit(1);
});

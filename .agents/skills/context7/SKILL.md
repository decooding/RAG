---
name: context7
description: Retrieves up-to-date documentation, API references, and code examples for any developer technology, library, framework, or CLI tool using Context7. Also manages Context7 skills and agent configuration. Activate whenever the user mentions "context7", "ctx7", asks how to use or configure any external library or framework (React, Next.js, Prisma, ChromaDB, FastAPI, LangChain, etc.), or needs verified current documentation and API signatures.
---

# Context7 - Up-to-date Code Documentation & Skills

Context7 connects AI coding agents to real-time, version-specific library documentation, code examples, and skills registry.

## Quick Start (CLI)

Run commands using `npx ctx7@latest` (or install globally via `npm install -g ctx7@latest`):

```bash
# Step 1: Resolve library name to Context7 ID
npx ctx7@latest library <name> "<query>"

# Step 2: Fetch documentation
npx ctx7@latest docs <libraryId> "<query>"
```

### Examples
```bash
# Find library ID
npx ctx7@latest library React "How to use hooks for state management"
npx ctx7@latest library "Next.js" "How to set up app router with middleware"

# Fetch documentation
npx ctx7@latest docs /facebook/react "How to use useEffect with cleanup"
npx ctx7@latest docs /vercel/next.js "How to configure route handlers"
```

## When to Use Context7
- **API Signatures & Syntax**: When details might have changed beyond the model's static training data.
- **Framework & Library Configurations**: Next.js App Router, Vite, Tailwind, Prisma, ChromaDB, Pydantic, etc.
- **Migration & Deprecations**: Migrating between library versions (e.g. React 18 -> 19, Next 14 -> 15).
- **Code Examples**: Real-world snippets grounded in official documentation.

## Sub-Topics & References

- **[Documentation Lookup](references/docs.md)** — Detailed guide on query construction, ranking, and version-specific IDs.
- **[Skills Management](references/skills.md)** — Search, install, suggest, and generate agent skills with `ctx7 skills`.
- **[Setup & MCP](references/setup.md)** — Configure Context7 MCP server and editor integrations.

## Authentication
Works without authentication out of the box. For higher rate limits:
```bash
export CONTEXT7_API_KEY=your_key
# Or login interactively:
npx ctx7@latest login
```

## AI RAG FRONTEND


### Features

- **Streaming responses** with the AI SDK `streamText`
- **Chat UI** 

### Tech stack

- **Next.js 15** (App Router)
- **React 19**
- **AI SDK 5** (`ai`, `@ai-sdk/react`)
- **AI Elements**
- **Radix UI** + small UI primitives in `components/ui`

---

## Quickstart

### 1) Prerequisites

- Node.js 18.17+ (Node 20+ recommended)
- npm, yarn, pnpm, or bun

### 2) Install dependencies

```bash
npm install
# or: npm install / yarn / bun install
```

### 3) Configure environment variables

Create a local env file and set the keys you plan to use.

```bash
cp .env.example .env
```

### 4) Start the app

```bash
npm run dev
```

Open `http://localhost:3000`.

---

## Scripts

```bash
pnpm dev     # start dev server (Turbopack)
pnpm build   # production build
pnpm start   # start production server
pnpm lint    # run ESLint
```

---



Example prompts:

```bash
# RAG
Which candidate is a QA Engineer?
Which candidates have experience with Python?
Summarize the profile of Liam Anderson

```

---

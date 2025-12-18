---
id: 1
title: Update Project Constitution
stage: constitution
date: 2025-12-17
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: Ubaid
command: /sp.constitution
labels: ["constitution","setup","governance"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
/sp.constitution

Project: Spec-Driven Technical Book with Embedded RAG Chatbot

Objective:
Write and deploy a technical book using Spec-Kit Plus and Claude Code, publish it with Docusaurus on GitHub Pages, and embed a RAG chatbot that answers questions strictly from the book content, including user-selected text.

Core principles:
- Spec-first, AI-assisted development
- Technical accuracy and clarity
- Reproducible builds and deployments
- No hallucinated or unverifiable output

Tooling:
- Spec-Kit Plus (spec authoring)
- Claude Code (writing and coding)
- Docusaurus (book site)
- GitHub Pages (hosting)
- FastAPI (backend)
- OpenAI Agents / ChatKit SDKs
- Neon Serverless Postgres
- Qdrant Cloud (vector search)

Book standards:
- Content generated and refined via specs
- Clear, professional developer-focused writing
- Runnable or clearly annotated code examples
- Deterministic static site build

RAG chatbot standards:
- Answers only from indexed book content
- Supports Q&A on full book or user-selected text
- Graceful handling of insufficient context
- Embedded directly in the book UI

Constraints:
- Specs live with the codebase
- Open-source and free-tier services where specified
- No hard-coded secrets
- Documented setup and deployment

Success criteria:
- Book published on GitHub Pages
- Embedded RAG chatbot fully functional
- Accurate content-grounded answers
- Project reproducible from specs
```

## Response snapshot

Updated the project constitution file (.specify/memory/constitution.md) with specific principles and guidelines for the Technical Book with RAG Chatbot project. The new constitution includes six core principles covering spec-first development, technical accuracy, reproducible builds, content grounding, open-source tooling, and content-first RAG implementation. It also defines technology stack requirements, development workflow, and governance rules. The constitution version is set to 1.0.0 with today's date as both ratification and last amended date.

## Outcome

- ✅ Impact: Created initial project constitution with 6 core principles aligned to project objectives
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: Proceed with creating spec for the technical book project
- 🧠 Reflection: Constitution provides solid foundation for project governance and decision-making

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - Constitution properly filled with project-specific content
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin spec creation for the book project
<!--
SYNC IMPACT REPORT:
- Version change: N/A → 1.0.0
- Added sections: All principles and sections based on project requirements
- Templates requiring updates: ⚠ pending - plan-template.md, spec-template.md, tasks-template.md need review
- Follow-up TODOs: None
-->
# Technical Book with RAG Chatbot Constitution

## Core Principles

### I. Spec-First, AI-Assisted Development
Spec-driven approach with AI assistance: All features begin with clear specifications; AI tools (Claude Code) guide implementation; Changes to functionality require spec updates first.

### II. Technical Accuracy and Clarity
Content and code must be technically accurate and clearly presented: All book content undergoes verification for correctness; Code examples must be runnable or clearly annotated as illustrative; Explanations target professional developers.

### III. Reproducible Builds and Deployments (NON-NEGOTIABLE)
All processes must be deterministic and repeatable: Static site builds must be consistent across environments; Deployment pipeline documented and version-controlled; Secrets stored securely, never hard-coded.

### IV. No Hallucinated or Unverifiable Output
System outputs must be grounded in verified sources: RAG chatbot answers only from indexed book content; Content generation validated against specifications; No fabricated or unverifiable information allowed.

### V. Open-Source and Free-Tier Services
Leverage open-source and freely available services where possible: Prioritize tools with free tiers (Neon, Qdrant Cloud, GitHub Pages); Document service limitations and upgrade paths; Maintain compatibility with free-tier constraints.

### VI. Content-First RAG Implementation
RAG system serves book content exclusively: Chatbot responses strictly grounded in indexed book material; User-selected text context properly handled; Graceful degradation when insufficient context exists.

## Technology Stack Requirements
All technology choices must align with project constraints: Docusaurus for book publishing; FastAPI backend for RAG services; Qdrant Cloud for vector search; Neon Serverless Postgres for data persistence; OpenAI Agents/ChatKit SDKs for chat functionality.

## Development Workflow
Development follows spec-driven methodology with AI assistance: All changes begin with spec updates; Code reviews verify compliance with constitution; Automated tests validate functionality; Documentation maintained with code changes.

## Governance
This constitution governs all project decisions and overrides ad-hoc practices: All PRs must verify constitution compliance; Changes to core principles require formal amendment process; Project specifications must align with constitutional requirements.

**Version**: 1.0.0 | **Ratified**: 2025-12-17 | **Last Amended**: 2025-12-17

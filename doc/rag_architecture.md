# Knowledge Architecture

## Overview

The system uses a seeded official-source knowledge base for mutual fund FAQs.

## Layers

1. Frontend
   - React + Vite
   - Search-first interface
   - Compact answer card with one source link

2. Backend
   - FastAPI
   - Controller layer for API routes
   - Model layer for request/response schemas
   - Service layer for policy checks and corpus lookup

3. Knowledge layer
   - Seeded corpus in `backend/data/corpus.json`
   - Source registry in `backend/data/source_list.csv` and `backend/data/source_list.md`
   - Exact field lookup plus lightweight token-overlap fallback for service questions

## Request flow

1. User sends query from frontend
2. Backend runs the policy layer for PII, advice, and performance refusals
3. Knowledge base detects scheme, fact type, or service-help intent
4. Seeded official content is returned with one source URL and last-updated information

# Architecture

## Scenario and boundaries

NimbusDesk is a fictional software product. This application supports a local engineer who needs manual-grounded answers and a repeatable way to compare chat models. The corpus is operator-controlled Markdown, not arbitrary public uploads. The API is intentionally local and unauthenticated.

## Data flow

`load_chunks` splits Markdown by headings, then applies bounded character splitting. Each chunk gets a stable content identifier, filename and section. `Store.ingest` embeds chunks through a LangChain Embeddings interface and upserts documents/vectors/metadata into a persistent Chroma collection. Stale IDs are removed after the upsert. A failed embedding call leaves the previous data untouched; a process failure during storage updates can require re-ingestion. This is not an atomic indexing service.

`Store.retrieve` embeds the query in the same space, uses cosine nearest-neighbor search, and filters by threshold. Explicit error-code queries first filter uppercase code text and bypass the similarity threshold, then require exact matches to avoid confusing AUTH-104 with another error. This restriction deliberately favors precision and can abstain on multi-code questions where no single passage contains every code.

The graph makes routing explicit: retrieval -> evidence check -> generation -> citation validation. With no qualifying passages, it abstains without calling a chat model. Live generation returns a validated JSON schema. Malformed JSON becomes a redacted error; unknown citation IDs cause abstention. A model's own insufficient-evidence decision also causes abstention.

## Comparison

The service performs a single retrieval and passes immutable-by-convention copies of those passages to each graph invocation. Models have configurable aliases and providers. The response stores model identities and separates retrieval time from model graph latency. Calls run sequentially, not concurrently. Differences in provider defaults, network conditions and rate limits remain confounders.

## Observation and privacy

Node wrappers collect operational events and durations. The service creates UUID run IDs and writes JSON via temporary-file replacement. Default disk output excludes questions, answers and source content; filenames, IDs and metrics remain. Full HTTP responses and the interactive UI necessarily include evidence. Optional external LangSmith tracing is controlled independently through its environment settings.

## Deliberate limits

- Demo vectors are lexical hash vectors, not learned semantic embeddings.
- Valid citations do not establish entailment; evaluation and manual review are still necessary.
- No multi-tenant isolation, authentication, background job queue, chat memory or production deployment is claimed.
- Runtime provider exceptions are sanitized. Operator diagnostics focus on configuration and node failure events rather than leaking raw SDK errors.
- Old collection signatures are retained. Operators should manage local data retention.
- Markdown ingestion is bounded by configured chunk size but is intended for small trusted corpora; large corpora need batching and atomic collection promotion.

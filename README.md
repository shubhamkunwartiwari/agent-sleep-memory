# Sleep-Consolidated Matrix Memory for AI Agents

A first-principles cognitive memory architecture and runnable Python simulation (`sleep_memory_sim.py`) that gives AI agents human-like, long-term memory—eliminating the storage bloat, semantic blurring, retrieval lag, and privacy leaks of flat vector databases.

---

## Why This Exists

Large Language Models (LLMs) are fundamentally stateless: every inference pass reconstructs context from scratch and forgets everything once the response finishes. Standard Retrieval-Augmented Generation (RAG) works around this by encoding past interactions into independent **1D vectors**, which introduces four bottlenecks at scale:

* **Semantic Averaging:** Compressing multi-step events into a single 1D vector blurs distinct facts together.
* **Topological Blindness:** Flat similarity scores cannot distinguish whether Fact A *caused*, *updated*, or *contradicted* Fact B.
* **Unbounded Clutter:** Without active forgetting rules, trivial noise accumulates indefinitely and degrades search accuracy.
* **Latency vs. Maintenance Trade-Off:** Deduplicating, summarizing, and reorganizing memory during a live conversation adds unacceptable response delay.

---

## System Architecture

The system splits memory operations across two decoupled planes: a **Foreground Actor Agent** (optimized for millisecond user responses) and a **Background Consolidator Sub-Agent** (optimized for offline reorganization during idle "sleep" cycles).

```text
                        ┌─────────────────────────┐
                        │      User Query         │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │  Sub-Millisecond Router │ (< 2 ms dot-product triage)
                        └─────┬─────────────┬─────┘
             Detail / Recent  │             │  Habits / Concepts
             (>85% conf.)     │             │  (>85% conf. | <85% = Parallel)
                              ▼             ▼
                 ┌──────────────────┐ ┌──────────────────────────┐
                 │    Episodic      │ │   "Memory Zip" Store     │
                 │   Scratchpad     │ │ • 2D Pattern Matrix      │
                 │  (Raw & Exact)   │ │ • 1D Recipe Vectors      │
                 └────────┬─────────┘ │ • Sparse Causal Graph    │
                          │           └────────────▲─────────────┘
                          │                        │
                 Idle Timer OR             Atomic Shadow Swap
                Buffer Watermark            (Copy-on-Write)
                          │                        │
                          ▼                        │
                 ┌─────────────────────────────────┴─────────────┐
                 │         Background Sleep Sub-Agent            │
                 │  1. Dual-Gate Forgetting (Freq/Recency + TTL) │
                 │  2. Compress Raw Logs -> 1D Recipe Vectors    │
                 │  3. Stateless Privacy Sanitizer -> Typed Slots│
                 │  4. Archive Raw Logs -> Immutable Cold Store  │
                 └───────────────────────────────────────────────┘
```

### The Six Layers

1. **Episodic Scratchpad (Short-Term Exact Buffer):** Captures raw, uncompressed interactions during live chat so high-entropy details (error codes, passwords, exact figures) are never blurred by lossy compression.
2. **"Memory Zip" (2D Pattern Matrix + Sparse 1D Recipe Vectors):** Recurring workflows and habits are stored *once* as canonical rows in a contiguous 2D **Pattern Matrix** (maximizing GPU/TPU cache locality and arithmetic intensity). Individual memories are stored as sparse **1D recipe vectors** referencing those rows (e.g., `0.8 × Pattern #12 + 0.3 × Pattern #47`), cutting storage overhead by **90–95%**.
3. **Sparse Relational Graph:** Tracks directed causal and temporal links (*causes*, *updates*, *contradicts*) using sparse adjacency lists instead of an $O(N^2)$ dense matrix, with Singular Value Decomposition (SVD) available to extract dominant reasoning pathways.
4. **Dual-Gate Forgetting & Permanent Invariants:**
   * **Gate 1 (Session Frequency + Recency):** Promotes habits seen across multiple *distinct sessions/days* while letting abandoned routines decay naturally.
   * **Gate 2 (Importance + Urgency + Explicit Time Frame):** Attaches a Time-To-Live (TTL) expiration timestamp to deadline-driven tasks so they auto-archive once expired.
   * **Permanent Invariant Override:** High-importance constraints (e.g., severe allergies) bypass decay entirely and remain pinned permanently.
5. **Sub-Millisecond Semantic Router:** Uses pre-computed embedding dot products (`< 2 ms`) to route queries to the Episodic Scratchpad or Pattern Matrix, falling back to a **parallel lookup across both stores** whenever routing confidence is below 85%.
6. **Asynchronous Sleep Consolidation & Stateless Privacy Isolation:** Runs off the critical path using Copy-on-Write shadow swapping. During sleep, a memoryless **Stateless Sanitizer Sub-Agent** strips private entities into typed slots (`[MEDICAL_HISTORY]`, `[LIFE_EVENT]`, `[FINANCIAL_RECORD]`), allowing a shared **Global Pattern Matrix** to learn generalized workflows via $k$-anonymous quorum thresholding while private bindings stay strictly local.

---

## Production Safeguards

| Failure Mode | What Goes Wrong | Architectural Mitigation |
| :--- | :--- | :--- |
| **Pointer Corruption** | Background writes shift Pattern Matrix rows mid-read. | **Copy-on-Write Shadow Swapping:** Build `Matrix_v2` in shadow memory and swap the read pointer atomically. |
| **Insomniac Agent** | High-traffic 24/7 bots never hit idle windows, overflowing buffers. | **Hybrid Watermark Triggers:** Trigger sleep on *either* an idle timeout or a scratchpad entry cap. |
| **Multi-Night "JPEG Drift"** | Repeatedly re-zipping compressed memories compounds rounding errors. | **One-Way Idempotent Compression:** Compress raw logs once, then archive raw logs to immutable cold storage. |
| **Mosaic Re-Identification** | Combining 4–5 anonymized tags uniquely fingerprints one person. | **Quorum Thresholding ($k$-anonymity):** Require $k \ge 50$ distinct users to generate a skeleton before global promotion. |
| **Proprietary Secret Leaks** | Category filters miss internal codenames or proprietary code. | **Allow-List Inversion:** Default all patterns to private; promote only verified generic templates. |

---

## Repository Structure

```text
├── README.md              # Project overview, architecture summary, and quick start
├── WHITEPAPER.md          # Full technical whitepaper and systems specification
└── sleep_memory_sim.py    # Self-contained Python simulation of all six layers
```

---

## Quick Start

Only `numpy` is required—no external API keys, vector databases, or paid models needed.

```bash
git clone [https://github.com/](https://github.com/)shubhamkunwartiwari/agent-sleep-memory.git
cd agent-sleep-memory
pip install numpy
python sleep_memory_sim.py
```

### Example Output

Running `sleep_memory_sim.py` simulates four distinct real-world memories passing through live capture, offline sleep consolidation, privacy sanitization, and sub-millisecond query routing:

```text
[Foreground] Captured raw memory M1: "I ate a sandwich for lunch today."
[Foreground] Captured raw memory M2: "By the way, I have a severe peanut allergy."
[Foreground] Captured raw memory M3: "Urgent: submit my tax filing by Friday."
[Foreground] Captured raw memory M4: "Book aisle seats for my Tokyo flight."

============================================================
🌙 [Sleep Sub-Agent] Starting offline consolidation cycle...
============================================================
  • Evaluating M1 -> FORGET (Low saliency noise (habit=0.70))
  • Evaluating M2 -> PROMOTE_PERMANENT (High importance bypass (Permanent Invariant))
  • Evaluating M3 -> EXPIRE (Deadline passed (5d > 3d TTL))
  • Evaluating M4 -> PROMOTE_HABIT (Strong habit score (2.14))
✨ [Sleep Sub-Agent] Shadow swap complete. Raw scratchpad archived to cold storage.

🌐 [Global Quorum Queue (Sanitized Skeletons)]:
   • By the way, I have a severe [MEDICAL_HISTORY].
   • Book aisle seats for my [LIFE_EVENT: LOCATION] flight.

🔍 [Router | 0.042 ms] Query: "Recommend a restaurant for my next flight"
   Route: PATTERN_MATRIX_ONLY (Pattern=100%, Episodic=0%)
   ↳ [Pattern Hit | sim=0.71] M4: Book aisle seats for my [LIFE_EVENT: LOCATION] flight.
     ↳ Private Bindings Available: {'[MEDICAL_HISTORY]': 'peanut allergy', '[LIFE_EVENT: LOCATION]': 'Tokyo'}

🔍 [Router | 0.031 ms] Query: "Fix the python bug before the deadline"
   Route: PARALLEL_BOTH_STORES (Pattern=50%, Episodic=50%)
   ↳ [Episodic] Scratchpad empty (all recent logs consolidated).
```

---

## Documentation & Wiki

* **[Full Technical Whitepaper (`WHITEPAPER.md`)](./WHITEPAPER.md)** — Complete architectural specification, hardware geometry analysis, and systems engineering design.
* **[Architecture Deep Dive](../../wiki/Architecture-Deep-Dive)** — Why aligned matrices outperform standalone vectors, how "Memory Zip" works, and how sleep consolidation operates off the critical path.
* **[Forgetting, Routing & Privacy](../../wiki/Forgetting-Routing-and-Privacy)** — Deep dive into dual-gate saliency, confidence-threshold routing, and stateless typed-slot sanitization.
* **[Production Failure Modes](../../wiki/Production-Failure-Modes)** — Detailed breakdown of distributed-systems edge cases and their architectural patches.

---

## Acknowledgments

The core architectural concepts and system design in this repository were developed from first principles through an interactive design and stress-testing session with **Gemini (Google)**, which assisted in formalizing the six-layer model, identifying production failure modes, authoring the whitepaper and documentation, and writing the Python reference simulation.

---

## License

MIT License — free to use, modify, and build upon.

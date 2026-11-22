# Sleep-Consolidated Matrix Memory for AI Agents

A first-principles cognitive memory architecture and runnable Python simulation (`sleep_memory_sim.py`) that explores how AI agents could keep long-term memory while avoiding some storage, blurring, latency, and privacy problems of flat vector databases.

> **Status:** This is a design proposal and a small prototype. The storage-reduction figures in the design docs are estimates, not benchmark results. See [Limitations](#limitations).

---

## Why This Exists

Large Language Models (LLMs) are stateless: every inference pass reconstructs context from scratch and forgets everything once the response finishes. Standard Retrieval-Augmented Generation (RAG) works around this by encoding past interactions into independent **1D vectors**, which introduces four bottlenecks at scale:

- **Semantic Averaging:** Compressing multi-step events into a single 1D vector blurs distinct facts together.
- **Topological Blindness:** Flat similarity scores cannot distinguish whether Fact A *caused*, *updated*, or *contradicted* Fact B.
- **Unbounded Clutter:** Without active forgetting rules, trivial noise accumulates indefinitely and degrades search accuracy.
- **Latency vs. Maintenance Trade-Off:** Deduplicating, summarizing, and reorganizing memory during a live conversation adds unacceptable response delay.

---

## System Architecture

The system splits memory operations across two decoupled planes: a **Foreground Actor Agent** (optimized for millisecond user responses) and a **Background Consolidator Sub-Agent** (optimized for offline reorganization during idle "sleep" cycles).

```
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

1. **Episodic Scratchpad (Short-Term Exact Buffer):** Captures raw, uncompressed interactions during live chat so high-entropy details (error codes, passwords, exact figures) are not blurred by lossy compression.
2. **"Memory Zip" (2D Pattern Matrix + Sparse 1D Recipe Vectors):** Recurring workflows and habits are stored *once* as canonical rows in a 2D **Pattern Matrix**. Individual memories are stored as sparse **1D recipe vectors** referencing those rows (e.g., `0.8 × Pattern #12 + 0.3 × Pattern #47`). The whitepaper estimates a 90–95% storage reduction for this design; this is not measured by the simulation.
3. **Sparse Relational Graph:** Tracks directed causal and temporal links (*causes*, *updates*, *contradicts*) using sparse adjacency lists instead of an $O(N^2)$ dense matrix. The simulation only records a simple `exemplifies` link. Singular Value Decomposition (SVD) is described in the whitepaper but not implemented.
4. **Dual-Gate Forgetting & Permanent Invariants:**
   - **Gate 1 (Session Frequency + Recency):** Promotes habits seen across multiple *distinct sessions/days* while letting abandoned routines decay.
   - **Gate 2 (Importance + Urgency + Explicit Time Frame):** Attaches a Time-To-Live (TTL) to deadline-driven tasks so they expire once the deadline passes.
   - **Permanent Invariant Override:** High-importance constraints (e.g., severe allergies) bypass decay and remain pinned.
5. **Sub-Millisecond Semantic Router:** Uses embedding dot products to route queries to the Episodic Scratchpad or Pattern Matrix, falling back to a **parallel lookup across both stores** whenever routing confidence is 85% or lower for a single store.
6. **Asynchronous Sleep Consolidation & Stateless Privacy Isolation:** Runs off the critical path using copy-on-write shadow swapping. A memoryless **Stateless Sanitizer** replaces private details with typed slots (`[MEDICAL_HISTORY]`, `[LIFE_EVENT]`, `[FINANCIAL_RECORD]`) before anything is queued for a shared **Global Pattern Matrix**. The whitepaper also describes k-anonymous quorum thresholding and allow-list sanitization; the simulation implements neither (see [Limitations](#limitations)).

---

## Production Safeguards

| Failure Mode                 | What Goes Wrong                                                      | Proposed Mitigation                                                                                                        |
| ---------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Pointer Corruption**       | Background writes shift Pattern Matrix rows mid-read.                | **Copy-on-Write Shadow Swapping:** Build `Matrix_v2` in shadow memory and swap the read pointer atomically.                |
| **Insomniac Agent**          | High-traffic 24/7 bots never hit idle windows, overflowing buffers.  | **Hybrid Watermark Triggers:** Trigger sleep on *either* an idle timeout or a scratchpad entry cap.                        |
| **Multi-Night "JPEG Drift"** | Repeatedly re-zipping compressed memories compounds rounding errors. | **One-Way Idempotent Compression:** Compress raw logs once, then archive raw logs to immutable cold storage.               |
| **Mosaic Re-Identification** | Combining 4–5 anonymized tags uniquely fingerprints one person.      | **Quorum Thresholding ($k$-anonymity):** Require $k \ge 50$ distinct users to generate a skeleton before global promotion. |
| **Proprietary Secret Leaks** | Category filters miss internal codenames or proprietary code.        | **Allow-List Inversion:** Default all patterns to private; promote only verified generic templates.                        |

---

## Repository Structure

```
├── README.md              # Project overview, architecture summary, and quick start
├── WHITEPAPER.md          # Full technical whitepaper and systems specification
├── LICENSE                # MIT license
└── sleep_memory_sim.py    # Self-contained Python simulation of the six layers
```

---

## Quick Start

Only `numpy` is required. No API keys, vector databases, or paid models are needed.

```bash
git clone https://github.com/shubhamkunwartiwari/agent-sleep-memory.git
cd agent-sleep-memory
pip install numpy
python sleep_memory_sim.py
```

### Example Output

Running `sleep_memory_sim.py` simulates four memories passing through live capture, offline sleep consolidation, privacy sanitization, and query routing. Timing values vary by machine.

```
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

🔍 [Router | 0.116 ms] Query: "Recommend a restaurant for my next flight"
   Route: PATTERN_MATRIX_ONLY (Pattern=100%, Episodic=0%)
   ↳ [Pattern Hit | sim=0.71] M4: Book aisle seats for my [LIFE_EVENT: LOCATION] flight.
     ↳ Private Bindings Available: {'[MEDICAL_HISTORY]': 'peanut allergy', '[LIFE_EVENT: LOCATION]': 'Tokyo'}

🔍 [Router | 0.015 ms] Query: "Fix the python bug before the deadline"
   Route: PARALLEL_BOTH_STORES (Pattern=29%, Episodic=71%)
     ↳ Private Bindings Available: {'[MEDICAL_HISTORY]': 'peanut allergy', '[LIFE_EVENT: LOCATION]': 'Tokyo'}
   ↳ [Episodic] Scratchpad empty (all recent logs consolidated).
```

---

## Limitations

This is a prototype for exploring the design, not a production memory system.

- **Embeddings are keyword flags**, not learned embeddings. The vectors for M1–M4 are hand-assigned in `__main__`.
- **Privacy sanitization is a block-list of regexes.** The allow-list approach described in the whitepaper is not implemented.
- **No k-anonymous quorum.** Sanitized skeletons go straight into `global_quorum_queue`.
- **No storage benchmark.** The 90–95% reduction is a design estimate.
- **SVD and causal-graph reasoning** are described but not implemented.

---

## Documentation & Wiki

- **[Full Technical Whitepaper (`WHITEPAPER.md`)](https://github.com/shubhamkunwartiwari/agent-sleep-memory/blob/main/WHITEPAPER.md)**: Complete architectural specification, hardware geometry analysis, and systems engineering design.
- **[Architecture Deep Dive](https://github.com/shubhamkunwartiwari/agent-sleep-memory/wiki/Architecture%E2%80%90Deep%E2%80%90Dive)**: Why aligned matrices outperform standalone vectors, how "Memory Zip" works, and how sleep consolidation operates off the critical path.
- **[Forgetting, Routing & Privacy](https://github.com/shubhamkunwartiwari/agent-sleep-memory/wiki/Forgetting%E2%80%90Routing%E2%80%90and%E2%80%90Privacy)**: Dual-gate saliency, confidence-threshold routing, and stateless typed-slot sanitization.
- **[Production Failure Modes](https://github.com/shubhamkunwartiwari/agent-sleep-memory/wiki/Production%E2%80%90Failure%E2%80%90Modes)**: Distributed-systems edge cases and their proposed patches.

---

## Acknowledgments

The core architectural concepts were developed from first principles through an interactive design and stress-testing session with **Gemini (Google)**, which helped formalize the six-layer model, identify production failure modes, draft the whitepaper and documentation, and write the Python reference simulation.

---

## License

MIT License. Free to use, modify, and build upon.

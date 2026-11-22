# Designing Human-Like Memory for AI Agents: From Stateless Models to Compressed, Privacy-Safe Sleep Consolidation

> **Scope of this document.** This paper describes a design. The companion simulation (`sleep_memory_sim.py`) implements a subset of it. Each section marks what the prototype does and what remains proposed. Figures labeled *estimate* are not measured.

## Abstract
To a casual user, modern artificial intelligence feels like it remembers everything said earlier in a conversation. Under the hood, however, a Large Language Model (LLM) has no built-in memory at all. Every time you send a message, the model re-reads the entire conversation from scratch. Current AI agents work around this by storing past conversations as flat lists of numbers called **1D vectors**. Flat vectors are fast for simple lookups, but they blur complex stories together, fail to track how facts connect, and grow costly as memories accumulate.

This paper proposes a memory design built from the ground up. It stacks vectors into **2D matrices**, maps relationships with **sparse graphs**, and compresses recurring habits with a **"Memory Zip" codebook**. It adds a **background "sleep" sub-agent** that reorganizes memories offline, filters noise with time-aware forgetting rules, routes lookups quickly, and separates private user details from shared knowledge.

---

## 1. The Illusion of AI Memory: How Stateless Models Work

If you ask an AI a question, get an answer, and ask a follow-up, it seems obvious that the AI remembered what you just said. In reality, **an AI model is stateless**. Once it finishes generating a reply, it retains nothing.

The illusion of memory is created by the chat application wrapping the model, working with four internal mechanisms:

* **The Context Window (The Working Desk):** Every time you send a message, the software gathers the system instructions, past messages, past replies, and your new message, and hands the whole bundle to the model. Everything the model can see in that moment is its **context window**.
* **Tokens and Embeddings (Translating Words into Coordinates):** The software splits text into **tokens** (words or parts of words) and converts each token into a list of numbers called an **embedding**. An embedding is a point in a high-dimensional map of meaning, so related words sit near each other.
* **Attention (Connecting the Dots):** Inside the model, **attention** scores how much each earlier token matters for the current one. In a message such as *"Fix the bug in it,"* attention links *"it"* to *"the Python script"* mentioned earlier.
* **The KV Cache (Saved Work for Speed):** To avoid recomputing past tokens for every new word, the server stores intermediate results called the **Key-Value (KV) Cache**, so only the newest tokens need processing.

**The Bottleneck:** The context window has a fixed size. Once a conversation exceeds it, older messages drop out. Long-term memory across sessions therefore needs an external store, and how that store is organized matters.

---

## 2. Vectors vs. Matrices: Why Shape Matters to AI Hardware

A sentence converted into coordinates is a **1D vector**, for example a row of 1,024 numbers. If you save 10,000 past memories, you can store them as **10,000 separate 1D vectors**, or stack them into one **2D matrix** of shape $10{,}000 \times 1{,}024$. Both hold the same numbers.

Modern GPUs process matrices more efficiently than many separate vectors, for two reasons:

1. **Memory locality:** Stacking vectors into one block means the processor reads contiguous memory instead of scattered addresses. Neighboring rows tend to be loaded into cache together.
2. **Arithmetic intensity:** Comparing one query against one vector at a time leaves the arithmetic units waiting on memory. A matrix lets one loaded block be multiplied against many targets at once, so more math is done per byte moved.

These are general hardware properties. The prototype stores its pattern codebook as a NumPy matrix to use them, but it does not benchmark the speedup.

### Where AI Uses Each Structure Today

| Memory Layer | Real-World Analogy | Structure Used | How It Works |
| :--- | :--- | :--- | :--- |
| **Long-Term Knowledge** (Model Weights) | Everything you learned in school | **Large 2D matrices** | Facts and language patterns are encoded in weight matrices during training. |
| **Short-Term Memory** (Context / KV Cache) | What is currently on your desk | **Matrix that grows by one row per token** | Each new token adds a row to a growing $N \times d$ matrix. |
| **External Memory** (Vector Databases) | A searchable filing cabinet | **Standalone 1D vectors** | Each document or chat chunk is stored as its own vector and searched by distance. |

*(Note: **State Space Models** try to avoid growing matrices by folding each new token into a fixed $d \times d$ state. Memory stays constant in size, but exact word-for-word detail can be lost.)*

---

## 3. Beyond Flat Vectors: Sub-Matrices and Connection Maps

Standard agents store each past memory as a single 1D vector in a database. This causes two problems:

* **Semantic Averaging:** A memory that contains several ideas, such as *"We tried PostgreSQL for the analytics app, it choked on heavy writes, so we switched to ClickHouse,"* is compressed into one vector that averages those ideas. A later search for *"why did we leave PostgreSQL?"* becomes fuzzy.
* **Topological Blindness:** A vector database measures similarity only. It cannot tell whether Memory A **caused**, **updated**, or **contradicted** Memory B.

### The Proposed Upgrade: 2D Sub-Matrices + A Connection Structure
Each memory could be a small **2D sub-matrix** ($\mathbf{M}_i$), with rows for different angles, actors, or steps of one event. A separate **connection structure** would record links between memories. Written as a dense matrix, this is an *adjacency matrix*.

*Prototype status:* The simulation stores one recipe vector per memory and no sub-matrices. Its graph is a sparse adjacency list with one link type, `exemplifies`. Causal (`causes`, `updates`, `contradicts`) links are not implemented.

### Solving the $N^2$ Explosion with Sparse Graphs
A dense connection matrix grows quadratically, $O(N^2)$. At 10 memories it has $10 \times 10 = 100$ cells. At 1,000,000 memories it would have $10^{12}$ cells, which is too large to hold in memory on ordinary hardware.

Most cells would be zero, because most memories are not directly linked. A **sparse graph** stores only the links that exist, so storage scales with the number of links.

**Singular Value Decomposition (SVD)** could compress a connection map into smaller factors to surface dominant patterns. This is *proposed only*; the prototype does not use SVD.

---

## 4. "Memory Zip": Compressing Experience with Codebooks and Recipe Vectors

Compression tools such as zip save each repeated pattern once and replace repeats with short references. The same idea can apply to agent memory:

1. **The Pattern Matrix (Shared Codebook):** Recurring templates, such as a preferred email format or a standard debugging routine, are stored once as rows of a **Pattern Matrix** ($\mathbf{C}$).
2. **1D Recipe Vectors (Lightweight Pointers):** Each memory is stored as a sparse 1D vector of weights over the codebook rows:

   $$
   \text{Memory}_i \approx 0.8 \times [\text{Pattern } 12] + 0.3 \times [\text{Pattern } 47]
   $$

   Because most weights are zero, a recipe takes less space than a full copy of the memory. The storage saving depends on how many rows each memory references and how sparse its recipe is. The prototype does not measure it. **Any specific percentage would be an estimate, not a result.**

*Prototype status:* The codebook is three hand-written patterns (coding, travel, health), not patterns extracted from data. Recipe weights are the dot products of each memory's vector with the codebook, zeroed below 0.2.

### The "JPEG Blur" Catch and the Brain's Solution
Storing memories as weights over a shared codebook is **lossy**, like saving a photo as a JPEG. It captures common routines well but blurs rare, one-off details such as a Wi-Fi password, a confirmation code, or an unusual exception.

The brain handles this trade-off with two systems: the **hippocampus** records recent events in detail, and the **neocortex** gradually extracts general knowledge. The design mirrors that split:
* **The Episodic Scratchpad:** A small, **uncompressed** buffer that keeps recent interactions verbatim.
* **The Zipped Pattern Matrix:** A **compressed** long-term store for habits and general knowledge.

---

## 5. Sleep-Time Consolidation: The Background Sub-Agent

Extracting patterns, updating the codebook, and resolving conflicts takes real computation. Running it during a live chat would add visible lag, so the design separates two workers:

* **The Foreground Actor Agent:** Handles the conversation. It reads the existing codebook and appends raw notes to the Episodic Scratchpad without organizing them.
* **The Background Consolidator Sub-Agent:** Runs during idle "sleep" cycles. It reviews the raw scratchpad, promotes or drops candidates, sanitizes them, compresses promoted memories into recipe vectors, and archives the raw log.

*Prototype status:* The simulation runs one sleep cycle when called. It does not detect idle time or run in parallel with the foreground agent. Contradiction resolution (for example, updating an address) is not implemented.

### Failure Modes and Proposed Fixes
Running a background process alongside a live agent creates these risks:

| Failure Mode | What Goes Wrong in Practice | Proposed Fix |
| :--- | :--- | :--- |
| **Pointer Corruption (Race Condition)** | If the sub-agent rewrites codebook rows while the foreground agent is reading, existing recipe vectors can point to the wrong rows. | **Copy-on-Write Shadow Swapping:** Build the new codebook off to the side and switch the read pointer in one step. *The prototype does this by deep-copying the recipe store and graph, then replacing them in a single assignment. It does not swap a live codebook.* |
| **The "Insomniac" Agent** | A 24/7 bot may never go idle, so the scratchpad keeps growing. | **Watermark Triggers:** Run consolidation on whichever comes first, an idle timeout or a scratchpad size cap. *Proposed only; the prototype has no triggers.* |
| **Multi-Night "JPEG Drift"** | Re-compressing already compressed memories repeatedly compounds rounding error. | **One-Way Compression:** Compress each raw entry once, then archive the raw log. *The prototype compresses each entry once and writes all raw entries to cold storage.* |

---

## 6. The Art of Forgetting: Dual-Gate Saliency and Time-Bounded Urgency

Without forgetting, the codebook fills with clutter, such as what the user ate for lunch months ago. During each sleep cycle, every candidate passes through a **Dual-Gate Saliency Filter**.

### Gate 1: Session Frequency + Recency (Habit and Decay)
* **Why both:** Frequency alone keeps abandoned habits alive forever. Recency alone lets a single recent mention look important. The combined score favors patterns that are both repeated and recent.
* **Anti-spam:** Frequency is counted across **distinct sessions or days**, not raw mentions. Fifteen mentions of a bug in one 10-minute session count as one session.

The prototype computes `habit_score = 0.4 × session_count + 0.6 × recency_score` and promotes memories at 1.2 or more. The caller supplies `session_count` and `recency_score`; the prototype does not compute them from timestamps.

### Gate 2: Importance + Urgency + Time Frame (TTL)
* **Deadlines should not wait** to build frequency. A task with a deadline is promoted immediately.
* **TTL binding:** A deadline attaches an expiration. Once it passes, the memory is expired.

*Prototype status:* A memory with a TTL is promoted while urgency is at least 0.7 and expired after the TTL. Expired and forgotten memories are not compressed into recipe vectors, but their raw entries are still written to cold storage with all other scratchpad entries.

### The "Silent Critical Constraint" Override
A life-critical fact, such as *"I have a severe peanut allergy,"* may be mentioned once with no deadline. Its score is low, so it would otherwise be forgotten. Memories with importance at or above 0.9 bypass decay as **Permanent Invariants**. The prototype implements this threshold.

---

## 7. Retrieval Routing: Looking in the Right Place Fast

With two stores, the Episodic Scratchpad and the Pattern Matrix, searching both on every query wastes work. A router decides where to look.

1. **Store-Level Routing:** Recent, detail-oriented questions go to the Episodic Scratchpad. Conceptual or preference questions go to the Pattern Matrix.
2. **Partition Pre-Filtering (proposed):** Patterns could be grouped into topic clusters so the router searches only the matching cluster. *Not implemented in the prototype.*

### Router Speed and Fallback
* **Dot-product router:** The prototype compares the query vector against two precomputed route vectors. It does this with a single dot product and no LLM call, so the decision is cheap. On the development machine the router took roughly 0.01–0.1 ms per query. This is a single measurement, not a benchmark.
* **Parallel fallback:** If neither store reaches 85% confidence, the router searches both and merges results. The prototype does this, and the example output shows it. The confidence is computed as the share of each route's score in the total, so it measures relative routing preference, not retrieval accuracy.

---

## 8. Scaling to Many Users: Privacy Isolation

Users should not each have to teach their agent basic skills. A design could keep one **Global Pattern Matrix** shared by all users and a **Personal Pattern Matrix** per user. The risk is that private facts reach the global store.

### Sanitization and Typed Slots
Before a candidate reaches the global store, a **Sanitizer** replaces personal details with typed slots such as `[MEDICAL_HISTORY]`, `[LIFE_EVENT]`, or `[FINANCIAL_RECORD]`:

* **Global store (skeleton):** *"When `[LIFE_EVENT: relocation]` occurs alongside `[MEDICAL_HISTORY]`, prioritize local healthcare transfer checklists."*
* **Personal store (bindings):** `[LIFE_EVENT] = "moving to Bengaluru"`, `[MEDICAL_HISTORY] = "shellfish allergy"`.

The prototype's sanitizer is a **block-list of regular expressions** covering three example categories (medical allergies, a few cities, a tax filing). It holds no state between calls. This is a demonstration only. A real deployment would need a far broader detector.

### Proposed Protections Beyond the Prototype
* **Mosaic (fingerprint) attacks:** Several individually generic tags together can identify one person. A proposed defense is **quorum thresholding** (k-anonymity): promote a skeleton only after at least $k$ distinct users produce the same one, with $k = 50$ as an example value. *Not implemented. The prototype appends sanitized skeletons directly to a queue.*
* **Company secrets:** A block-list cannot catch unknown codenames or proprietary code. The proposed alternative is an **allow-list**: keep everything private by default and share only patterns that match verified, generic templates. *Not implemented. The prototype uses a block-list.*

---

## 9. Putting It All Together: Life of a Memory

Here is how the design would handle one interaction, using the prototype's behavior where it exists:

1. **Live conversation:** You say, *"Remind me to renew my passport before my Tokyo flight next month, and remember I prefer aisle seats."* The foreground agent answers and appends the raw message to the Episodic Scratchpad.
2. **Sleep cycle:** The sub-agent evaluates each candidate. A deadline with urgency gets a TTL. A preference with high importance becomes a permanent invariant. *The prototype's example handles deadlines and importance this way.*
3. **Compression and privacy split:** Promoted memories become recipe vectors. The sanitizer replaces private details with typed slots before anything is queued for the global store. The raw log is archived.
4. **Later recall:** A later query such as *"Book me a flight to London"* routes to the codebook, and private bindings are looked up for the user. *The prototype shows the routing and binding lookup. Cross-checking active deadlines is not implemented.*

### Architectural Comparison

| Feature | Standard AI Chat (Context Only) | Flat 1D Vector Database (Standard RAG) | Proposed Sleep-Consolidated Matrix Design |
| :--- | :--- | :--- | :--- |
| **Memory Lifespan** | Lost when the context window fills | Persists, but accumulates clutter | Persists, with decay and deadline expiration (prototype: yes) |
| **Detail vs. Big Picture** | Exact recall until the window overflows | Blurs multi-step stories into one average | Keeps an exact scratchpad and a compressed codebook (prototype: yes) |
| **Understanding Connections** | Only within the active window | Measures similarity only | Sparse links (prototype: one link type only) |
| **Storage Cost** | Grows with conversation length | Grows linearly with memory count | Recipe vectors may reduce storage; *not measured* |
| **Contradiction Cleanup** | Manual or absent | Old and new facts can conflict at search time | Proposed; not implemented |
| **Multi-User Privacy** | Isolated per session | Separate stores, no shared learning | Block-list sanitization in prototype; allow-list and quorum proposed |

The design aims to separate fast interaction from slower offline consolidation, keep exact details when they matter, compress repeated patterns, let irrelevant information fade, and protect private information while still learning from many users. The simulation demonstrates the core loop. The remaining pieces are proposals for future work.

---

## Acknowledgments

This architecture was designed from first principles and refined through an interactive engineering and stress-testing collaboration with **Gemini (Google)**, which contributed to formalizing the six-layer model, identifying production failure modes, drafting this whitepaper, and implementing the reference simulation (`sleep_memory_sim.py`).

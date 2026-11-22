"""
Six-Layer Sleep-Consolidated Matrix Memory Simulation
=====================================================

A lightweight, dependency-minimal prototype demonstrating:

1. Uncompressed Episodic Scratchpad (exact short-term recall)
2. Pattern Matrix Codebook + Sparse 1D Recipe Vectors ("Memory Zip")
3. Sparse Relational Graph (causal/conceptual links without O(N^2) bloat)
4. Dual-Gate Forgetting (Session Frequency/Recency + Time-Bounded Urgency + Permanent Invariants)
5. Sub-Millisecond Semantic Router (with confidence-threshold parallel fallback)
6. Asynchronous Sleep Sub-Agent (Copy-on-Write Shadow Swapping + Stateless Privacy Sanitizer)

Known simplifications (this is a demo, not a production system):
- Embeddings are keyword flags over a fixed concept list, not learned embeddings.
  The vectors passed to RawMemory in __main__ are hand-assigned.
- The privacy sanitizer is a block-list of regexes. The whitepaper's allow-list
  design is NOT implemented here.
- The global queue does NOT enforce k-anonymous quorum thresholding. Skeletons
  are appended directly.

Requirements:
    pip install numpy
"""

import copy
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Helper: Lightweight Deterministic Embedder (No API Key Required)
# ---------------------------------------------------------------------------

CONCEPT_AXES = [
    "coding", "python", "bug", "travel", "flight",
    "medical", "allergy", "deadline", "tax", "lunch",
]


def embed_text(text: str) -> np.ndarray:
    """Maps text to a normalized 1D vector across our concept dimensions.

    Each concept present in the text sets its axis to 1.0. If no concept
    is found, a small fallback weight on the first axis keeps the vector
    non-zero so it can be normalized.
    """
    vec = np.zeros(len(CONCEPT_AXES), dtype=np.float32)
    lower = text.lower()
    for i, word in enumerate(CONCEPT_AXES):
        if word in lower:
            vec[i] = 1.0
    if np.linalg.norm(vec) == 0:
        vec[0] = 0.1
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RawMemory:
    id: str
    text: str
    vector: np.ndarray
    session_count: int = 1        # Distinct sessions/days seen (anti-spam)
    recency_score: float = 1.0    # 1.0 = brand new, decays over time
    importance: float = 0.5       # 0.0 to 1.0 (>= 0.9 triggers Permanent Invariant)
    urgency: float = 0.0          # 0.0 to 1.0
    ttl_days: Optional[int] = None  # Explicit time frame (Time-To-Live)
    days_elapsed: int = 0


@dataclass
class CompressedMemory:
    id: str
    recipe_vector: np.ndarray     # Sparse 1D pointer weights into Pattern Matrix
    summary_tag: str
    is_permanent: bool = False
    ttl_remaining: Optional[int] = None


# ---------------------------------------------------------------------------
# Layer 6: Stateless Privacy Sanitizer
# ---------------------------------------------------------------------------

class StatelessPrivacySanitizer:
    """
    Runs in a memoryless sandbox. Strips private details into typed slots
    so the Global Matrix learns workflows without leaking personal data.
    """

    PRIVATE_PATTERNS = {
        r"peanut allergy|shellfish allergy": "[MEDICAL_HISTORY]",
        r"tokyo|london|bengaluru": "[LIFE_EVENT: LOCATION]",
        r"tax filing": "[FINANCIAL_RECORD]",
    }

    @staticmethod
    def sanitize(text: str) -> Tuple[str, Dict[str, str]]:
        """Returns (global_skeleton, private_bindings). Holds zero state."""
        skeleton = text
        private_bindings: Dict[str, str] = {}
        for pattern, slot_tag in StatelessPrivacySanitizer.PRIVATE_PATTERNS.items():
            match = re.search(pattern, skeleton, flags=re.IGNORECASE)
            if match:
                private_bindings[slot_tag] = match.group(0)
                skeleton = re.sub(pattern, slot_tag, skeleton, flags=re.IGNORECASE)
        return skeleton, private_bindings


# ---------------------------------------------------------------------------
# Core Memory Engine (Foreground + Background Sleep Consolidation)
# ---------------------------------------------------------------------------

class CognitiveMemoryAgent:

    def __init__(self, dim: int = len(CONCEPT_AXES)):
        self.dim = dim

        # Layer 1: Uncompressed Episodic Scratchpad (exact recent facts)
        self.episodic_scratchpad: List[RawMemory] = []
        self.cold_storage: List[RawMemory] = []

        # Layer 2: Personal Pattern Matrix (Codebook) + 1D Recipe Vectors
        # Starts with 3 base canonical patterns: Coding, Travel, Medical
        self.pattern_matrix: np.ndarray = np.vstack([
            embed_text("coding python bug"),
            embed_text("travel flight"),
            embed_text("medical allergy"),
        ])
        self.pattern_labels: List[str] = ["Coding_Workflow", "Travel_Prefs", "Health_Constraints"]
        self.recipe_store: List[CompressedMemory] = []

        # Layer 3: Sparse Relational Graph (adjacency list, not O(N^2) matrix)
        self.sparse_graph: Dict[str, List[Tuple[str, str]]] = {}

        # Layer 6: Personal Private Bindings & Global Queue (no quorum check yet)
        self.personal_bindings: Dict[str, str] = {}
        self.global_quorum_queue: List[str] = []

        # Layer 5: Sub-Millisecond Router Prototypes
        self.route_episodic = embed_text("bug deadline tax lunch")
        self.route_pattern = embed_text("coding python travel flight medical allergy")

    # --- Foreground Action: Fast Capture ---

    def observe(self, mem: RawMemory) -> None:
        """Foreground agent logs raw memory instantly with zero compression lag."""
        self.episodic_scratchpad.append(mem)
        print(f"[Foreground] Captured raw memory {mem.id}: \"{mem.text}\"")

    # --- Layer 4: Dual-Gate Saliency & Forgetting Filter ---

    def _evaluate_saliency(self, mem: RawMemory) -> Tuple[str, str]:
        """Decides whether a raw memory is PROMOTED, EXPIRED, or FORGOTTEN."""

        # Override: Silent Critical Constraint (Permanent Invariant)
        if mem.importance >= 0.9:
            return "PROMOTE_PERMANENT", "High importance bypass (Permanent Invariant)"

        # Gate 2: Importance + Urgency + Time Frame (TTL)
        if mem.ttl_days is not None:
            if mem.days_elapsed > mem.ttl_days:
                return "EXPIRE", f"Deadline passed ({mem.days_elapsed}d > {mem.ttl_days}d TTL)"
            if mem.urgency >= 0.7:
                return "PROMOTE_TTL", f"Urgent active deadline ({mem.ttl_days - mem.days_elapsed}d left)"

        # Gate 1: Session Frequency + Recency (Habit & Decay Gate)
        habit_score = (mem.session_count * 0.4) + (mem.recency_score * 0.6)
        if habit_score >= 1.2:
            return "PROMOTE_HABIT", f"Strong habit score ({habit_score:.2f})"

        return "FORGET", f"Low saliency noise (habit={habit_score:.2f})"

    # --- Layer 5: Background Sleep-Time Consolidation ---

    def run_sleep_cycle(self) -> None:
        """
        Offline sub-agent zips raw memories into 1D recipe vectors, prunes noise,
        sanitizes for global learning, and swaps matrices atomically.
        """
        print("\n" + "=" * 60)
        print("🌙 [Sleep Sub-Agent] Starting offline consolidation cycle...")
        print("=" * 60)

        # Copy-on-Write (Shadow Swapping): work on shadow copies so live reads never break
        shadow_recipes = copy.deepcopy(self.recipe_store)
        shadow_graph = copy.deepcopy(self.sparse_graph)

        for mem in self.episodic_scratchpad:
            decision, reason = self._evaluate_saliency(mem)
            print(f"  • Evaluating {mem.id} -> {decision} ({reason})")

            if decision.startswith("PROMOTE"):
                # 1. "Memory Zip": project raw vector onto Pattern Matrix to get 1D recipe.
                #    Weights = dot product against codebook rows, sparsified.
                raw_weights = self.pattern_matrix @ mem.vector
                sparse_recipe = np.where(raw_weights > 0.2, raw_weights, 0.0)

                # 2. Stateless Privacy Sanitization
                skeleton, bindings = StatelessPrivacySanitizer.sanitize(mem.text)
                self.personal_bindings.update(bindings)
                if bindings:
                    self.global_quorum_queue.append(skeleton)

                # 3. Store compressed 1D recipe
                ttl_left = (mem.ttl_days - mem.days_elapsed) if mem.ttl_days is not None else None
                compressed = CompressedMemory(
                    id=mem.id,
                    recipe_vector=sparse_recipe,
                    summary_tag=skeleton,
                    is_permanent=(decision == "PROMOTE_PERMANENT"),
                    ttl_remaining=ttl_left,
                )
                shadow_recipes.append(compressed)

                # 4. Update Sparse Relational Graph (link to strongest matching pattern)
                top_pattern = self.pattern_labels[int(np.argmax(raw_weights))]
                shadow_graph.setdefault(top_pattern, []).append((mem.id, "exemplifies"))

        # One-Way Consolidation: archive every raw scratchpad entry to immutable cold storage
        self.cold_storage.extend(self.episodic_scratchpad)

        # Atomic Shadow Swap
        self.recipe_store = shadow_recipes
        self.sparse_graph = shadow_graph
        self.episodic_scratchpad.clear()
        print("✨ [Sleep Sub-Agent] Shadow swap complete. Raw scratchpad archived to cold storage.\n")

    # --- Layer 5: Sub-Millisecond Semantic Router ---

    def retrieve(self, query: str) -> None:
        t0 = time.perf_counter()
        q_vec = embed_text(query)

        # Fast dot-product routing
        score_ep = float(np.dot(q_vec, self.route_episodic))
        score_pat = float(np.dot(q_vec, self.route_pattern))
        total = score_ep + score_pat + 1e-8
        conf_ep, conf_pat = score_ep / total, score_pat / total
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"🔍 [Router | {elapsed_ms:.3f} ms] Query: \"{query}\"")

        # Confidence-Threshold Fallback (>85% single store, else parallel)
        if conf_pat > 0.85:
            route = "PATTERN_MATRIX_ONLY"
        elif conf_ep > 0.85:
            route = "EPISODIC_SCRATCHPAD_ONLY"
        else:
            route = "PARALLEL_BOTH_STORES"

        print(f"   Route: {route} (Pattern={conf_pat:.0%}, Episodic={conf_ep:.0%})")

        if route in ("PATTERN_MATRIX_ONLY", "PARALLEL_BOTH_STORES"):
            for comp in self.recipe_store:
                # Reconstruct approximate vector from 1D recipe * Pattern Matrix
                reconstructed = comp.recipe_vector @ self.pattern_matrix
                sim = float(np.dot(q_vec, reconstructed))
                if sim > 0.3:
                    print(f"   ↳ [Pattern Hit | sim={sim:.2f}] {comp.id}: {comp.summary_tag}")
            if self.personal_bindings:
                print(f"     ↳ Private Bindings Available: {self.personal_bindings}")

        if route in ("EPISODIC_SCRATCHPAD_ONLY", "PARALLEL_BOTH_STORES"):
            if not self.episodic_scratchpad:
                print("   ↳ [Episodic] Scratchpad empty (all recent logs consolidated).")
            for mem in self.episodic_scratchpad:
                sim = float(np.dot(q_vec, mem.vector))
                if sim > 0.3:
                    print(f"   ↳ [Episodic Hit | sim={sim:.2f}] {mem.id}: {mem.text}")

        print()


# ---------------------------------------------------------------------------
# Run the End-to-End Simulation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = CognitiveMemoryAgent()

    # 1. Foreground agent captures 4 very different real-world memories
    agent.observe(RawMemory(
        id="M1", text="I ate a sandwich for lunch today.",
        vector=embed_text("lunch"), session_count=1, recency_score=0.5, importance=0.1,
    ))
    agent.observe(RawMemory(
        id="M2", text="By the way, I have a severe peanut allergy.",
        vector=embed_text("medical allergy"), session_count=1, recency_score=0.2, importance=0.98,
    ))
    agent.observe(RawMemory(
        id="M3", text="Urgent: submit my tax filing by Friday.",
        vector=embed_text("deadline tax"), session_count=1, urgency=0.95, ttl_days=3, days_elapsed=5,
    ))
    agent.observe(RawMemory(
        id="M4", text="Book aisle seats for my Tokyo flight.",
        vector=embed_text("travel flight"), session_count=4, recency_score=0.9, importance=0.7,
    ))

    # 2. Trigger offline sleep consolidation
    agent.run_sleep_cycle()

    # 3. Inspect what made it into the Global Queue (privacy check)
    print("🌐 [Global Quorum Queue (Sanitized Skeletons)]:")
    for skel in agent.global_quorum_queue:
        print(f"   • {skel}")
    print()

    # 4. Test live retrieval routing
    agent.retrieve("Recommend a restaurant for my next flight")
    agent.retrieve("Fix the python bug before the deadline")

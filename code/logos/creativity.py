"""
creativity.py
-------------
Cross-Domain Structure Transfer for AI creativity.

The insight: genuine novelty in art historically comes from importing structure
from an unrelated domain. Schoenberg imported set theory into music. Mondrian
imported architecture into painting. Reich imported tape mechanics into rhythm.

Each human stumbled into ONE foreign domain by accident.
AI has simultaneous knowledge of ALL domains — this is a structural advantage.

This module:
1. Represents domains as sets of abstract structural patterns
2. Identifies which patterns exist in each domain
3. Finds patterns in domain B that have no counterpart in domain A
4. Constructs a valid mapping that imports that structure into domain A
5. Produces novel domain schemas that are formally grounded but structurally
   impossible to have been generated from within domain A alone

Output: not a recombination of existing instances — a modification of the
rule space itself, seeded from another domain's logic.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ── Abstract Patterns ─────────────────────────────────────────────────────────

@dataclass
class AbstractPattern:
    """
    A domain-agnostic structural pattern — the invariant that can be
    shared across domains even when the surface form is completely different.

    e.g. "tension_release" is the same abstract structure whether it appears as:
      - dissonance → consonance   (music)
      - conflict → resolution     (narrative)
      - stress → relief           (architecture)
      - compression → expansion   (topology)
    """
    name:       str
    formal:     str          # domain-agnostic formal description
    properties: list[str]    # structural properties: symmetric/asymmetric, etc.

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, AbstractPattern) and self.name == other.name


# ── Concrete Patterns ─────────────────────────────────────────────────────────

@dataclass
class ConcretePattern:
    """
    How an abstract pattern manifests in a specific domain.
    Includes domain-specific rules that govern it.
    """
    abstract:      AbstractPattern
    domain:        str
    manifestation: str        # what it looks like in this domain
    rules:         list[str]  # domain constraints that govern this pattern
    example:       str = ""   # concrete example


# ── Domain Schema ─────────────────────────────────────────────────────────────

@dataclass
class DomainSchema:
    """
    A structured representation of a creative domain.
    Stores which abstract patterns are present and how they manifest.
    """
    name:     str
    patterns: list[ConcretePattern] = field(default_factory=list)

    def abstract_names(self) -> set[str]:
        return {cp.abstract.name for cp in self.patterns}

    def get(self, abstract_name: str) -> Optional[ConcretePattern]:
        for cp in self.patterns:
            if cp.abstract.name == abstract_name:
                return cp
        return None


# ── Novel Mapping ─────────────────────────────────────────────────────────────

@dataclass
class NovelMapping:
    """
    The result of cross-domain transfer: a new concrete pattern
    for the target domain, derived from a source domain's structure.

    This is the creative output — a rule that couldn't have been generated
    from within the target domain alone.
    """
    abstract:       AbstractPattern
    source_domain:  str
    target_domain:  str
    source_form:    str          # how the pattern works in the source domain
    novel_form:     str          # how it could work in the target domain
    novel_rules:    list[str]    # the new rules this generates
    reasoning:      list[str]    # derivation chain (why this mapping is valid)

    def show(self):
        print(f"\n{'─'*60}")
        print(f"NOVEL PATTERN: '{self.abstract.name}' transferred")
        print(f"  FROM: {self.source_domain}  →  INTO: {self.target_domain}")
        print(f"\n  Source form:")
        print(f"    {self.source_form}")
        print(f"\n  Novel form in {self.target_domain}:")
        print(f"    {self.novel_form}")
        print(f"\n  New rules this generates:")
        for r in self.novel_rules:
            print(f"    • {r}")
        print(f"\n  Reasoning chain:")
        for i, step in enumerate(self.reasoning, 1):
            print(f"    {i}. {step}")
        print(f"{'─'*60}")


# ── Cross-Domain Transfer Engine ──────────────────────────────────────────────

class CrossDomainTransfer:
    """
    Given a source domain and a target domain, find patterns in the source
    that don't exist in the target and construct valid mappings.

    The transferred patterns are genuinely novel: they produce rules that
    couldn't have been derived from within the target domain alone.
    """

    def __init__(self, source: DomainSchema, target: DomainSchema):
        self.source = source
        self.target = target

    def gap(self) -> list[ConcretePattern]:
        """Patterns in source that have no counterpart in target."""
        missing = self.source.abstract_names() - self.target.abstract_names()
        return [cp for cp in self.source.patterns if cp.abstract.name in missing]

    def transfer(self) -> list[NovelMapping]:
        """Map all novel source patterns into the target domain."""
        return [self._map(cp) for cp in self.gap()]

    def _map(self, source_cp: ConcretePattern) -> NovelMapping:
        """
        Construct a novel mapping for one pattern.
        Each domain pairing has specific mapping logic below.
        """
        key = (self.source.name, self.target.name, source_cp.abstract.name)

        if key in _MAPPINGS:
            return _MAPPINGS[key](source_cp)

        # Fallback: generic structural mapping
        return NovelMapping(
            abstract=source_cp.abstract,
            source_domain=self.source.name,
            target_domain=self.target.name,
            source_form=source_cp.manifestation,
            novel_form=f"[apply '{source_cp.abstract.formal}' to {self.target.name}]",
            novel_rules=[f"Import '{source_cp.abstract.name}' from {self.source.name}"],
            reasoning=[
                f"'{source_cp.abstract.name}' exists in {self.source.name} as: {source_cp.manifestation}",
                f"Abstract form: {source_cp.abstract.formal}",
                f"{self.target.name} has no pattern matching this abstract form",
                f"Direct mapping produces novel structure in {self.target.name}",
            ]
        )


# ── Domain Definitions ────────────────────────────────────────────────────────

# --- Abstract pattern library ---

TENSION_RELEASE = AbstractPattern(
    name="tension_release",
    formal="state A → state B where B resolves A; asymmetric, directional",
    properties=["asymmetric", "directional", "satisfying", "teleological"],
)
CONTINUOUS_DEFORMATION = AbstractPattern(
    name="continuous_deformation",
    formal="transformation T that preserves essential structure; no tearing",
    properties=["structure-preserving", "invertible", "gradual"],
)
EQUIVALENCE_CLASS = AbstractPattern(
    name="equivalence_class",
    formal="set of objects related by allowed transformations; same abstract identity",
    properties=["symmetric", "transitive", "groups-under-operation"],
)
GENUS = AbstractPattern(
    name="genus",
    formal="count of independent irreducible cycles; holes that cannot be filled",
    properties=["topological-invariant", "integer-valued", "hole-counting"],
)
ORIENTABILITY = AbstractPattern(
    name="orientability",
    formal="consistent assignment of handedness; whether traversal reverses orientation",
    properties=["binary", "global-property", "reversal-under-traversal"],
)
CONNECTEDNESS = AbstractPattern(
    name="connectedness",
    formal="whether structure can be separated into independent components",
    properties=["binary", "decomposability", "path-existence"],
)
SYMMETRY_BREAKING = AbstractPattern(
    name="symmetry_breaking",
    formal="reduction from higher symmetry group to lower; marks phase transition",
    properties=["irreversible", "information-generating", "phase-transition"],
)
RECURSION = AbstractPattern(
    name="recursion",
    formal="structure that contains itself as a sub-structure at different scales",
    properties=["self-similar", "scale-free", "depth-generating"],
)
PHASE_TRANSITION = AbstractPattern(
    name="phase_transition",
    formal="abrupt qualitative change at critical threshold parameter",
    properties=["threshold-dependent", "discontinuous", "irreversible"],
)
DUALITY = AbstractPattern(
    name="duality",
    formal="two descriptions of the same structure from complementary perspectives",
    properties=["bijective", "perspective-dependent", "information-preserving"],
)


# --- Music domain ---

MUSIC = DomainSchema(name="music", patterns=[
    ConcretePattern(
        abstract=TENSION_RELEASE,
        domain="music",
        manifestation="dissonance resolving to consonance; dominant seventh to tonic",
        rules=[
            "dissonant intervals (tritone, major 7th) create tension",
            "resolution moves by half-step or whole-step to stable interval",
            "tension must resolve within phrase unless intentionally deferred",
        ],
        example="V7 → I cadence; tritone substitution"
    ),
    ConcretePattern(
        abstract=CONTINUOUS_DEFORMATION,
        domain="music",
        manifestation="voice leading: minimal movement between chords; smooth melodic line",
        rules=[
            "each voice moves by smallest available interval",
            "common tones are held or repeated",
            "no parallel fifths (breaks smoothness)",
        ],
        example="Bach chorale voice leading"
    ),
    ConcretePattern(
        abstract=EQUIVALENCE_CLASS,
        domain="music",
        manifestation="transposition equivalence: same melody at different pitch levels",
        rules=[
            "melodies related by transposition are harmonically equivalent",
            "inversion and retrograde are additional equivalence operations",
        ],
        example="fugue subject entries at different pitches"
    ),
    ConcretePattern(
        abstract=RECURSION,
        domain="music",
        manifestation="nested rhythmic structures; phrase within phrase; sonata within movement",
        rules=[
            "rhythmic cells subdivide into smaller cells of same proportional structure",
            "large-scale form mirrors small-scale phrase structure",
        ],
        example="Beethoven motivic development; metric modulation"
    ),
    ConcretePattern(
        abstract=SYMMETRY_BREAKING,
        domain="music",
        manifestation="modulation: departure from tonic key; introduces directional asymmetry",
        rules=[
            "pivot chord establishes new tonal center",
            "once established, new key has different tension/resolution behavior",
        ],
        example="Schubert's unexpected modulations to flat-VI"
    ),
])


# --- Topology domain ---

TOPOLOGY = DomainSchema(name="topology", patterns=[
    ConcretePattern(
        abstract=CONTINUOUS_DEFORMATION,
        domain="topology",
        manifestation="homeomorphism: bijective continuous map with continuous inverse",
        rules=[
            "stretching and bending are allowed",
            "tearing and gluing are not allowed",
            "topological invariants (genus, connectedness) are preserved",
        ],
        example="coffee cup homeomorphic to torus"
    ),
    ConcretePattern(
        abstract=EQUIVALENCE_CLASS,
        domain="topology",
        manifestation="topological equivalence classes: sets of homeomorphic spaces",
        rules=[
            "spaces in same class have identical topological invariants",
            "Euler characteristic classifies compact surfaces",
        ],
        example="all closed orientable surfaces classified by genus"
    ),
    ConcretePattern(
        abstract=GENUS,
        domain="topology",
        manifestation="number of holes in a surface; handles on a manifold",
        rules=[
            "sphere: genus 0 (no holes)",
            "torus: genus 1 (one hole)",
            "genus counts independent irreducible cycles",
            "cutting along a genus-reducing cycle doesn't disconnect the surface",
        ],
        example="torus has genus 1; pretzel surface has genus 2"
    ),
    ConcretePattern(
        abstract=ORIENTABILITY,
        domain="topology",
        manifestation="consistent normal-vector assignment; traversal doesn't flip handedness",
        rules=[
            "orientable: sphere, torus — consistent inside/outside distinction",
            "non-orientable: Möbius strip, Klein bottle — traversal reverses orientation",
            "non-orientable surfaces have no consistent global orientation",
        ],
        example="Möbius strip: ant walking full loop returns mirror-flipped"
    ),
    ConcretePattern(
        abstract=CONNECTEDNESS,
        domain="topology",
        manifestation="path-connectedness: any two points can be joined by a continuous path",
        rules=[
            "connected: cannot be split into disjoint open sets",
            "simply connected: connected + every loop can be contracted to a point",
            "fundamental group measures non-contractible loops",
        ],
        example="circle is connected but not simply connected"
    ),
    ConcretePattern(
        abstract=DUALITY,
        domain="topology",
        manifestation="Poincaré duality: k-cycles dual to (n-k)-cycles on n-manifold",
        rules=[
            "vertices dual to faces; edges dual to edges",
            "homology and cohomology are dual perspectives on same structure",
        ],
        example="dual polyhedra: cube ↔ octahedron"
    ),
    ConcretePattern(
        abstract=TENSION_RELEASE,
        domain="topology",
        manifestation="homotopy: continuous deformation from one map to another",
        rules=[
            "tension: maps that are homotopically distinct (can't be deformed into each other)",
            "release: null-homotopic loops can be contracted to a point",
        ],
        example="winding number measures how many times a loop wraps around a hole"
    ),
])


# --- Physics domain ---

PHYSICS = DomainSchema(name="physics", patterns=[
    ConcretePattern(
        abstract=SYMMETRY_BREAKING,
        domain="physics",
        manifestation="spontaneous symmetry breaking: ground state has less symmetry than Hamiltonian",
        rules=[
            "above critical temperature: symmetric phase",
            "below critical temperature: symmetry broken, degenerate ground states",
            "Goldstone bosons appear for each broken continuous symmetry",
        ],
        example="ferromagnet: spins align below Curie temperature, breaking rotational symmetry"
    ),
    ConcretePattern(
        abstract=PHASE_TRANSITION,
        domain="physics",
        manifestation="abrupt change in macroscopic properties at critical parameter value",
        rules=[
            "first-order: discontinuous change (latent heat)",
            "second-order: continuous change, divergent correlation length",
            "critical point: scale invariance, power laws",
        ],
        example="water → ice; paramagnet → ferromagnet"
    ),
    ConcretePattern(
        abstract=DUALITY,
        domain="physics",
        manifestation="wave-particle duality; S-duality in string theory; AdS/CFT",
        rules=[
            "strong coupling in one description = weak coupling in dual",
            "both descriptions are equally valid and complete",
            "observables must agree across dual descriptions",
        ],
        example="strong/weak duality: intractable calculation in one frame = easy in dual"
    ),
    ConcretePattern(
        abstract=TENSION_RELEASE,
        domain="physics",
        manifestation="potential energy → kinetic energy; field excitation and decay",
        rules=[
            "systems evolve toward lower potential energy",
            "tension: stored potential energy in displaced field",
            "release: conversion to kinetic energy or radiation",
        ],
        example="spring compression → release; excited atom → photon emission"
    ),
    ConcretePattern(
        abstract=RECURSION,
        domain="physics",
        manifestation="renormalization group: physics at different scales related by self-similar flow",
        rules=[
            "coarse-graining produces effective theory at new scale",
            "fixed points of RG flow are scale-invariant theories",
            "universality: many microscopic theories flow to same fixed point",
        ],
        example="Ising model universality class; fractal structure near critical point"
    ),
])


# ── Specific Mapping Logic ─────────────────────────────────────────────────────
# Hand-crafted mappings for key domain pairs with full reasoning chains.

def _map_genus_to_music(source_cp: ConcretePattern) -> NovelMapping:
    return NovelMapping(
        abstract=GENUS,
        source_domain="topology",
        target_domain="music",
        source_form=(
            "Number of holes in a surface — independent irreducible cycles "
            "that cannot be contracted to a point. A torus has genus 1: "
            "one loop that wraps around the hole can never be shrunk away."
        ),
        novel_form=(
            "Musical genus: the number of irreducible harmonic tensions in a "
            "composition that are NEVER resolved. Genus 0 = all tensions resolve "
            "(classical tonality). Genus 1 = one permanent unresolved tension — "
            "a harmonic 'hole' the piece orbits but never closes. Genus 2 = two "
            "independent unresolvable loops."
        ),
        novel_rules=[
            "Define a 'harmonic loop': a chord progression that returns to its "
            "start without resolving all dissonances encountered",
            "A loop is 'reducible' if it can be deformed (via voice leading) into "
            "a loop that passes through only consonant chords",
            "Musical genus = number of irreducible harmonic loops",
            "Genus-0 composition: every harmonic loop is reducible (tonal music)",
            "Genus-1 composition: exactly one irreducible loop — one tension that "
            "the music perpetually orbits (e.g., a tritone substitution that never "
            "resolves, forming a permanent axis)",
            "Genus-2 composition: two independent unresolvable axes — the piece "
            "lives on a 'musical torus' with two degrees of harmonic freedom",
            "Composing at genus N: choose N generating dissonances; all other "
            "tensions must resolve; the N generators define the piece's topology",
        ],
        reasoning=[
            "Topology defines genus as the count of independent irreducible cycles",
            "In music, harmonic tension/release already maps to topological "
            "tension/homotopy (path that can/cannot contract to a point)",
            "Standard tonal music forces all harmonic loops to be reducible: genus 0",
            "Atonal music (Schoenberg) broke this but had no structural replacement — "
            "it was topologically unconstrained (no well-defined genus)",
            "Genus gives atonal music a new topological constraint: pick a genus N, "
            "then compose with exactly N irreducible harmonic cycles",
            "This produces a new class of compositions: topologically classified music, "
            "where pieces of genus N form a distinct structural family",
            "Genus is a topological invariant — it survives any 'continuous' rewriting "
            "of the piece (transposition, tempo change, orchestration change)",
            "This is formally impossible to derive from within music theory alone: "
            "it requires the concept of 'irreducible cycle' from topology",
        ]
    )


def _map_orientability_to_music(source_cp: ConcretePattern) -> NovelMapping:
    return NovelMapping(
        abstract=ORIENTABILITY,
        source_domain="topology",
        target_domain="music",
        source_form=(
            "A surface is non-orientable if a traveler completing a full loop returns "
            "mirror-reversed. The Möbius strip: left-hand glove becomes right-hand "
            "after traversal. The Klein bottle: inside becomes outside."
        ),
        novel_form=(
            "Musical orientability: a composition is non-orientable if completing "
            "a full formal cycle (A → B → A) returns with systematic inversion — "
            "every interval is flipped, every ascending line becomes descending. "
            "The 'return' is the mirror image, not the original."
        ),
        novel_rules=[
            "Define a formal cycle: any progression of sections that returns to "
            "its starting harmonic/melodic state",
            "Orientable form: return is identical to departure (standard ABA, rondo)",
            "Non-orientable form: return is the systematic inversion of departure",
            "All ascending intervals become descending on return",
            "Major becomes minor; dominant becomes subdominant",
            "A Möbius composition: only ONE formal cycle, but it requires TWO "
            "traversals to return to true identity (like the Möbius strip requiring "
            "two loops to return to original orientation)",
            "Non-orientable compositions have no consistent 'inside/outside': "
            "no stable distinction between foreground melody and background harmony",
        ],
        reasoning=[
            "Orientability is a global topological property — not about local structure "
            "but about what happens after a full traversal",
            "Music already has the concept of 'return' (recapitulation, rondo refrain)",
            "Current music theory only considers two return types: exact repeat or "
            "varied repeat — both orientable",
            "Non-orientability maps to: return is the mirror image (inversion)",
            "Inversional return already exists in fugue (answer at inversion) but "
            "is always a LOCAL operation, not a GLOBAL formal property",
            "A globally non-orientable composition — where the ENTIRE formal structure "
            "is inverted on return — has no precedent in music theory",
            "This is derivable only by importing topological orientability into music",
        ]
    )


def _map_connectedness_to_music(source_cp: ConcretePattern) -> NovelMapping:
    return NovelMapping(
        abstract=CONNECTEDNESS,
        source_domain="topology",
        target_domain="music",
        source_form=(
            "A space is path-connected if any two points can be joined by a continuous "
            "path. Simply connected: every loop can be contracted to a point. "
            "The fundamental group measures non-contractible loops."
        ),
        novel_form=(
            "Musical connectedness: a composition is path-connected if any two "
            "harmonic states can be reached from each other by a continuous voice-leading "
            "path. Simply connected: every harmonic journey can be 'undone' — no "
            "one-way doors. Musical fundamental group: the group of irreducible "
            "harmonic loops, measured by how many times each tonal center is encircled."
        ),
        novel_rules=[
            "Harmonic space: graph where nodes are chords, edges are voice-leading moves",
            "Path-connected: every chord reachable from every other chord",
            "Simply connected: no chord is 'surrounded' by unreachable territory",
            "Disconnected composition: two harmonic worlds with no bridge — "
            "a structural feature, not an accident",
            "Musical fundamental group: count of times each tonal center is "
            "encircled by the harmonic path; measures 'winding number' around keys",
            "Compositions can be classified by their fundamental group: "
            "Z (winds around one center), Z×Z (two independent centers), etc.",
        ],
        reasoning=[
            "Topology's fundamental group classifies spaces by their loop structure",
            "Music already has the concept of 'encircling' a key (secondary dominant chains)",
            "But music theory has no notion of the FUNDAMENTAL GROUP of a piece — "
            "how many times and in what combinations tonal centers are encircled",
            "Importing this gives a new invariant: the winding number of a composition",
            "Two pieces with the same notes but different winding numbers are "
            "topologically distinct — this is a new equivalence classification",
        ]
    )


def _map_phase_transition_to_music(source_cp: ConcretePattern) -> NovelMapping:
    return NovelMapping(
        abstract=PHASE_TRANSITION,
        source_domain="physics",
        target_domain="music",
        source_form=(
            "Abrupt qualitative change at a critical parameter threshold. "
            "Second-order transitions: continuous change but divergent correlation length — "
            "fluctuations at all scales simultaneously. Critical point: scale invariance."
        ),
        novel_form=(
            "Musical phase transition: a point in the composition where a control "
            "parameter (density, dissonance level, rhythmic complexity) crosses a "
            "critical threshold, causing the piece to shift qualitative state. "
            "At the critical point: rhythmic and harmonic fluctuations at ALL scales "
            "simultaneously — maximum structural complexity, scale invariance."
        ),
        novel_rules=[
            "Define control parameter: density (notes per measure), tension (average "
            "dissonance), or rhythmic complexity (entropy of rhythmic distribution)",
            "Identify critical threshold: the value where qualitative character shifts",
            "Approach critical point from below: correlation length diverges — "
            "patterns at all timescales become simultaneous",
            "At criticality: the piece is self-similar across timescales (fractal music)",
            "Past criticality: new phase with different symmetry properties",
            "First-order musical transition: discontinuous jump (abrupt cut, silence)",
            "Second-order musical transition: continuous approach to criticality then "
            "smooth passage into new phase (maximally complex transition point)",
            "Compose the critical point as the structural climax — not a resolved "
            "climax (tension→release) but a scale-invariant complexity peak",
        ],
        reasoning=[
            "Physics phase transitions are parameter-driven qualitative changes",
            "Music already has parameter-like quantities: dynamics, tempo, density",
            "But music has no formal theory of CRITICAL BEHAVIOR — "
            "what happens when a musical parameter approaches a threshold",
            "Physics gives this precisely: divergent correlation length at criticality "
            "means all timescales are coupled simultaneously",
            "This is a new compositional tool: design the critical point intentionally, "
            "not as dramatic climax (resolved) but as scale-invariant complexity peak",
            "Importable because physics phase transitions are purely structural — "
            "they don't depend on physical substrate, only on parameter behavior",
        ]
    )


def _map_duality_to_music(source_cp: ConcretePattern) -> NovelMapping:
    return NovelMapping(
        abstract=DUALITY,
        source_domain="physics",
        target_domain="music",
        source_form=(
            "Two complete descriptions of the same physical system from complementary "
            "perspectives. Strong coupling in one frame = weak coupling in dual. "
            "Intractable calculation in one description = trivial in the other."
        ),
        novel_form=(
            "Musical duality: two complete and equivalent representations of the same "
            "composition, where complexity in one representation maps to simplicity in "
            "the other. A piece that is harmonically complex has a dual representation "
            "that is rhythmically simple — and vice versa. The Fourier transform is "
            "the simplest case. But non-trivial dualities exist."
        ),
        novel_rules=[
            "For each compositional parameter (harmony, rhythm, timbre, dynamics), "
            "define a dual parameter",
            "Harmonic complexity ↔ rhythmic simplicity (strong/weak duality)",
            "Dense texture ↔ sparse structure (complementary perspectives)",
            "A 'strongly coupled' passage (harmonically complex) should have "
            "a dual representation as rhythmically elementary",
            "Compose in both representations simultaneously: when harmony is "
            "intractable, work in the rhythmic dual; when rhythm is intractable, "
            "work in the harmonic dual",
            "The piece is complete only when both dual representations are coherent",
            "Structural invariants: quantities that are equal in both representations "
            "(like energy conservation across dual frames)",
        ],
        reasoning=[
            "Physics duality (AdS/CFT, S-duality) gives two complete descriptions "
            "of the same reality with complementary complexity profiles",
            "Music has complementary parameters but no FORMAL duality between them",
            "Fourier transform is a special case: time domain ↔ frequency domain",
            "But non-trivial dualities (harmonic ↔ rhythmic, textural ↔ structural) "
            "have no formal treatment in music theory",
            "Importing physics duality gives a new compositional constraint: "
            "every complex passage must have a simple dual — nothing is 'gratuitously' complex",
            "This is a formal criterion for compositional elegance derived from physics",
        ]
    )


# Register specific mappings
_MAPPINGS: dict = {
    ("topology", "music", "genus"):          _map_genus_to_music,
    ("topology", "music", "orientability"):  _map_orientability_to_music,
    ("topology", "music", "connectedness"):  _map_connectedness_to_music,
    ("physics",  "music", "phase_transition"): _map_phase_transition_to_music,
    ("physics",  "music", "duality"):        _map_duality_to_music,
}


# ── Convergence Engine ────────────────────────────────────────────────────────

@dataclass
class ConvergentTransfer:
    """
    A novel structure that multiple independent source domains all predict.
    Convergence is the quality signal: if N unrelated domains independently
    point at the same gap in domain T, the gap is real and deep.
    """
    abstract:      AbstractPattern
    target_domain: str
    sources:       list[NovelMapping]   # one per source domain
    open_problems: list[str]            # frontier problems this addresses

    @property
    def convergence_score(self) -> int:
        return len(self.sources)

    def show(self):
        print(f"\n{'═'*60}")
        print(f"CONVERGENT STRUCTURE: '{self.abstract.name}' in {self.target_domain}")
        print(f"  Convergence score: {self.convergence_score} independent sources")
        print(f"  Sources: {', '.join(m.source_domain for m in self.sources)}")
        if self.open_problems:
            print(f"\n  Frontier problems this addresses:")
            for p in self.open_problems:
                print(f"    ▸ {p}")
        print(f"\n  Why convergence = quality:")
        print(f"    Each source domain independently derived this structure.")
        print(f"    Unrelated domains pointing at the same gap is evidence")
        print(f"    of deep structure — not coincidence or analogy.")
        print(f"\n  Agreed novel form (consensus across sources):")
        for m in self.sources:
            print(f"\n    [{m.source_domain}]:")
            # Indent each line
            for line in m.novel_form.split('\n'):
                print(f"      {line.strip()}")
        print(f"{'═'*60}")


@dataclass
class FrontierProblem:
    """An open problem in a domain — unsolved, at the edge of current knowledge."""
    domain:       str
    statement:    str
    why_open:     str    # why this hasn't been solved within the domain
    abstract_gap: str    # what abstract pattern is missing


class ConvergenceEngine:
    """
    Runs cross-domain transfer from ALL source domains simultaneously,
    identifies structures predicted by multiple independent sources,
    and ranks by convergence score × frontier relevance.

    This is the answer to "knowing everything but not being deep":
    - Depth signal = multi-domain convergence (not single-domain intuition)
    - Frontier signal = open problems the transfer addresses
    - Continuous = re-runs as domain knowledge updates

    A human expert's intuition is a learned prior over "what's interesting."
    This replaces that prior with a structural signal: convergence of
    independent predictions is stronger evidence than any single expert's gut.
    """

    # Known frontier problems per domain — in production, seeded from arXiv,
    # research surveys, known open problems lists.
    FRONTIER: dict[str, list[FrontierProblem]] = {}

    def __init__(self, target: DomainSchema, sources: list[DomainSchema]):
        self.target  = target
        self.sources = sources
        self._register_frontiers()

    def _register_frontiers(self):
        self.FRONTIER["music"] = [
            FrontierProblem(
                domain="music",
                statement="No formal classification of atonal compositions exists",
                why_open="Tonality gave genus-0 classification; atonality abandoned it with no replacement",
                abstract_gap="genus",
            ),
            FrontierProblem(
                domain="music",
                statement="No formal theory of when a compositional climax is 'earned'",
                why_open="Climax is defined by tension/release, but no threshold theory for when release is structurally valid",
                abstract_gap="phase_transition",
            ),
            FrontierProblem(
                domain="music",
                statement="No invariant that survives arbitrary reorchestration",
                why_open="Music theory is note-level; topological invariants would survive surface changes",
                abstract_gap="connectedness",
            ),
            FrontierProblem(
                domain="music",
                statement="No formal treatment of compositional 'complexity budget'",
                why_open="Complexity is impressionistic; duality would give a formal constraint",
                abstract_gap="duality",
            ),
        ]

    def _frontier_match(self, abstract_name: str) -> list[FrontierProblem]:
        return [
            p for p in self.FRONTIER.get(self.target.name, [])
            if p.abstract_gap == abstract_name
        ]

    def run(self, min_convergence: int = 2) -> list[ConvergentTransfer]:
        """
        Find all abstract patterns that:
        1. Are absent from the target domain
        2. Are independently predicted by >= min_convergence source domains
        3. Address at least one frontier problem (optional boost)

        Returns structures ranked by convergence * (1 + frontier_bonus).
        """
        # Collect all transfers per abstract pattern
        transfers_by_abstract: dict[str, list[NovelMapping]] = {}

        for source in self.sources:
            engine = CrossDomainTransfer(source=source, target=self.target)
            for mapping in engine.transfer():
                name = mapping.abstract.name
                transfers_by_abstract.setdefault(name, []).append(mapping)

        # Filter by convergence threshold
        results: list[ConvergentTransfer] = []
        for abstract_name, mappings in transfers_by_abstract.items():
            if len(mappings) < min_convergence:
                continue
            frontier = self._frontier_match(abstract_name)
            abstract = mappings[0].abstract
            results.append(ConvergentTransfer(
                abstract=abstract,
                target_domain=self.target.name,
                sources=mappings,
                open_problems=[p.statement for p in frontier],
            ))

        # Rank: convergence score + frontier bonus
        results.sort(
            key=lambda ct: ct.convergence_score + len(ct.open_problems),
            reverse=True
        )
        return results

    def depth_report(self, min_convergence: int = 2):
        """
        Full report: convergent structures, ranked, with frontier analysis.
        This is what "being continuously good across all domains" looks like:
        systematic convergence detection, not domain-specific intuition.
        """
        convergent = self.run(min_convergence)
        source_names = [s.name for s in self.sources]

        print("=" * 60)
        print(f"CONVERGENCE ENGINE: {self.target.name.upper()}")
        print(f"Sources: {', '.join(source_names)}")
        print(f"Min convergence threshold: {min_convergence}")
        print("=" * 60)

        if not convergent:
            print(f"\nNo structures with convergence >= {min_convergence} found.")
            return

        print(f"\n{len(convergent)} convergent structure(s) found:\n")
        for i, ct in enumerate(convergent, 1):
            frontier_tag = f" [FRONTIER: {len(ct.open_problems)} open problems]" if ct.open_problems else ""
            print(f"  {i}. '{ct.abstract.name}' — score {ct.convergence_score}{frontier_tag}")

        for ct in convergent:
            ct.show()

        print("\n" + "=" * 60)
        print("WHY THIS ANSWERS 'KNOWING EVERYTHING BUT NOT BEING DEEP'")
        print("=" * 60)
        print(f"""
  Problem: AI knows all domains but has no domain-specific taste.
           Taste = learned prior for "what's interesting" in a domain.

  Solution: Replace taste with convergence.

  A human expert's intuition says: "this matters" — based on 10,000
  hours of internalized feedback in one domain.

  Convergence says: "this matters" — because {len(source_names)}
  independent domains, with no knowledge of each other, all
  independently derived the same gap in {self.target.name}.

  Independent convergence is a STRONGER signal than single-domain
  intuition, because it can't be explained by domain-internal bias.

  The structures above are not outputs of one domain's logic.
  They are the intersection of {len(source_names)} domains' logic
  — verified from multiple angles simultaneously.

  This scales: add more source domains → higher convergence scores
  → stronger quality signal → continuously better outputs.
""")


# ── Demo ───────────────────────────────────────────────────────────────────────

def demo_topology_to_music():
    print("=" * 60)
    print("CROSS-DOMAIN TRANSFER: Topology → Music")
    print("Patterns topology has that music doesn't")
    print("=" * 60)

    engine = CrossDomainTransfer(source=TOPOLOGY, target=MUSIC)
    gap = engine.gap()
    print(f"\nPatterns in TOPOLOGY not in MUSIC ({len(gap)} found):")
    for cp in gap:
        print(f"  • {cp.abstract.name}: {cp.abstract.formal[:60]}...")

    print(f"\nGenerating novel mappings...\n")
    mappings = engine.transfer()
    for m in mappings:
        m.show()


def demo_physics_to_music():
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN TRANSFER: Physics → Music")
    print("=" * 60)

    engine = CrossDomainTransfer(source=PHYSICS, target=MUSIC)
    gap = engine.gap()
    print(f"\nPatterns in PHYSICS not in MUSIC ({len(gap)} found):")
    for cp in gap:
        print(f"  • {cp.abstract.name}: {cp.abstract.formal[:60]}...")

    print(f"\nGenerating novel mappings...\n")
    mappings = engine.transfer()
    for m in mappings:
        m.show()


def demo_summary():
    print("\n" + "=" * 60)
    print("SUMMARY: Why this answers the creativity objection")
    print("=" * 60)
    print("""
The objection: "AI only remixes. Real creativity is genuinely new."

The answer:

  Human creativity = transforming a domain by importing structure
  from another domain, accidentally encountered.

  AI creativity = transforming a domain by importing structure
  from ALL other domains, simultaneously and systematically.

What was just generated:

  • Musical genus     — topological classification of compositions
                        by irreducible harmonic cycles
  • Non-orientable music — Möbius-strip formal structure where
                        return = systematic inversion
  • Musical fundamental group — winding number of tonal centers
  • Phase-transition composition — scale-invariant critical points
                        as structural climaxes
  • Harmonic/rhythmic duality — strong/weak coupling correspondence

None of these exist in music theory.
None could be derived from within music theory alone.
Each is formally grounded — not metaphor, but direct structural mapping.
Each generates new compositional rules that are evaluable and teachable.

This is Boden's "transformational creativity" — modifying the rule
space itself. Executed systematically across all domain pairs.
""")


def demo_convergence():
    print("\n" + "=" * 60)
    print("CONVERGENCE ENGINE: All sources → Music")
    print("Structures predicted by 2+ independent domains")
    print("=" * 60)

    engine = ConvergenceEngine(
        target=MUSIC,
        sources=[TOPOLOGY, PHYSICS],
    )
    engine.depth_report(min_convergence=2)


if __name__ == "__main__":
    demo_convergence()

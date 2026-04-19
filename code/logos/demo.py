"""
demo.py
-------
Shows LOGOS doing something chain-of-thought cannot:
explicit confidence propagation through a multi-step chain,
with contradiction detection and OOD flagging.

Three demos:
  1. Classic syllogism — shows provenance chain
  2. Confidence decay — shows u(c) dropping through a long chain
  3. Contradiction catch — shows graph halting on conflict
"""

from logos import LOGOS, Rule


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: Syllogism with provenance
# ─────────────────────────────────────────────────────────────────────────────

def demo_syllogism():
    print("=" * 60)
    print("DEMO 1: Syllogism with explicit provenance chain")
    print("=" * 60)

    logos = LOGOS()

    logos.assert_premise("All humans are mortal",    confidence=0.99)
    logos.assert_premise("Socrates is a human",      confidence=0.99)
    logos.assert_premise("All mortals will die",     confidence=0.99)

    logos.add_rule(Rule(
        id="human_mortal",
        pattern=["All humans are mortal", "Socrates is a human"],
        conclusion="Socrates is mortal",
        confidence=1.0,
    ))
    logos.add_rule(Rule(
        id="mortal_dies",
        pattern=["All mortals will die", "Socrates is mortal"],
        conclusion="Socrates will die",
        confidence=1.0,
    ))

    logos.reason()
    logos.show()
    print(logos.explain("Socrates will die"))


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: Confidence decay through a long chain
# ─────────────────────────────────────────────────────────────────────────────

def demo_confidence_decay():
    print("\n" + "=" * 60)
    print("DEMO 2: Confidence decay — u(c) propagates through chain")
    print("Each step 90% confident. Chain of 5 = 0.9^5 = 59% final.")
    print("=" * 60)

    logos = LOGOS()

    logos.assert_premise("A is true",   confidence=0.90)
    logos.assert_premise("A implies B", confidence=0.90)
    logos.assert_premise("B implies C", confidence=0.90)
    logos.assert_premise("C implies D", confidence=0.90)
    logos.assert_premise("D implies E", confidence=0.90)

    logos.add_rule(Rule("a_to_b", ["A is true", "A implies B"], "B is true", 0.90))
    logos.add_rule(Rule("b_to_c", ["B is true", "B implies C"], "C is true", 0.90))
    logos.add_rule(Rule("c_to_d", ["C is true", "C implies D"], "D is true", 0.90))
    logos.add_rule(Rule("d_to_e", ["D is true", "D implies E"], "E is true", 0.90))

    logos.reason()
    logos.show()

    e_node = logos.query("E is true")
    if e_node:
        print(f"Final confidence in E: {e_node.confidence:.3f}")
        print(f"(Chain-of-thought would not tell you this.)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: Contradiction detection
# ─────────────────────────────────────────────────────────────────────────────

def demo_contradiction():
    print("\n" + "=" * 60)
    print("DEMO 3: Contradiction catch — graph halts on conflict")
    print("=" * 60)

    logos = LOGOS()

    logos.assert_premise("The defendant was in Paris",   confidence=0.85)
    logos.assert_premise("The crime occurred in London", confidence=0.95)
    logos.assert_premise("Paris is not London",          confidence=1.00)

    logos.add_rule(Rule(
        id="alibi_rule",
        pattern=["The defendant was in Paris", "Paris is not London"],
        conclusion="The defendant was not in London",
        confidence=0.99,
    ))
    logos.add_rule(Rule(
        id="presence_rule",
        pattern=["The crime occurred in London", "The defendant was not in London"],
        conclusion="not The defendant committed the crime",
        confidence=0.90,
    ))

    logos.reason()
    logos.show()

    audit = logos.audit()
    print(f"Active: {audit['active']}  Suspended: {audit['suspended']}  Contradicted: {audit['contradicted']}")
    print("\nReasoning log:")
    for entry in audit["log"]:
        print(f"  {entry}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_syllogism()
    demo_confidence_decay()
    demo_contradiction()

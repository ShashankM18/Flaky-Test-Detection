"""Prompts. The baseline follows the template in Berndt et al. (SANER 2026), Fig. 1.

For the exact wording and the six few-shot demonstrations, replace these with the
prompts from their replication package (Zenodo DOI 10.5281/zenodo.18311270) so that
your baseline is directly comparable.
"""

ZERO_SHOT = """Role: Expert Software Engineer.
Task: Classify the provided test as flaky or not.
Instructions:
1) Analyze the <CODE> segment, which contains an existing test.
2) Classify the test as either flaky or not.
3) In your response, only include the following labels, refrain from using any natural language.
0: the test is not flaky
1: the test is flaky

<CODE>
{code}
</CODE>"""

ZERO_SHOT_COT = """Role: Expert Software Engineer.
Task: Classify the provided test as flaky or not.
Instructions:
1) Analyze the <CODE> segment, which contains an existing test.
2) Think about ways to improve the test or to fix existing issues with the test code.
3) Identify potential sources of flakiness. Think step by step.
4) Classify the test as either flaky or not based on your thoughts on existing issues.
5) In your response, only include the following labels, refrain from using any natural language.
0: the test is not flaky
1: the test is flaky

<CODE>
{code}
</CODE>"""

# RAG variant: same instructions, plus a <CONTEXT> block with retrieved repository code.
ZERO_SHOT_COT_CONTEXT = """Role: Expert Software Engineer.
Task: Classify the provided test as flaky or not.
Instructions:
1) Analyze the <CODE> segment, which contains an existing test.
2) Use the <CONTEXT> segment, which contains related code from the same repository
   (setup/teardown methods, fields, helper methods and code under test), to understand
   state, side effects and dependencies that are not visible in the test itself.
3) Think about ways to improve the test or to fix existing issues with the test code.
4) Identify potential sources of flakiness. Think step by step.
5) Classify the test as either flaky or not based on your thoughts on existing issues.
6) In your response, only include the following labels, refrain from using any natural language.
0: the test is not flaky
1: the test is flaky

<CONTEXT>
{context}
</CONTEXT>

<CODE>
{code}
</CODE>"""


def build_context(parts: dict, include: tuple[str, ...]) -> str:
    """Assemble context sections from java_parser.extract() output (+ retrieved code)."""
    titles = {
        "fields": "Fields of the test class",
        "setup_teardown": "Setup / teardown methods",
        "helpers": "Helper methods called by the test",
        "retrieved": "Retrieved code from the repository",
    }
    out = []
    for key in include:
        items = parts.get(key) or []
        if items:
            out.append(f"// ---- {titles[key]} ----\n" + "\n\n".join(items))
    return "\n\n".join(out) if out else "// (no additional context found)"

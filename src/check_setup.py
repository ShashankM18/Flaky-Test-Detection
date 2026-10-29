"""Day-one sanity check: Java parsing works and the OpenRouter key works.

  python src/check_setup.py
"""
import tempfile
from pathlib import Path

SAMPLE = """package demo;
import org.junit.*;
public class CacheTest {
    private static Cache cache = new Cache();
    @Before public void setUp() { cache.put("a", 1); }
    @Test public void testSize() { fill(); Assert.assertEquals(2, cache.size()); }
    private void fill() { cache.put("b", 2); }
}
"""


def check_parser():
    from java_parser import extract
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "src/test/java/demo"
        p.mkdir(parents=True)
        (p / "CacheTest.java").write_text(SAMPLE)
        r = extract(Path(d), "demo.CacheTest.testSize")
    assert r and r["helpers"] and r["setup_teardown"] and r["fields"], r
    print("[ok] Tree-sitter Java parsing (found test, setup, field, helper)")


def check_llm():
    from config import LLM_MODEL, LLM_PROVIDER
    from llm_client import Cache, ask
    rec = ask("Reply with only the digit 1.", Cache("_setup_check"), "check", run=0)
    print(f"[ok] LLM call: model={LLM_MODEL}, provider requested={LLM_PROVIDER or '(any)'}, "
          f"served by={rec['provider_served']}, reply={rec['content']!r}, "
          f"tokens={rec['usage'].get('total_tokens')}")


if __name__ == "__main__":
    check_parser()
    check_llm()

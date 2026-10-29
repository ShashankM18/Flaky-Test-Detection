"""Static extraction of a test and its *structural* context with Tree-sitter.

Given a checked-out repo and a fully-qualified test name (pkg.Class.method), returns:
  test_code      - the test method itself (what the base paper used)
  setup_teardown - @Before/@After/@BeforeEach/@AfterAll/... methods of the test class
  fields         - field declarations of the test class (shared/static state)
  helpers        - methods of the same class (or its superclass in the repo) called by the test
  callees        - names of methods/classes the test touches (input for production-code retrieval)
"""
import re
from functools import lru_cache
from pathlib import Path

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

JAVA = Language(tsjava.language())
_parser = Parser(JAVA)

SETUP_ANNOTATIONS = {
    "Before", "After", "BeforeEach", "AfterEach", "BeforeClass", "AfterClass",
    "BeforeAll", "AfterAll", "BeforeMethod", "AfterMethod",  # JUnit 4/5, TestNG
}


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


@lru_cache(maxsize=2048)
def _parse(path: str):
    src = Path(path).read_bytes()
    return _parser.parse(src), src


def find_test_file(repo: Path, fqn: str, module_path: str = ""):
    """pkg.Outer.method or pkg.Outer$Inner.method -> path of Outer.java."""
    parts = fqn.split(".")
    pkg, cls = parts[:-2], parts[-2].split("$")[0]
    rel = Path(*pkg, f"{cls}.java") if pkg else Path(f"{cls}.java")
    base = repo / module_path if module_path and module_path != "." else repo
    for root in (base, repo):
        hits = [p for p in root.rglob(f"{cls}.java") if str(p).endswith(str(rel))]
        if hits:
            # prefer test source folders
            hits.sort(key=lambda p: ("test" not in str(p).lower(), len(str(p))))
            return hits[0]
    return None


def _class_node(tree, src, class_name):
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type in ("class_declaration", "enum_declaration", "record_declaration"):
            name = n.child_by_field_name("name")
            if name is not None and _text(name, src) == class_name:
                return n
        stack.extend(n.children)
    return None


def _annotations(method, src):
    mods = next((c for c in method.children if c.type == "modifiers"), None)
    if mods is None:
        return set()
    out = set()
    for a in mods.children:
        if a.type in ("marker_annotation", "annotation"):
            name = a.child_by_field_name("name")
            if name is not None:
                out.add(_text(name, src).split(".")[-1])
    return out


def _members(cls_node):
    body = cls_node.child_by_field_name("body")
    return list(body.named_children) if body is not None else []


def _invocations(node, src):
    """Names of methods called inside node: (object_text_or_None, method_name)."""
    calls, stack = [], [node]
    while stack:
        n = stack.pop()
        if n.type == "method_invocation":
            name = n.child_by_field_name("name")
            obj = n.child_by_field_name("object")
            calls.append((_text(obj, src) if obj is not None else None, _text(name, src)))
        stack.extend(n.children)
    return calls


def _types_used(node, src):
    out, stack = set(), [node]
    while stack:
        n = stack.pop()
        if n.type == "type_identifier":
            out.add(_text(n, src))
        stack.extend(n.children)
    return out


def extract(repo: Path, fqn: str, module_path: str = "") -> dict | None:
    path = find_test_file(Path(repo), fqn, module_path)
    if path is None:
        return None
    tree, src = _parse(str(path))
    class_chain = fqn.split(".")[-2].split("$")
    cls = _class_node(tree, src, class_chain[-1])
    method_name = fqn.split(".")[-1]
    method_name = re.sub(r"\[.*\]$", "", method_name)  # parameterized: test[0]
    if cls is None:
        return None

    methods = [m for m in _members(cls) if m.type == "method_declaration"]
    test = next((m for m in methods
                 if _text(m.child_by_field_name("name"), src) == method_name), None)
    if test is None:
        return None

    setup = [_text(m, src) for m in methods if _annotations(m, src) & SETUP_ANNOTATIONS]
    fields = [_text(f, src) for f in _members(cls) if f.type == "field_declaration"]

    # Helpers: same-class methods called (transitively, depth 2) by the test.
    by_name = {}
    for m in methods:
        by_name.setdefault(_text(m.child_by_field_name("name"), src), []).append(m)
    helpers, seen, frontier = [], {method_name}, [test]
    for _ in range(2):
        nxt = []
        for node in frontier:
            for obj, name in _invocations(node, src):
                if obj in (None, "this") and name in by_name and name not in seen:
                    seen.add(name)
                    for m in by_name[name]:
                        helpers.append(_text(m, src))
                        nxt.append(m)
        frontier = nxt

    superclass = cls.child_by_field_name("superclass")
    calls = _invocations(test, src)
    return {
        "test_file": str(path.relative_to(repo)),
        "test_class": class_chain[-1],
        "superclass": _text(superclass, src).replace("extends", "").strip() if superclass else None,
        "test_code": _text(test, src),
        "setup_teardown": setup,
        "fields": fields,
        "helpers": helpers,
        "callees": sorted({n for _, n in calls} - seen),
        "types_used": sorted(_types_used(test, src)),
    }

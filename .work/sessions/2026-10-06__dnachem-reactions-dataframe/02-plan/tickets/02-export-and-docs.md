# Ticket 02: export-and-docs

**Model:** haiku
**Effort:** low

**Acceptance Criteria:**
- [ ] `from g4utils.DnaChem import load_reaction_table` works, and `"load_reaction_table"` is in `__all__` (keep its sort order).
- [ ] The README's DnaChem section documents `load_reaction_table`: accepted inputs, columns, the filter examples `df[df["product_H2O2"] > 0]` and `df[df["reactant_OH"] > 0]`, quoted access for charged species (`df["product_OH-"]`), and joining with `load_reactions` on `reactionId`.

**Files to Touch:**
- `src/g4utils/DnaChem/__init__.py`
- `README.md`

**Verification Step:**

Run:
```bash
python -c "from g4utils.DnaChem import load_reaction_table, __all__; assert 'load_reaction_table' in __all__; print('ok')" && python -m pytest tests -q
```

Expected:
`ok`, then all tests pass.

**Notes:**

Put the README text after the `load_reactions()` paragraph (around `README.md:191`), in the same style. Mention that species names are Short names (see CONTEXT.md), with no `*` radical marks.

# Ground-truth schema + normalization spec — sample.pdf

Defines what counts as a correct answer and how values are written, so two correct transcriptions can't disagree. The spec is part of the GT: scoring is undefined without it.

## Scope
- Score **filled-in entries only** (handwritten values). Printed form labels are layout, not field gold.
- Canonical gold: `cells/page{1,2,3}.cells.json` (full-page, per-cell, tri-state).
- Three scored layers, each its own metric:
  - `cells/*.json` filled cells → field extraction (per-key exact match / F1)
  - `cells/*.json` empty cells → empty-fidelity (hallucination check)
  - `cells/page3.cells.json` (`table_A_income_breakdown`) → table parse (cell accuracy / TEDS)
  - full-text reading-order CER → **deferred** (needs Japanese reader; no fabricated prose gold)
- Only keys listed GOLD in `scored_mask.md` are scored. CANDIDATE and EXCLUDED are never scored.

## Number normalization
- Integer yen, **no commas, no ¥, half-width digits**: `1,947,075` → `1947075`.
- Full-width digits → half-width (NFKC): `１９４７` → `1947`.
- Unit is JPY throughout; stored as the integer, `unit` field carries `JPY`.
- **Empty vs zero are distinct**: blank cell = `null` (not scored); a written `0` = `0`.
- Non-money quantities (e.g. 作付面積 planted area, in アール/`a`) follow the same rule: store the **bare integer**, declare the unit once in the table's `units` map — never suffix the value (`150a` → `150`).

## Text normalization
- Apply NFKC. Full-width ASCII/digits → half-width; kanji/kana kept as written.
- Phone: digits + hyphens, half-width: `095-2853-4416`.
- Name kept in kanji as printed; **reading (kana) is a separate key**, not interchangeable.

## Field keys (in `cells/*.json`)
- Per page; ids: `h.*` header, `pl.N` P/L income, `e.N` expenses, `r.N` result (page2); `i.*` income, `t.*` tax, `o.*` other (page1); `m.*` misc, `table_A/B/C/D` tables (page3).

## `no` field (human cross-ref to the form)
- Numbered fields → **plain ASCII**: `"60"`, range `"31-47"` (circled glyph is presentation-only, rendered in `full-page.md`). Why: Unicode has no circled glyph >50; plain ASCII is uniform + scalable.
- Katakana income rows (㋐㋑㋒…) → keep the **circled katakana** as-is. Why: faithful, renders, no Unicode gap.
- Grouped cells → hyphen range + `"grouped": true`. Never fabricate a range endpoint; if the extent is unverified, mark `status: "unread"` (e.g. `㋓+`).

## Match rule
- Numbers: exact integer equality after normalization.
- Text: exact string after NFKC; near-miss (1 char) logged separately, not auto-credited.

## Confidence → scoring
- `H` printed or arithmetic-cross-validated → GOLD.
- `M` single legible read → CANDIDATE (confirm before scoring).
- `L` ambiguous / needs zoom or Japanese reader → EXCLUDED.

## Freeze
- GT is keyed to source sha256 in `SOURCE.sha256`. Edits = new version, never silent overwrite.

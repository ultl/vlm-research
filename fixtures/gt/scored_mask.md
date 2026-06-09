# Scored mask — what counts in the metric

Canonical gold is `cells/page{1,2,3}.cells.json`. Keys below are `pageN <id>` from those files. Models are scored ONLY on GOLD; CANDIDATE and EXCLUDED are never scored (scoring against a guess invalidates the metric).

## GOLD — scored now (arithmetic-validated or printed)
- form codes: page1/2/3 `meta.form` = FA2204 / FA3100 / FA3125; page3 `h.year` = R06
- page2 `pl.1` 1947075
- page2 `pl.2` 1510000
- page2 `pl.3` 1000000
- page2 `pl.4` 4457075
- page2 `pl.5` 0
- page2 `pl.6` 0
- page2 `pl.7` 4457075
- page3 `m.subsidy` 500000
- page3 `m.3` 1000000
- page3 `table_A.totals.sales_amount_1` 1947075
- page3 `table_A.totals.home_business_consumption_2` 1510000

## GOLD-EMPTY — scored for hallucination
- Every cell with `status == "empty"` in `cells/*.json` (47 cells). Model MUST emit blank; inventing a value = fail. This is the non-tech-user trust test.

## CANDIDATE — confirm before scoring (single legible read)
- page2 `h.address`, `h.industry`, `h.name`, `h.kana`, `h.phone`
- page2 `r.46` 3334984 — does NOT reconcile with read expenses, confirm before trusting
- page2 `e.9` 27685, `e.11` 12638, `e.14` 78730 — legible, no cross-check
- page3 `m.insurance` 500000

## EXCLUDED — not scoreable yet (reason)
- page2 `h.farm_name` — kana unclear, needs reader
- page3 `table_A.rows[*]` — sum-check FAILS, one row misread
- page1 `i.i` 8264854 — single read, conflicts with page-2 total
- page1 `unread` cells (deduction/tax `0000` marks, `a.2`, `t.*`, `o.66_67`) — ambiguous
- page2 `unread` cells (`e.20` depreciation, `e.31`/`e.35` subtotals, `r.36`/`r.47`/`r.48`)
- Any full Japanese prose / reading-order text — needs Japanese reader; no fabricated gold
- Handwritten-kanji name *reading* correctness (`h.kana`) — needs reader

## Promotion path
- CANDIDATE → GOLD: operator confirms the digits against `pages/pageN.png`.
- EXCLUDED → CANDIDATE: zoom re-read (page-3 rows, page-2 depreciation, page-1 marks) or Japanese reader (prose, farm name); page-1 `i.i` needs operator decision on the 8,264,854 / 4,457,075 conflict.

# Provenance + confidence notes — sample.pdf gold

Metadata *about* the reference. Kept out of the scored files on purpose.

## Provenance
- Source: `sample.pdf`, sha256 `3c5089cf1a79c5748bb4aba2f7cf73dcc840206ebcce23016d3c9f80aa9b0391`, 3-page CCITT-G4 B&W scan (Apeos C5570), no text layer.
- Pages extracted to PNG (`../pages/page{1,2,3}.png`) by wrapping raw G4 streams in TIFF → Pillow.
- Values read by Claude (Opus) at full-res zoom crops.
- **Epistemic status: SILVER.** Made by a model from the same family as bake-off competitors → not an external oracle. Operator verification promotes CANDIDATE→GOLD; a Japanese reader is required for prose/kanji.

## How the high-confidence numbers were locked
- Form arithmetic cross-validates the totals:
  - `①1947075 + ②1510000 + ③1000000 = ④4457075` ✓
  - `④ = ⑦ (income total)`; `⑤ = ⑥ = 0` ✓
  - `③ = 補助金500000 + 収入保険料補てん金500000` ✓
  - `①` identical on page 2 and page 3 ✓
- Arithmetic agreement is what upgrades a messy-handwriting read to GOLD.

## Known errors / open items
- Page-3 row breakdown sum (1,847,075) ≠ ① total (1,947,075) → one row misread. Rows EXCLUDED.
- page2 `r.46` — value 3,334,984 legible, but row 46 vs 47 placement ambiguous.
- **Expense reconciliation gap**: read expenses ⑨27,685 + ⑪12,638 + ⑭78,730 = 119,053, but ⑦4,457,075 − 所得3,334,984 ⇒ ~1,122,091 expected. ~1.0M of expenses unseen (減価償却費? blank at this read) → page2 `r.46` cannot be trusted as GOLD.
- All other page-2 expense rows (⑧⑩⑫⑬⑮⑯⑰, middle column ⑱–㉝) read as blank.
- Farm name (kana) unresolved.

## Page 1 (FA2204 第一表)
- Near-blank form. Only marks: deduction rows ⑰/⑱ show circled `0`s; 収入金額等・農業 ⑧ reads ~8,264,854.
- 8,264,854 **conflicts** with page-2 income 計⑦ 4,457,075 → single unreliable read, EXCLUDED.
- No arithmetic on page 1 cross-validates → no page-1 GOLD.

## Document map
| Page | Form | Title |
|---|---|---|
| 1 | FA2204 | 確定申告書 第一表 |
| 2 | FA3100 | 青色申告決算書（農業所得用）損益計算書 |
| 3 | FA3125 | 収入金額の内訳 |

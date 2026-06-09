# Full-page transcription — sample.pdf (all cells, blanks included)

Human-readable render of `cells/page{1,2,3}.cells.json`. Status: `filled` / **EMPTY** (confirmed blank) / *unread* (not legibly transcribed). Machine form in the JSON; this is the "see everything" view.

## Page 1 — FA2204 確定申告書 第一表 (near-blank)
| No | 科目 (label) | English | Value | Status |
|---|---|---|---|---|
| — | 住所 | address | 佐久市下越289 | filled |
| — | 氏名 | name | 髙橋大介 | filled |
| ㋑ | 事業 農業（収入） | agriculture revenue | 8,264,854 | filled (L — conflicts pg2, excluded) |
| ② | 所得金額 農業 | agriculture income | — | *unread* (expect 2,684,984) |
| ⑰⑱ ㉓㉔ | 各種控除 | deductions | — | *unread* (ambiguous 0000 marks) |
| ㉚–㊼ | 税金の計算 | tax calc | — | *unread* (essentially blank) |
| (60) | 青色申告特別控除額 | blue-return deduction | 650,000 | filled |
| (rest) | other fields | — | — | EMPTY / unread |

## Page 2 — FA3100 青色申告決算書 損益計算書
### Header
| Field | Value | Status |
|---|---|---|
| 住所 address | 佐久市下越289 | filled |
| 業種名 business | 農業 | filled |
| 農園名 farm name | — | *unread* (kana unclear) |
| フリガナ reading | タカハシ ダイスケ | filled |
| 氏名 name | 髙橋大介 | filled |
| 電話 phone | 095-2853-4416 | filled |

### 収入金額 (income) — all cross-validated
| No | 科目 | English | Value | Status |
|---|---|---|---|---|
| ① | 販売金額 | sales | 1,947,075 | filled |
| ② | 家事消費・事業消費金額 | home/business consumption | 1,510,000 | filled |
| ③ | 雑収入 | misc income | 1,000,000 | filled |
| ④ | 小計（①+②+③） | subtotal | 4,457,075 | filled |
| ⑤ | 棚卸高 期首 | opening inventory | 0 | filled |
| ⑥ | 棚卸高 期末 | closing inventory | 0 | filled |
| ⑦ | 計（④−⑤+⑥） | income total | 4,457,075 | filled |

### 経費 (expenses)
| No | 科目 | English | Value | Status |
|---|---|---|---|---|
| ⑧ | 租税公課 | taxes & dues | — | EMPTY |
| ⑨ | 種苗費 | seeds | 27,685 | filled |
| ⑩ | 素畜費 | livestock | — | EMPTY |
| ⑪ | 肥料費 | fertilizer | 12,638 | filled |
| ⑫ | 飼料費 | feed | — | EMPTY |
| ⑬ | 農具費 | farm tools | — | EMPTY |
| ⑭ | 農薬衛生費 | pesticide/hygiene | 78,730 | filled |
| ⑮–⑲ ㉑–㉝ | 諸費目 | other expense lines | — | EMPTY |
| ⑳ | 減価償却費 | depreciation | — | *unread* (likely nonzero) |
| ㉛/㉟ | 小計 / 経費計 | expense subtotal/total | — | *unread* (~1,122,091) |

### 所得 (result)
| No | 科目 | English | Value | Status |
|---|---|---|---|---|
| ㊵ | 繰戻額等 計 | reserves reversed | 0 | filled |
| ㊶ | 専従者給与 | family salary | — | EMPTY |
| ㊻ | 青色申告特別控除前の所得金額 | income before deduction | 3,334,984 | filled (does not reconcile) |
| ㊼ | 青色申告特別控除額 | blue deduction | — | *unread* (expect 650,000) |
| ㊽ | 所得金額（㊻−㊼） | income | — | *unread* (expect 2,684,984) |

## Page 3 — FA3125 収入金額の内訳
### (A) 収入金額の内訳 — rows EXCLUDED (sum-check fails), totals GOLD
| 区分 category | 作付面積 area (a) | 販売金額 sales | 家事消費 | Status |
|---|---|---|---|---|
| 根菜類 root veg | 150 | 947,075 | 1,510,000 | filled (L) |
| 葉菜類 leaf veg | 150 | 200,000 | 0 | filled (L) |
| 果菜類 fruit veg | 250 | 200,000 | 0 | filled (L) |
| 軟弱野菜 soft veg | — | 200,000 | 0 | filled (L) |
| 杯橋? | 450 | 300,000 | 0 | filled (L) |
| ① 合計 | — | **1,947,075** | **1,510,000** | filled (H) |
| check | sum rows = 1,847,075 ≠ 1,947,075 → one row misread |

### (右) 雑収入 misc income
| 区分 | 金額 | Status |
|---|---|---|
| 補助金 subsidy | 500,000 | filled |
| 収入保険料補てん金 insurance comp | 500,000 | filled |
| ③ 合計 | 1,000,000 | filled |

### (B) 農産物以外の棚卸高 — all EMPTY · (C) 雇人費 ㉒ = 0 · (D) 専従者給与 — EMPTY

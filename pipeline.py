"""Build catalog.json and rules.json from retail invoice lines.

Expects a CSV in the UCI Online Retail shape (one row per invoice line):
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.
Column names are matched loosely. Cancellations, non-positive quantities and
non-product codes such as POST are dropped. Association rules are mined with
Apriori and written in the same schema the app already loads.

Example:
    python pipeline.py transactions.csv --out-rules rules.json --out-catalog catalog.json
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from engine import NON_PRODUCT

COLUMN_ALIASES = {
    "invoice": ("invoiceno", "invoice", "invoice_no", "invoicenumber", "order_id", "orderid"),
    "stock": ("stockcode", "stock_code", "sku", "code", "product_code", "productcode"),
    "description": ("description", "name", "product", "product_name", "productname"),
    "quantity": ("quantity", "qty"),
    "price": ("unitprice", "unit_price", "price"),
}


def _normalise_header(name):
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")


def _column_map(fieldnames):
    lookup = {_normalise_header(name): name for name in fieldnames}
    mapping = {}
    for role, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                mapping[role] = lookup[alias]
                break
    missing = [role for role in ("invoice", "stock") if role not in mapping]
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing)
            + ". Found: " + ", ".join(fieldnames)
        )
    return mapping


def _float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        columns = _column_map(reader.fieldnames)
        rows = []
        for raw in reader:
            rows.append({
                "invoice": (raw.get(columns["invoice"]) or "").strip(),
                "stock": (raw.get(columns["stock"]) or "").strip(),
                "description": (raw.get(columns["description"]) or "").strip() if "description" in columns else "",
                "quantity": _float(raw.get(columns["quantity"])) if "quantity" in columns else None,
                "price": _float(raw.get(columns["price"])) if "price" in columns else None,
            })
        return rows


def clean_transactions(rows):
    """Group stock codes by invoice. Returns (baskets, description counts)."""
    descriptions = defaultdict(Counter)
    baskets = defaultdict(set)
    kept = 0
    for row in rows:
        invoice = row["invoice"]
        stock = row["stock"].upper()
        if not invoice or not stock:
            continue
        if invoice.upper().startswith("C"):
            continue
        if row["quantity"] is not None and row["quantity"] <= 0:
            continue
        if row["price"] is not None and row["price"] <= 0:
            continue
        if stock in NON_PRODUCT:
            continue
        baskets[invoice].add(stock)
        if row["description"]:
            descriptions[stock][row["description"]] += 1
        kept += 1
    transactions = [frozenset(items) for items in baskets.values() if items]
    return transactions, descriptions, kept


def _join_candidates(previous, size):
    ordered = [tuple(sorted(itemset)) for itemset in previous]
    ordered.sort()
    known = set(ordered)
    candidates = []
    seen = set()
    prefix = size - 2
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            a = ordered[left]
            b = ordered[right]
            if a[:prefix] != b[:prefix]:
                break
            merged = tuple(sorted(set(a) | set(b)))
            if len(merged) != size or merged in seen:
                continue
            if all(merged[:index] + merged[index + 1:] in known for index in range(size)):
                seen.add(merged)
                candidates.append(frozenset(merged))
    return candidates


def frequent_itemsets(transactions, min_support, max_size=4):
    count_needed = max(1, math.ceil(min_support * len(transactions))) if transactions else 1
    item_counts = Counter()
    for basket in transactions:
        item_counts.update(basket)
    frequent = {item for item, count in item_counts.items() if count >= count_needed}
    baskets = []
    for basket in transactions:
        reduced = frozenset(item for item in basket if item in frequent)
        if reduced:
            baskets.append(reduced)

    counts = {frozenset([item]): item_counts[item] for item in frequent}
    current = set(counts)
    size = 2
    while current and size <= max_size:
        if size == 2:
            pair_counts = Counter()
            for basket in baskets:
                items = sorted(basket)
                for pair in combinations(items, 2):
                    pair_counts[pair] += 1
            current = {
                frozenset(pair): count
                for pair, count in pair_counts.items()
                if count >= count_needed
            }
        else:
            candidates = _join_candidates(current, size)
            candidate_set = set(candidates)
            found = Counter()
            if candidate_set:
                for basket in baskets:
                    if len(basket) < size:
                        continue
                    for combo in combinations(sorted(basket), size):
                        itemset = frozenset(combo)
                        if itemset in candidate_set:
                            found[itemset] += 1
            current = {itemset: count for itemset, count in found.items() if count >= count_needed}
        counts.update(current)
        size += 1
    return counts


def generate_rules(counts, transaction_count, min_confidence, min_lift, top_n):
    rules = []
    for itemset, both in counts.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        support_both = both / transaction_count
        for width in range(1, len(items)):
            for chosen in combinations(items, width):
                antecedent = frozenset(chosen)
                consequent = itemset - antecedent
                confidence = both / counts[antecedent]
                if confidence < min_confidence:
                    continue
                support_consequent = counts[consequent] / transaction_count
                support_antecedent = counts[antecedent] / transaction_count
                lift = support_both / (support_antecedent * support_consequent)
                if lift < min_lift:
                    continue
                rules.append({
                    "antecedents": sorted(antecedent),
                    "consequents": sorted(consequent),
                    "support": round(support_both, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 2),
                })
    rules.sort(key=lambda rule: (-rule["lift"], -rule["confidence"], -rule["support"]))
    # The same split can be reached from only one itemset, but rounding can
    # still leave duplicates if counts collide. Keep the strongest copy.
    unique = []
    seen = set()
    for rule in rules:
        key = (tuple(rule["antecedents"]), tuple(rule["consequents"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(rule)
        if len(unique) >= top_n:
            break
    return unique


def catalogue_from_descriptions(descriptions):
    items = []
    for code in sorted(descriptions):
        name, _ = descriptions[code].most_common(1)[0]
        items.append({"code": code, "name": name})
    return items


def build(path, min_support, min_confidence, min_lift, top_n, max_size):
    rows = load_rows(Path(path))
    transactions, descriptions, kept_lines = clean_transactions(rows)
    if not transactions:
        raise ValueError("No transactions left after cleaning.")
    counts = frequent_itemsets(transactions, min_support, max_size=max_size)
    rules = generate_rules(counts, len(transactions), min_confidence, min_lift, top_n)
    return {
        "invoices": len(transactions),
        "lines_kept": kept_lines,
        "frequent_itemsets": len(counts),
        "rules": rules,
        "catalog": catalogue_from_descriptions(descriptions),
    }


def main():
    parser = argparse.ArgumentParser(description="Mine association rules from invoice lines.")
    parser.add_argument("transactions", type=Path, help="CSV of invoice lines")
    parser.add_argument("--out-rules", type=Path, default=Path("rules.json"))
    parser.add_argument("--out-catalog", type=Path, default=None)
    parser.add_argument("--min-support", type=float, default=0.015)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-lift", type=float, default=1.2)
    parser.add_argument("--top", type=int, default=500)
    parser.add_argument("--max-itemset", type=int, default=4)
    args = parser.parse_args()

    result = build(
        args.transactions,
        args.min_support,
        args.min_confidence,
        args.min_lift,
        args.top,
        args.max_itemset,
    )
    args.out_rules.write_text(json.dumps(result["rules"]), encoding="utf-8")
    if args.out_catalog is not None:
        args.out_catalog.write_text(json.dumps(result["catalog"]), encoding="utf-8")
    print(
        f"Invoices: {result['invoices']}. Lines kept: {result['lines_kept']}. "
        f"Frequent itemsets: {result['frequent_itemsets']}. "
        f"Rules written: {len(result['rules'])} -> {args.out_rules}"
    )


if __name__ == "__main__":
    main()

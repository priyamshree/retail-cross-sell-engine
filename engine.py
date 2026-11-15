"""Recommendation engine for mined association rules.

A rule is directly applicable only when every antecedent is already present.
Ranking aggregates every such rule for a product instead of keeping the first
match and sorting by lift alone.

Score for one rule:
    confidence * log(1 + lift) * support_weight * specificity
where support_weight rises with support and specificity rises with the number
of matched antecedents. A product's score is the sum of those rule scores.
"""

import math
import re
from collections import Counter, defaultdict

NON_PRODUCT = {
    "POST", "DOT", "M", "D", "S", "B",
    "AMAZONFEE", "BANK CHARGES", "CRUK", "PADS",
}

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "with", "for", "to",
    "set", "pack", "packs", "assorted", "design", "designs",
}

MIN_SIMILARITY = 0.18
TOKEN_RE = re.compile(r"[a-z0-9]+")


def rule_strength(rule):
    antecedents = len(rule["antecedents"])
    support_weight = 0.35 + 0.65 * min(rule["support"] / 0.05, 1.0)
    specificity = 1 + 0.2 * (antecedents - 1)
    return rule["confidence"] * math.log1p(rule["lift"]) * support_weight * specificity


def _tokens(text):
    return [
        tok for tok in TOKEN_RE.findall(text.lower())
        if tok not in STOPWORDS and len(tok) > 1 and not tok.isdigit()
    ]


class TfidfIndex:
    """Cosine similarity over product-name tokens. Built once at startup."""

    def __init__(self, items):
        self.items = items
        docs = [_tokens(item["name"]) for item in items]
        df = Counter()
        for doc in docs:
            df.update(set(doc))
        n = max(len(docs), 1)
        self.idf = {tok: math.log((1 + n) / (1 + freq)) + 1 for tok, freq in df.items()}
        self.vectors = []
        self.norms = []
        for doc in docs:
            tf = Counter(doc)
            length = sum(tf.values()) or 1
            vec = {tok: (count / length) * self.idf[tok] for tok, count in tf.items()}
            norm = math.sqrt(sum(weight * weight for weight in vec.values())) or 1.0
            self.vectors.append(vec)
            self.norms.append(norm)

    def search(self, text, limit=12, exclude=()):
        counts = Counter(_tokens(text))
        if not counts:
            return []
        known = {tok: count for tok, count in counts.items() if tok in self.idf}
        if not known:
            return []
        length = sum(known.values())
        query = {tok: (count / length) * self.idf[tok] for tok, count in known.items()}
        qnorm = math.sqrt(sum(weight * weight for weight in query.values())) or 1.0
        skip = set(exclude)
        scored = []
        for index, vec in enumerate(self.vectors):
            code = self.items[index]["code"]
            if code in skip:
                continue
            dot = sum(query[tok] * vec[tok] for tok in query if tok in vec)
            if dot <= 0:
                continue
            similarity = dot / (qnorm * self.norms[index])
            if similarity < MIN_SIMILARITY:
                continue
            scored.append((similarity, index))
        scored.sort(key=lambda pair: (-pair[0], self.items[pair[1]]["name"]))
        return scored[:limit]


class CatalogueEngine:
    def __init__(self, catalog, rules):
        self.catalog = list(catalog)
        self.rules = []
        self.code_to_name = {}
        self.codes = set()
        for item in self.catalog:
            code = str(item["code"])
            self.code_to_name[code] = item["name"]
            self.codes.add(code)

        self.by_item = defaultdict(list)
        self.direct_by_item = defaultdict(list)
        in_rules = set()
        for index, raw in enumerate(rules):
            antecedents = [str(code) for code in raw["antecedents"]]
            consequents = [str(code) for code in raw["consequents"]]
            rule = {
                "id": index,
                "antecedents": antecedents,
                "consequents": consequents,
                "ant": frozenset(antecedents),
                "support": float(raw["support"]),
                "confidence": float(raw["confidence"]),
                "lift": float(raw["lift"]),
            }
            rule["strength"] = rule_strength(rule)
            self.rules.append(rule)
            in_rules.update(antecedents)
            in_rules.update(consequents)
            for code in antecedents:
                self.by_item[code].append(rule)
            if len(antecedents) == 1:
                self.direct_by_item[antecedents[0]].append(rule)

        self.in_rules = in_rules
        self.covered_codes = self.codes & in_rules
        self.index = TfidfIndex(self.catalog)

    def label(self, code):
        return self.code_to_name.get(code, code).strip()

    def _public_item(self, code):
        return {"code": code, "name": self.label(code)}

    def stats(self):
        catalogue_size = len(self.catalog)
        covered = len(self.covered_codes)
        return {
            "catalog_size": catalogue_size,
            "rule_count": len(self.rules),
            "products_in_rules": covered,
            "products_without_rules": catalogue_size - covered,
            "coverage_pct": round(100 * covered / catalogue_size, 1) if catalogue_size else 0,
        }

    def catalog_public(self):
        result = []
        for item in self.catalog:
            code = str(item["code"])
            if code in NON_PRODUCT:
                continue
            result.append({
                "code": item["code"],
                "name": self.label(code),
                "covered": code in self.covered_codes,
                "direct_count": len(self.direct_by_item.get(code, [])),
                "rule_count": len(self.by_item.get(code, [])),
            })
        return result

    def _recommendable(self, code, held):
        return code not in held and code not in NON_PRODUCT and code in self.codes

    def _passes(self, rule, floors):
        return (
            rule["support"] >= floors["min_support"]
            and rule["confidence"] >= floors["min_confidence"]
            and rule["lift"] >= floors["min_lift"]
        )

    def _explain(self, slot):
        support = f"{slot['support'] * 100:.1f}%"
        confidence = f"{slot['confidence'] * 100:.0f}%"
        lift = slot["lift"]
        metrics = f"The strongest match has support {support}, confidence {confidence} and lift {lift}."
        if slot.get("via"):
            sources = ", ".join(via["name"] for via in slot["via"][:3])
            return (
                f"{slot['name']} is suggested because shoppers who bought {sources} also bought it, "
                f"and those names are the closest catalogue matches. {metrics}"
            )
        others = ", ".join(slot["because"])
        count = slot["rule_count"]
        word = "rule" if count == 1 else "rules"
        return (
            f"{slot['name']} is recommended because it appears in baskets that also contain {others}. "
            f"{count} association {word} support that pattern. {metrics}"
        )

    def _aggregate(self, contributions):
        """contributions: iterable of (rule, weight, via or None)."""
        buckets = {}
        for rule, weight, via in contributions:
            if weight <= 0:
                continue
            held = set(rule["antecedents"])
            because = [self.label(code) for code in rule["antecedents"]]
            for code in rule["consequents"]:
                if not self._recommendable(code, held):
                    continue
                slot = buckets.get(code)
                if slot is None:
                    slot = {
                        "code": code,
                        "name": self.label(code),
                        "score": 0.0,
                        "rule_count": 0,
                        "confidence": 0.0,
                        "lift": 0.0,
                        "support": 0.0,
                        "because": [],
                        "applicable": via is None,
                        "via": [],
                        "_best": -1.0,
                        "_seen_via": set(),
                    }
                    buckets[code] = slot
                slot["score"] += weight
                slot["rule_count"] += 1
                if weight > slot["_best"]:
                    slot["_best"] = weight
                    slot["confidence"] = rule["confidence"]
                    slot["lift"] = rule["lift"]
                    slot["support"] = rule["support"]
                    slot["because"] = because
                if via is not None and via["code"] not in slot["_seen_via"]:
                    slot["_seen_via"].add(via["code"])
                    slot["via"].append(via)
                    slot["applicable"] = False

        ranked = []
        for slot in buckets.values():
            slot.pop("_best")
            slot.pop("_seen_via")
            slot["score"] = round(slot["score"], 4)
            slot["confidence"] = round(slot["confidence"], 4)
            slot["lift"] = round(slot["lift"], 2)
            slot["support"] = round(slot["support"], 4)
            if not slot["via"]:
                slot.pop("via")
            slot["explanation"] = self._explain(slot)
            ranked.append(slot)
        ranked.sort(key=lambda rec: (-rec["score"], -rec["confidence"], -rec["lift"], rec["name"]))
        return ranked

    def _satisfied(self, basket, floors):
        seen = set()
        contributions = []
        for code in basket:
            for rule in self.by_item.get(code, []):
                if rule["id"] in seen or not rule["ant"].issubset(basket):
                    continue
                if not self._passes(rule, floors):
                    continue
                seen.add(rule["id"])
                contributions.append((rule, rule["strength"], None))
        return contributions

    def recommend(self, codes, limit=10, floors=None):
        floors = floors or {"min_support": 0, "min_confidence": 0, "min_lift": 0}
        basket = []
        unknown = []
        seen = set()
        for code in codes:
            if code in seen:
                continue
            seen.add(code)
            if code not in self.codes:
                unknown.append(code)
            else:
                basket.append(code)
        held = set(basket)
        applied = self._satisfied(held, floors)
        recommendations = [
            rec for rec in self._aggregate(applied)
            if rec["code"] not in held
        ][:limit]

        almost = []
        for rule in self.rules:
            if not self._passes(rule, floors):
                continue
            overlap = rule["ant"] & held
            missing = rule["ant"] - held
            if len(missing) != 1 or not overlap:
                continue
            target = next(iter(missing))
            if target in NON_PRODUCT or target not in self.codes:
                continue
            consequents = [
                self._public_item(code)
                for code in rule["consequents"]
                if self._recommendable(code, held | {target})
            ]
            if not consequents:
                continue
            almost.append({
                "missing": self._public_item(target),
                "recommend": consequents,
                "because": [self.label(code) for code in rule["antecedents"]],
                "confidence": round(rule["confidence"], 4),
                "lift": round(rule["lift"], 2),
                "support": round(rule["support"], 4),
                "applicable": False,
                "score": round(rule["strength"], 4),
                "explanation": (
                    f"Not ready to recommend yet. Add {self.label(target)} and the rule "
                    f"would suggest {', '.join(item['name'] for item in consequents)} "
                    f"at support {rule['support'] * 100:.1f}%, confidence {rule['confidence'] * 100:.0f}% "
                    f"and lift {round(rule['lift'], 2)}."
                ),
            })

        already = {rec["code"] for rec in recommendations}
        almost = [
            row for row in almost
            if any(item["code"] not in already for item in row["recommend"])
        ]
        almost.sort(key=lambda row: (-row["score"], -row["confidence"]))
        deduped = []
        seen_missing = set()
        for row in almost:
            if row["missing"]["code"] in seen_missing:
                continue
            seen_missing.add(row["missing"]["code"])
            deduped.append(row)
            if len(deduped) == 5:
                break

        return {
            "recommendations": recommendations,
            "almost": deduped,
            "uncovered": [
                self._public_item(code) for code in basket if code not in self.covered_codes
            ],
            "unknown": unknown,
            "rules_applied": len(applied),
        }

    def _conditional_rows(self, code, floors, limit=6):
        rows = []
        for rule in self.by_item.get(code, []):
            if len(rule["antecedents"]) < 2 or not self._passes(rule, floors):
                continue
            also = [self._public_item(other) for other in rule["antecedents"] if other != code]
            recommend = [
                self._public_item(item)
                for item in rule["consequents"]
                if item not in NON_PRODUCT and item in self.codes
            ]
            if not recommend:
                continue
            rows.append({
                "also_requires": also,
                "recommend": recommend,
                "if_has": [self._public_item(item) for item in rule["antecedents"]],
                "confidence": round(rule["confidence"], 4),
                "lift": round(rule["lift"], 2),
                "support": round(rule["support"], 4),
                "score": round(rule["strength"], 4),
                "applicable": False,
                "explanation": (
                    "This is not a recommendation on its own. It applies only when the basket also contains "
                    + ", ".join(item["name"] for item in also)
                    + f", at support {rule['support'] * 100:.1f}%, confidence {rule['confidence'] * 100:.0f}% "
                    + f"and lift {round(rule['lift'], 2)}."
                ),
            })
        rows.sort(key=lambda row: (-row["score"], -row["confidence"]))
        return rows[:limit]

    def _fallback(self, code, floors, limit=6):
        matches = self.index.search(self.label(code), limit=8, exclude={code})
        contributions = []
        for similarity, index in matches:
            item = self.index.items[index]
            if item["code"] not in self.direct_by_item:
                continue
            via = {
                "code": item["code"],
                "name": self.label(item["code"]),
                "similarity": round(similarity, 3),
            }
            for rule in self.direct_by_item[item["code"]]:
                if self._passes(rule, floors):
                    contributions.append((rule, rule["strength"] * similarity, via))
        ranked = []
        for rec in self._aggregate(contributions):
            if rec["code"] == code:
                continue
            rec["source"] = "similar_product"
            ranked.append(rec)
            if len(ranked) == limit:
                break
        return ranked

    def product(self, code, limit=8, floors=None):
        floors = floors or {"min_support": 0, "min_confidence": 0, "min_lift": 0}
        if code not in self.codes:
            return None
        direct_rules = [
            rule for rule in self.direct_by_item.get(code, [])
            if self._passes(rule, floors)
        ]
        direct = self._aggregate((rule, rule["strength"], None) for rule in direct_rules)[:limit]
        for rec in direct:
            rec["applicable"] = True
        return {
            "code": code,
            "name": self.label(code),
            "covered": code in self.covered_codes,
            "has_direct": bool(direct),
            "direct": direct,
            "conditional": self._conditional_rows(code, floors),
            "fallback": [] if direct else self._fallback(code, floors),
        }

    def similar(self, query, limit=12, floors=None):
        floors = floors or {"min_support": 0, "min_confidence": 0, "min_lift": 0}
        text = (query or "").strip()
        if not text:
            return {"query": "", "matches": [], "suggestions": [], "bundle": [], "bundle_explanation": ""}
        found = self.index.search(text, limit=limit)
        matches = []
        contributions = []
        for similarity, index in found:
            item = self.index.items[index]
            covered = item["code"] in self.covered_codes
            matches.append({
                "code": item["code"],
                "name": self.label(item["code"]),
                "similarity": round(similarity, 3),
                "covered": covered,
            })
            if item["code"] not in self.direct_by_item:
                continue
            via = {
                "code": item["code"],
                "name": self.label(item["code"]),
                "similarity": round(similarity, 3),
            }
            for rule in self.direct_by_item[item["code"]]:
                if self._passes(rule, floors):
                    contributions.append((rule, rule["strength"] * similarity, via))
        suggestions = []
        for rec in self._aggregate(contributions):
            suggestions.append(rec)
            if len(suggestions) == 6:
                break
        bundle = []
        seen = set()
        for match in matches:
            if not match["covered"] or match["code"] in seen:
                continue
            bundle.append({**match, "role": "similar"})
            seen.add(match["code"])
            if sum(item["role"] == "similar" for item in bundle) == 2:
                break
        for rec in suggestions:
            if rec["code"] in seen:
                continue
            bundle.append({
                "code": rec["code"],
                "name": rec["name"],
                "role": "complement",
                "explanation": rec["explanation"],
            })
            seen.add(rec["code"])
            if sum(item["role"] == "complement" for item in bundle) == 3:
                break
        return {
            "query": text,
            "matches": matches,
            "suggestions": suggestions,
            "bundle": bundle,
            "bundle_explanation": (
                "The bundle pairs the closest existing products with complements those products are bought with. "
                "Similarity is TF-IDF on the name and description. Complements come from direct association rules."
            ) if bundle else "",
        }

    def analytics(self):
        rules = self.rules
        supports = sorted(rule["support"] for rule in rules)
        confidences = [rule["confidence"] for rule in rules]
        lifts = [rule["lift"] for rule in rules]
        count = len(rules) or 1
        mid = len(supports) // 2
        if not supports:
            median_support = 0
        elif len(supports) % 2:
            median_support = supports[mid]
        else:
            median_support = (supports[mid - 1] + supports[mid]) / 2
        multi = sum(1 for rule in rules if len(rule["antecedents"]) > 1)
        direct_products = {
            code for code in self.direct_by_item
            if code in self.codes and code not in NON_PRODUCT
        }
        edges = [1, 2, 3, 5, 8, 12, 20]
        histogram = []
        for index, low in enumerate(edges):
            high = edges[index + 1] if index + 1 < len(edges) else None
            if high is None:
                bucket = sum(1 for lift in lifts if lift >= low)
                label = f"{low}+"
            else:
                bucket = sum(1 for lift in lifts if low <= lift < high)
                label = f"{low}–{high}"
            histogram.append({"label": label, "count": bucket})

        top_rules = []
        for rule in sorted(rules, key=lambda item: -item["strength"])[:12]:
            top_rules.append({
                "antecedent_codes": list(rule["antecedents"]),
                "consequent_codes": list(rule["consequents"]),
                "antecedents": [self.label(code) for code in rule["antecedents"]],
                "consequents": [self.label(code) for code in rule["consequents"]],
                "support": round(rule["support"], 4),
                "confidence": round(rule["confidence"], 4),
                "lift": round(rule["lift"], 2),
                "score": round(rule["strength"], 4),
            })

        catalogue_size = len(self.catalog)
        covered = len(self.covered_codes)
        return {
            "source": (
                "Figures describe the stored association rules only. "
                "Invoice counts, basket size, revenue, country and monthly sales are not in this file, "
                "so they are not shown. Precision, recall and Apriori versus FP-Growth timing need the original invoices."
            ),
            "kpis": {
                "catalog_size": catalogue_size,
                "rule_count": len(rules),
                "products_in_rules": covered,
                "products_with_direct_rules": len(direct_products),
                "products_without_rules": catalogue_size - covered,
                "coverage_pct": round(100 * covered / catalogue_size, 1) if catalogue_size else 0,
                "single_item_rules": len(rules) - multi,
                "multi_item_rules": multi,
                "mean_confidence": round(sum(confidences) / count, 4) if rules else 0,
                "mean_lift": round(sum(lifts) / count, 2) if rules else 0,
                "median_support": round(median_support, 4),
            },
            "lift_histogram": histogram,
            "scatter": [
                {
                    "support": round(rule["support"], 4),
                    "confidence": round(rule["confidence"], 4),
                    "lift": round(rule["lift"], 2),
                    "antecedents": [self.label(code) for code in rule["antecedents"]],
                    "consequents": [self.label(code) for code in rule["consequents"]],
                }
                for rule in rules
            ],
            "top_rules": top_rules,
            "network": self.network(),
        }

    def network(self, limit=16):
        chosen = sorted(self.rules, key=lambda rule: -rule["strength"])[:40]
        degree = {}
        edges = {}
        for rule in chosen:
            left = [code for code in rule["antecedents"] if code in self.codes and code not in NON_PRODUCT]
            right = [code for code in rule["consequents"] if code in self.codes and code not in NON_PRODUCT]
            for source in left:
                for target in right:
                    if source == target:
                        continue
                    key = tuple(sorted((source, target)))
                    current = edges.get(key)
                    if current is None or rule["lift"] > current["lift"]:
                        edges[key] = {
                            "source": source,
                            "target": target,
                            "lift": round(rule["lift"], 2),
                            "confidence": round(rule["confidence"], 4),
                        }
                    degree[source] = degree.get(source, 0) + 1
                    degree[target] = degree.get(target, 0) + 1
        keep = {code for code, _ in sorted(degree.items(), key=lambda item: -item[1])[:limit]}
        return {
            "nodes": [{"id": code, "name": self.label(code)} for code in sorted(keep, key=self.label)],
            "links": [
                edge for edge in edges.values()
                if edge["source"] in keep and edge["target"] in keep
            ],
        }

    def what_if(self, codes, limit=5):
        scenarios = (
            {"name": "Conservative", "min_confidence": 0.7, "min_support": 0.02, "min_lift": 5},
            {"name": "Balanced", "min_confidence": 0.5, "min_support": 0.015, "min_lift": 2},
            {"name": "Exploratory", "min_confidence": 0.3, "min_support": 0.015, "min_lift": 1},
        )
        compared = []
        for scenario in scenarios:
            floors = {
                "min_support": scenario["min_support"],
                "min_confidence": scenario["min_confidence"],
                "min_lift": scenario["min_lift"],
            }
            kept = sum(1 for rule in self.rules if self._passes(rule, floors))
            recommendations = []
            if codes:
                recommendations = self.recommend(codes, limit=limit, floors=floors)["recommendations"]
            compared.append({
                **scenario,
                "rules_kept": kept,
                "rules_total": len(self.rules),
                "recommendations": recommendations,
            })
        return {
            "note": (
                "Each count is how many stored rules pass that cutoff. "
                "The rules are not mined again."
            ),
            "scenarios": compared,
        }

    def presets(self):
        """Curated sample baskets and featured products for one-click UI testing."""
        sample_baskets = [
            {
                "title": "Regency Tea Party",
                "subtitle": "Triggers high-lift multi-item tea set & cakestand rules",
                "codes": ["22697", "22423"],
            },
            {
                "title": "Retrospot Lunch Bundle",
                "subtitle": "Triggers overlapping lunch bag & storage rules",
                "codes": ["20725", "20727"],
            },
            {
                "title": "Bakelike Alarm Clocks",
                "subtitle": "Triggers colour-variant cross-sell rules",
                "codes": ["22726", "22727"],
            },
            {
                "title": "Single Item (Needs 1 More)",
                "subtitle": "Demonstrates 'One Item Away' conditional rule unlocking",
                "codes": ["22423"],
            },
        ]
        valid_baskets = []
        for b in sample_baskets:
            items = [self._public_item(c) for c in b["codes"] if c in self.codes]
            if items:
                valid_baskets.append({
                    "title": b["title"],
                    "subtitle": b["subtitle"],
                    "items": items,
                })
        return {"baskets": valid_baskets}
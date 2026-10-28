# GEMS Prize — cash eligibility, commercial-intent and submission cutoff preflight

**Checked 2026-10-28 against the [DOE/NLR September 2026 official rules](https://www.nlr.gov/docs/fy26osti/96647.pdf) and [live DrivenData competition page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/966/).**
**Status: BLOCKED FOR CASH-ELIGIBILITY CERTIFICATION; no registration, rule acceptance, application or submission is authorized by this document.**
This is a factual preparation aid, not legal advice or an assertion about Sven Hardy Benson's citizenship, residence, business registration, personal affiliations, or actual commercial plans.

## Why this changes the pursuit decision

The repo has substantial existing geospatial ML and reproducibility code, but **technical performance does not establish cash-prize eligibility**. Section 1.3 restricts the types of competitors who may win. Section 1.4 also requires a U.S.-performed/U.S.-benefiting solution and a commercialization statement. Registration and submission involve a certification under penalty of perjury. The prior Opps registry phrase "High if rules accepted" was too permissive: merely accepting rules does not establish eligibility.

### Required facts and evidence — applicant-controlled, not inferred

| Gate | Primary rule | Evidence to verify privately | Current evidence / decision |
|---|---|---|---|
| Competing identity | §1.3 | Choose exactly one registered competitor: individual, team (captain), or private entity; match DrivenData account and award payee | **UNKNOWN**; a GitHub organization name is not a legal entity |
| Individual eligibility | §1.3 | If individual, proof of U.S. citizenship or permanent residency; legal age | **UNKNOWN**; never infer nationality from author name, location or GitHub |
| Team eligibility | §1.3 | If team, captain is U.S. citizen/permanent resident and other members legally authorized to work in U.S.; agreed prize split | **UNKNOWN**; do not invent team members |
| Entity eligibility | §1.3 | If private entity, U.S. incorporation **and** primary U.S. place of business; authorized signatory | **UNKNOWN**; no Svyable incorporation evidence reviewed |
| Exclusions | §1.3 | No disqualifying DOE/federal/FFRDC/DrivenData affiliation, debarment, restricted talent-program participation, foreign-country-of-concern control, or other listed exclusion | **UNKNOWN**; applicant must verify |
| U.S. activity | §1.4 | Contemporaneous evidence that a majority of activities supporting the submission occur in the U.S., plus a plausible U.S.-market benefit | **UNKNOWN**; cloud-region selection alone is not confirmed to satisfy the rule |
| Commercial intent | §1.4 | Genuine intent to commercialize early-stage technology and establish a viable U.S.-based business with revenue **not solely IP licensing** | **UNKNOWN**; existing open-source code alone is insufficient |
| Rights and verification | §§3.2, 3.5, A.4–A.5 | Rights to all training/external data and dependencies; shareable with organizers for independent verification; reproducible code; safe handling of public materials | **PARTIAL**: source/provenance processes exist; full external-data and contributor rights not certified |
| Generative-AI disclosure | §3.2 | Complete, truthful description of AI use across solution elements | **PARTIAL**: `docs/AI_DISCLOSURE.md` exists; reconcile against final actual solution |
| Winner payment | Appendix A.2 | Ability to provide W-9, ACH information and identity evidence within 30 days of notice | **UNKNOWN**; do not collect banking/tax details in public repo |
| Registration certification | §1.3, Appendix A.1 | Applicant reads exact live terms and personally authorizes any perjury certification and acceptance | **NOT AUTHORIZED** |

**No eligibility fact above should be represented as "yes" without real evidence.** Store private evidence outside this public repository; record only a non-sensitive verified status and review date if the applicant authorizes it.

## Two conflicting official deadline statements

- **Live DrivenData page:** December 3, 2026, **23:59 UTC**.
- **DOE/NLR official rules, Appendix A.1:** submission by **5:00 p.m. Eastern Time** on the prize deadline, which is **22:00 UTC** on December 3 (Eastern Standard Time).

These differ by **1 hour 59 minutes**. No organizer clarification was found in this check. For internal risk management, **plan to finalize before 22:00 UTC December 3**, and do not represent that planning cutoff as an organizer ruling. The applicant or organizer must resolve the discrepancy before relying on the later platform time. The single final selected submission is used for both award rounds; three scored submissions per rolling week are allowed.

## Cash-prize economics, not the $300,000 headline

Official awards: Phase 1 five × $10,000; Phase 2 $100,000 / $70,000 / $40,000 / $25,000 / $15,000. One selected submission is scored twice. There are no guaranteed payments. Sponsor: U.S. DOE Office of Geothermal, administered with NLR and DrivenData.

**Illustrative *conditional-on-eligibility* incremental screening models**, not calibrated winning odds. Sunk model development is excluded; remaining 3-hour compliance/evidence preflight costs $105 at an assumed $35/hour. Final-rank probabilities below are mutually exclusive; the Phase 1 top-five event can overlap a Phase 2 award:

| Assumption | Phase 1 probability | Phase 2 probabilities (1st / 2nd / 3rd / 4th / 5th) | Conditional expected gross prize | Net after $105 |
|---|---:|---|---:|---:|
| Cautious | 0.2% | 0.01% / 0.02% / 0.03% / 0.04% / 0.05% | $73.50 | **−$31.50** |
| Upside sensitivity | 1.0% | 0.05% / 0.05% / 0.10% / 0.15% / 0.20% | $292.50 | **+$187.50** |

Formula: `EV = P(eligible) × [10,000 × P(Phase 1 top-five) + sum(Phase 2 prize_i × P(rank_i))] − 105`. **P(eligible) cannot be responsibly assigned from public repository evidence**. If the actual applicant fails a mandatory rule, cash-prize probability is zero; do not spend more time modeling or submitting. If eligibility is confirmed, only fresh *held-out unknown-fault* evidence should justify higher placement odds. Current repo scores against known faults are not a reliable estimate of hidden-test rank.

## Commercialization evidence: a conditional outline, not a representation

Section 1.4 asks for a viable U.S.-based business with revenue not solely from licensing IP. If and only if this is an actual plan the applicant intends to pursue, a possible **existing-asset** path is: Svyable's GEMS geospatial ML pipeline produces reproducible candidate-fault screening and analyst QA packages for U.S. geothermal exploration workflows. Potential paid work would be per-project geospatial analysis, verification and model-assisted interpretation, rather than simply licensing model weights. This is a **hypothesis**, not a current product, paying customer, geology certification, signed commitment or evidence of revenue. It would require qualified geoscience review, rights clearance, commercial demand and a viable delivery budget. Do not paste it into a certification as if already true.

## Concrete next gates, in order

1. **Private applicant check** of identity route, citizenship/incorporation, exclusions, U.S. activities, commercial intent and award-payee ability. Stop if any mandatory gate fails; no guessed declarations.
2. **Official clarification** of the 22:00 vs 23:59 UTC deadline and any ambiguity about §1.4 commercialization or where activities must be performed. The [October 13 forum question](https://community.drivendata.org/t/eligibility-majority-of-activities-performed-in-the-u-s-rules-1-4-and-participating-without-prize-eligibility/11565) raises both issues but shows no organizer answer in the public page checked here. Do not contact organizers without approval.
3. **Technical proof**: on the authorized training host, recover real official-data held-out spatial, complete-fault and trace-completion scores, including independent comparison to a frozen baseline. The committed [experiment log](EXPERIMENT_LOG.md) records extensive code and a results-integrity audit, but also missing per-fold detail artifacts; do not claim a hidden-label score.
4. **Only after approval:** review live DrivenData terms, registration/certification, external-data rights, generative-AI disclosure, exact final GeoTIFF validator and selection of one submission. Never automate acceptance or certification.

## Source links

- [DOE/NLR official rules, 19 pages](https://www.nlr.gov/docs/fy26osti/96647.pdf): §1.3 eligibility (PDF pp. 5–7); §1.4 U.S. activity and commercialization (p. 7); §3.2 format and AI disclosure (p. 9); Appendix A.1 timing/certification (p. 13); Appendix A.2 payment (p. 14); Appendix A.4–A.5 publicity, IP and rights (pp. 14–16).
- [DrivenData live prize page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/966/): award breakdown and 23:59 UTC deadline.
- [Unanswered organizer-forum eligibility question](https://community.drivendata.org/t/eligibility-majority-of-activities-performed-in-the-u-s-rules-1-4-and-participating-without-prize-eligibility/11565).

**Decision 2026-10-28: WATCH for cash eligibility, not an authorized entry.** Preserve the technical work; do not make additional cash-award assumptions or pay for training until the applicant-specific gates are resolved.

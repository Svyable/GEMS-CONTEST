# Rules and compliance checklist

This is an engineering checklist, not a substitute for the official rules.

## Before competing

**Cash eligibility is NOT established by registration alone.** Read [the 2026-10-28 primary-rules eligibility and commercialization preflight](PRIZE_ELIGIBILITY_COMMERCIALIZATION_PREFLIGHT_2026-10-28.md) before accepting any rules or certifying eligibility. Section 1.3 requires a qualifying U.S. individual/team captain/entity, and Section 1.4 requires U.S.-performed work and genuine U.S. commercial intent. These are applicant-controlled facts, not public-repo inferences.


- [ ] Register on DrivenData and accept the GEMS rules/data terms.
- [ ] Verify §1.3 eligibility privately: U.S. citizen/permanent-resident individual or captain, or U.S.-incorporated entity with primary U.S. business; age, exclusions, team work authorization, and registered payee.
- [ ] Verify §1.4: majority of submission-supporting activities performed in the U.S., U.S. market benefit, and genuine U.S.-business commercialization intent beyond IP licensing.
- [ ] Confirm W-9/ACH award-payee readiness and understand that registration/submission involves a certification under penalty of perjury; only applicant may accept.
- [ ] Decide the official participating entity/team early enough to avoid final-submission ambiguity.
- [ ] Keep the official rules URL and competition/forum clarification pages in `docs/SOURCES.md` under version control.

## During development

- [ ] Do not redistribute organizer data unless the data terms permit it.
- [ ] Track source, version/date, license, and use rationale for every external dataset.
- [ ] Keep generative-AI use documented in `docs/AI_DISCLOSURE.md`; the rules explicitly allow it but require disclosure of extent/use in the submission narrative.
- [ ] Log exact code commit, config, data fingerprints, seed, CV score and leaderboard score for every submission.
- [ ] Do not treat public leaderboard labels/scores as a source of hidden ground truth.
- [ ] Keep code/assets reproducible and free of malware.

## Submission

**Deadline conflict:** DrivenData lists **2026-12-03 23:59 UTC**; official DOE/NLR rules Appendix A.1 says **5:00 p.m. ET = 22:00 UTC** that day. Plan to finish before the **earlier** time pending organizer clarification; this is an internal safety cutoff, not an official resolution.


- [ ] Run the template-based GeoTIFF validator.
- [ ] Confirm EPSG:32611, 100 m, one band, float32, values in [0,1].
- [ ] Confirm valid-data mask/bounds match the official template.
- [ ] Archive the exact artifact hash plus the code commit/config that generated it.
- [ ] Respect the scored-submission quota: the organizer says the three-submission allowance uses a rolling window rather than a calendar-week reset.
- [ ] Before the deadline, select exactly one final submission for evaluation across both prize rounds.

## If we become finalists

- [ ] Package complete code/assets and resource requirements.
- [ ] Verify a clean environment can reproduce the selected submission and run on new samples.
- [ ] Prepare documentation consistent with DrivenData's winning-model template.
- [ ] Review which submission materials are designated public before including proprietary/confidential information.
- [ ] Mark any permitted confidential/proprietary material exactly as required by the official rules.

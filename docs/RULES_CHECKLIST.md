# Rules and compliance checklist

This is an engineering checklist, not a substitute for the official rules.

## Before competing

- [ ] Register on DrivenData and accept the GEMS rules/data terms.
- [ ] Confirm the competing individual/team/entity satisfies the eligibility requirements.
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

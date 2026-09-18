# Agentic retrieval versus single-shot retrieval

> **Demonstration artifact.** This note is a demonstration artifact produced by an automated system from public documents. It is not a credit rating, not investment advice, and has not been reviewed by a rating committee. Every claim carries a citation to the source document and page; verify anything you intend to rely on.
> Figures quoted below are automated output published as engineering
> evidence, not as an opinion on any issuer.


Both arms share one index, one embedding model, one evidence cap, one
router, one set of specialists and one synthesis step. The only variable
is how evidence is found: the control issues the analyst's question
verbatim as a single search; the agentic arm plans sub-queries, reflects
on sufficiency and iterates.

Evidence cap for both arms: 10 chunks.

| question | arm | pages reached | findings | cost | calls | termination |
| --- | --- | --- | --- | --- | --- | --- |
| As at 30 June 2026, what are APSEZ's Net Gea... | agentic | 3 | 8 | $0.1592 | 6 | sufficient |
| As at 30 June 2026, what are APSEZ's Net Gea... | single_shot | 3 | 7 | $0.1280 | 4 | single_shot |
| What does AGEL's FY25 auditor's report flag ... | agentic | 10 | 7 | $0.0985 | 7 | iteration_ceiling |
| What does AGEL's FY25 auditor's report flag ... | single_shot | 6 | 7 | $0.0681 | 3 | single_shot |
| Compare the disclosed financing profiles of ... | agentic | 8 | 9 | $0.2126 | 8 | iteration_ceiling |
| Compare the disclosed financing profiles of ... | single_shot | 9 | 10 | $0.2190 | 4 | single_shot |

## Per question

### As at 30 June 2026, what are APSEZ's Net Gearing and DSCR, what covenant thresholds do they run against, and how much headroom remains?

- agentic reached: apsez-q1fy27 p.16, apsez-q1fy27 p.17, apsez-q1fy27 p.34
- single-shot reached: apsez-q1fy27 p.12, apsez-q1fy27 p.16, apsez-q1fy27 p.34
- **only the loop reached**: apsez-q1fy27 p.17
- **only single-shot reached**: apsez-q1fy27 p.12
- cost: agentic $0.1592 against single-shot $0.1280 (1.2x)

### What does AGEL's FY25 auditor's report flag as Emphasis of Matter and Key Audit Matters, and what related-party and contingent-liability exposures accompany them?

- agentic reached: agel-fy25 p.142, agel-fy25 p.182, agel-fy25 p.229, agel-fy25 p.242, agel-fy25 p.27, agel-fy25 p.302, agel-fy25 p.360, agel-fy25 p.4, agel-fy25 p.71, agel-fy25 p.97
- single-shot reached: agel-fy25 p.1, agel-fy25 p.136, agel-fy25 p.138, agel-fy25 p.142, agel-fy25 p.5, agel-fy25 p.9
- **only the loop reached**: agel-fy25 p.182, agel-fy25 p.229, agel-fy25 p.242, agel-fy25 p.27, agel-fy25 p.302, agel-fy25 p.360, agel-fy25 p.4, agel-fy25 p.71, agel-fy25 p.97
- **only single-shot reached**: agel-fy25 p.1, agel-fy25 p.136, agel-fy25 p.138, agel-fy25 p.5, agel-fy25 p.9
- cost: agentic $0.0985 against single-shot $0.0681 (1.4x)

### Compare the disclosed financing profiles of AGEL and APSEZ on debt maturity structure, interest-rate exposure and reliance on related-party or promoter funding, and say which shows more refinancing risk on the disclosed evidence.

- agentic reached: agel-fy25 p.240, agel-fy25 p.69, agel-fy25 p.71, apsez-q1fy27 p.12, apsez-q1fy27 p.17, apsez-q1fy27 p.24, apsez-q1fy27 p.30, apsez-q1fy27 p.33
- single-shot reached: agel-fy25 p.226, agel-fy25 p.69, agel-fy25 p.72, agel-fy25 p.8, apsez-q1fy27 p.13, apsez-q1fy27 p.17, apsez-q1fy27 p.21, apsez-q1fy27 p.28, apsez-q1fy27 p.33
- **only the loop reached**: agel-fy25 p.240, agel-fy25 p.71, apsez-q1fy27 p.12, apsez-q1fy27 p.24, apsez-q1fy27 p.30
- **only single-shot reached**: agel-fy25 p.226, agel-fy25 p.72, agel-fy25 p.8, apsez-q1fy27 p.13, apsez-q1fy27 p.21, apsez-q1fy27 p.28
- cost: agentic $0.2126 against single-shot $0.2190 (1.0x)

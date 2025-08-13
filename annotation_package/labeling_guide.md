# Email Priority Labeling Guide (Urgent / Normal / Low)

## Task
Assign exactly one label to each email (subject + first unquoted body): **Urgent**, **Normal**, or **Low**.

## Definitions
- **Urgent** — Delay likely causes immediate customer/business impact (outage, payment failure, blocked access, security/privacy incident, executive escalation, deadlines today/within hours).
- **Normal** — Requires action soon (by EOD/tomorrow) without immediate damage.
- **Low** — Non‑blocking or long‑term (feature ideas, FYIs, newsletters, general announcements; “no rush”).

## Language cues (guidance, not absolute)
- **Urgent:** “production down”, “checkout failing”, “today”, “ASAP”, “impacting customers”, “breach”
- **Normal:** “by tomorrow”, “please investigate”, “can you look”
- **Low:** “nice to have”, “feature request”, “whenever possible”, “no rush”, newsletters

## Tie‑breakers
- Security/privacy even plausible → **Urgent**.
- “Urgent” in subject but benign content → **Normal**, unless impact/time pressure is explicit.
- If truly uncertain → **Normal** and leave a note.

## Examples
- “Payment failing at checkout — need workaround **today**.” → **Urgent**
- “User can’t reset password; please check by **tomorrow**.” → **Normal**
- “Request: export dashboard to PDF, **no rush**.” → **Low**
- “High‑profile client blocked on SSO.” → **Urgent**
- “Bug seen once in staging; not reproducible; track for later.” → **Low**

## Quality controls
- Dual‑label 10–15% of items; adjudicate conflicts.
- Track inter‑annotator agreement (Cohen’s κ ≥ 0.75).

## Output format
CSV columns: `subject, body, label, annotator, notes`
Labels must be exactly one of: `Urgent`, `Normal`, `Low`.

# Corpus

Public financial documents only. Annual reports, bond offer documents and
disclosures published on an issuer's investor-relations pages or an exchange
site. Nothing confidential, internal, draft, paywalled or behind a login ever
goes in here (NFR-3, INVARIANT-aa1873bf).

PDFs are not committed — `.gitignore` excludes `corpus/*.pdf`. `manifest.json`
is the committed record of what the corpus contains and where each file came
from, so the corpus is reproducible from public sources without redistributing
the documents.

## Before adding a document

1. Download it yourself from the issuer's own investor-relations page or an
   exchange filing page. Record that exact page URL, not a search result.
2. Confirm the PDF has a real text layer — open it and try to select text. OCR
   is an explicit non-goal, so a scanned image-only PDF will silently extract
   nothing.
3. Add an entry to `manifest.json`. A missing or non-`http(s)` `source_url`
   makes the manifest fail to load, by design.

## Entry shape

```json
{
  "documents": [
    {
      "doc_id": "example-ar-fy25",
      "title": "Annual Report 2024-25",
      "issuer": "Example Issuer Limited",
      "doc_type": "annual_report",
      "published": "2025-06-30",
      "source_url": "https://www.example.com/investors/annual-report-fy25.pdf",
      "filename": "example-ar-fy25.pdf"
    }
  ]
}
```

`doc_id` is what citations resolve against, so it must be unique and stable —
changing it invalidates every citation already written against it.

## Still open

The two documents for this demo are not chosen yet. They must be annual reports
of **two different** Adani listed entities: the peer/sector specialist compares
issuers, and two documents from one issuer would leave it with nothing to
compare (ASSUMPTION-762e7a43).

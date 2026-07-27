# Repository deposition checklist

- [ ] Choose and add an author-approved software licence (MIT, BSD-3-Clause, Apache-2.0, or an institution-approved alternative).
- [ ] Confirm that the Figshare dataset licence permits the documented download and processing workflow.
- [x] Run `python smoke_check.py` in the prepared release environment.
- [ ] Test one short CPU smoke run and one full GPU entry point from repository-relative paths.
- [x] Confirm that no raw participant data, credentials, absolute paths, private logs, manuscripts, or reviewer documents are present.
- [x] Create the private GitHub repository and push this directory only.
- [ ] Create release `v1.0.0-review` (or the final chosen version).
- [ ] Connect the GitHub repository to Zenodo and archive the release.
- [ ] Test the Zenodo DOI and, if review is confidential, the private reviewer link outside the author account.
- [ ] Replace `[AUTHOR INPUT NEEDED: repository DOI/URL]` in the manuscript and response letter.

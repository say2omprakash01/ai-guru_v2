AI Guru v2 — how to source Bhagavad Gita text (legal + authenticity)

WHAT THIS REPO CONTAINS
- A small UTF-8 seed CSV under data/processed/ for pipeline development.
- Neutral English MEANING lines are short paraphrases for demos — not a published translation.
- For production, YOU must choose a translation you have the right to use and ship.

AUTHENTICITY: PRABHUPADA VS GITA PRESS (EDITORIAL, NOT LEGAL ADVICE)
- Srila Prabhupada (Bhaktivedanta Book Trust): Strong Gaudiya Vaishnava framing; verse-by-verse purport style in full books. Many readers want this explicitly as “the” reading tradition.
- Gita Press (Gorakhpur): Widely used in India; generally concise, traditional Hindu household tone; Hindi/English editions are common.
- “Which is more authentic?” is a tradition/commitment question. Scholarly critical editions focus on Sanskrit recensions; translations are always interpretive. Pick one primary line for your product narrative, attribute it clearly in the UI, and stay consistent.

COPYRIGHT (IMPORTANT)
- Modern translations and commentaries (including typical BBT and Gita Press volumes) are usually COPYRIGHTED.
- Do not scrape full copyrighted books into a public repo or commercial product without permission/license.
- Options: obtain permission from the publisher; use a clearly public-domain English translation for an OSS demo; or keep licensed text private (your deployment only) and never commit it to git.

HOW TO REACH ~700 VERSES
1) Export from a source you are allowed to use (licensed dataset, your own transcription, or publisher-approved export).
2) Normalize to our CSV columns (see data/processed/gita_verses.csv header).
3) Run: python scripts/validate_gita_csv.py data/processed/gita_verses.csv
4) Keep a copy of the license/permission in data/raw/ if you distribute the text.

NOTE ON OPEN DATASETS / APIs
- Open-source *code* (MIT/GPL) does not automatically mean every bundled translation is redistributable for your use case. Read each dataset’s terms and author list before production use.

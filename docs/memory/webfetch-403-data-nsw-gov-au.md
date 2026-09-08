---
name: webfetch-403-data-nsw-gov-au
description: WebFetch gets 403 on data.nsw.gov.au; requests/history.py gets 200 on the same URL.
metadata:
  type: reference
---

WebFetch tool returns HTTP 403 on data.nsw.gov.au (bot-detection), but Python's requests library (used by fuel_signal/history.py) gets a clean 200 on the same URL. For any ad-hoc scraping/checking of this dataset (e.g. resource discovery, checking published month coverage), use requests/history.py's discover_price_resources() via Bash, or the in-app Browser tool — not WebFetch.

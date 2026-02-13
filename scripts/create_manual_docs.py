import json
from datetime import datetime, timezone, timedelta

docs = [
    {
        "doc_id": "manual_test_001",
        "title": "Google Surprise Announcement: Gemini 3.5 Likely Coming in June",
        "raw_text": "Sources close to Google suggest that Gemini 3.5 is on track for a June 2026 release, earlier than expected. The new model promises significant reasoning improvements.",
        "source_type": "journalist",
        "author_type": "journalist",
        "timestamp": "2026-02-04T18:00:00",
        "url": "https://techcrunch.com/2026/02/04/google-gemini-3-5-june",
        "query_used": "Gemini 3.5 release date",
        "query_type": "active_monitoring",
        "event_id": "gemini-3pt5-release-2026",
        "extracted_entities": [
            {"text": "Gemini 3.5", "type": "PRODUCT", "confidence": 0.99, "start": 30, "end": 40},
            {"text": "Google", "type": "ORG", "confidence": 0.99, "start": 0, "end": 6}
        ],
        "normalized_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    },
    {
        "doc_id": "manual_test_002",
        "title": "Rumor: Gemini 3.5 Facing Compute Shortages, May Delay to July",
        "raw_text": "Unverified reports indicate Google is facing compute constraints that could push Gemini 3.5 launch back to July or August 2026.",
        "source_type": "social_media",
        "author_type": "industry_observer",
        "timestamp": "2026-02-05T12:00:00",
        "url": "https://twitter.com/ai_insider/status/123456789",
        "query_used": "Gemini 3.5 delay",
        "query_type": "active_monitoring",
        "event_id": "gemini-3pt5-release-2026",
        "extracted_entities": [
            {"text": "Gemini 3.5", "type": "PRODUCT", "confidence": 0.99, "start": 20, "end": 30},
            {"text": "Google", "type": "ORG", "confidence": 0.99, "start": 50, "end": 56}
        ],
        "normalized_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    },
     {
        "doc_id": "manual_test_003",
        "title": "DeepMind CEO hints at 'Mid-Year' surprise for Gemini",
        "raw_text": "Demis Hassabis mentioned in an interview that a '0.5' update to the Gemini series is ready for deployment mid-year.",
        "source_type": "official",
        "author_type": "executive",
        "timestamp": "2026-02-05T15:00:00",
        "url": "https://deepmind.google/news/demis-interview",
        "query_used": "Gemini 3.5 demis hassabis",
        "query_type": "targeted_search",
        "event_id": "gemini-3pt5-release-2026",
        "extracted_entities": [
            {"text": "Gemini", "type": "PRODUCT", "confidence": 0.99, "start": 60, "end": 66},
            {"text": "DeepMind", "type": "ORG", "confidence": 0.99, "start": 0, "end": 8},
             {"text": "Demis Hassabis", "type": "PERSON", "confidence": 0.99, "start": 0, "end": 14}
        ],
        "normalized_at": datetime.now(timezone.utc).isoformat()
    }
]

for doc in docs:
    path = f"data/normalized/{doc['doc_id']}.json"
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Created {path}")

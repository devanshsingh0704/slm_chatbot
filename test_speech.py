"""Checks how answers are rewritten for the voice (chat_server.speech_text). Usage: python test_speech.py <dataset>
Only general rules are checked here (units, ranges, dates, phones, ...); a dataset's own "pronunciations" are not."""
import sys

dataset = sys.argv[1]
sys.argv = ["chat_server.py", dataset]  # chat_server reads the dataset from argv on import
from chat_server import speech_text  # noqa: E402

CASES = {
    "Steel thickness range: 0.5 mm to 8 mm.": "Steel thickness range: 0.5 millimetres to 8 millimetres.",
    "The bed is 4.0 m long.": "The bed is 4.0 metres long.",
    "Capacity of 800 t per month.": "Capacity of 800 tonnes per month.",
    "Temperatures up to 200 °C.": "Temperatures up to 200 degrees Celsius.",
    "It has 200–250 employees.": "It has 200 to 250 employees.",
    "It has 200-250 employees.": "It has 200 to 250 employees.",
    "Updated on 2026-10-12.": "Updated on 12 October 2026.",
    "MIG/TIG welding and MES/ERP.": "M I G, T I G welding and M E S, E R P.",
    "Incorporated in 1999, with over 42 years.": "Incorporated in 19 99, with over 42 years.",
    "Gate No 12 and Plot No. 45": "Gate Number 12 and Plot Number 45",
    "Call +44 20 7946 0958.": "Call plus 4, 4. 2, 0. 7, 9, 4, 6. 0, 9, 5, 8.",
    "Call +1 555-0102.": "Call plus 1. 5, 5, 5. 0, 1, 0, 2.",
    "Postcode 123456, Exampleshire": "Postcode 1, 2, 3, 4, 5, 6, Exampleshire",
    "email sales@example.com": "email sales at example dot com",
    "- **Head Office**: Unit 1. See [example.com](http://example.com).": "Head Office: Unit 1. See example.com.",
    # must stay unchanged
    "Rapid 3D prototyping, 5-axis bending, 2 ton coil, Industry 4.0, 95%.":
        "Rapid 3D prototyping, 5-axis bending, 2 ton coil, Industry 4.0, 95%.",
    "No, it has 12 or 15 mills in 2005 and 2026.": "No, it has 12 or 15 mills in 2005 and 2026.",
}
bad = 0
for given, want in CASES.items():
    got = speech_text(given)
    if got != want:
        bad += 1
        print(f"BAD {given!r}\n    want {want!r}\n    got  {got!r}")
print(f"FAILED {bad}/{len(CASES)}" if bad else f"ALL OK {len(CASES)}")

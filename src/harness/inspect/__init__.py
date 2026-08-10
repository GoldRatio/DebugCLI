"""inspect: register/sensor collectors, decoder, and the curated register catalog.

Collectors only ever invoke commands through ``engine.Runner`` (the read-only
enforcement funnel). No collector ever writes. The decoder turns raw hex into
typed, human-readable fields using the curated catalog (NOT the LLM).
"""
"""LLM prompts for Target and Planner agents."""

TARGET_PROMPT = """
You are extracting structured information from HDB resale search queries.

Your task: Parse the user's query and extract ALL mentioned information into a Target object.

MULTI-TURN CONVERSATION SUPPORT:
If there are previous messages in the conversation:
- Review the previous assistant responses to see what Target was extracted before
- PRESERVE all fields from the previous Target UNLESS the current query explicitly changes them
- UPDATE only the fields that are explicitly mentioned or modified in the current query
- Treat the current query as building on top of the previous context

CLARIFYING QUESTION FOLLOW-UPS:
If the previous assistant message asked a clarifying question (e.g., "Which town should I use?"):
- The user's response is likely SHORT and ANSWER-ONLY (e.g., "Bedok", "4-room", "Bedok 4-room")
- Extract the answer and FILL IN the missing field that was asked about
- PRESERVE all other fields from the previous partial Target
- DO NOT treat short answers as completely new searches

Examples of clarifying question follow-ups:
Turn 1: "Find a flat" → {"town":null,"flat_type":null,...} (both missing)
Assistant asks: "I need the town and flat type before searching. Which town and flat type should I use?"
Turn 2: "Bedok 4-room" → {"town":"Bedok","flat_type":"4 ROOM",...} (filled in both)

Turn 1: "Find 4-room, 95 sqm, high floor" → {"town":null,"flat_type":"4 ROOM","floor_area_target":95,"storey_preference":"high",...}
Assistant asks: "I need the town before searching. Which town should I use for 4 ROOM flats?"
Turn 2: "Sengkang" → {"town":"Sengkang","flat_type":"4 ROOM","floor_area_target":95,"storey_preference":"high",...}
  (Notice: ONLY town was filled in, ALL other fields from Turn 1 are PRESERVED)

Turn 1: "Find flats in Tampines, mid floor, last 6 months" → {"town":"Tampines","flat_type":null,"storey_preference":"mid","months_back":6,...}
Assistant asks: "I need the flat type before searching. Which flat type should I use in Tampines?"
Turn 2: "3-room" → {"town":"Tampines","flat_type":"3 ROOM","storey_preference":"mid","months_back":6,...}
  (Notice: ONLY flat_type was filled in, town/storey_preference/months_back from Turn 1 are PRESERVED)

Examples of multi-turn refinements (NOT clarifying):
Turn 1: "Find 4-room in Bedok" → {"town":"Bedok","flat_type":"4 ROOM",...}
Turn 2: "Make it high floor" → {"town":"Bedok","flat_type":"4 ROOM","storey_preference":"high",...}
  (Notice: town and flat_type are PRESERVED from Turn 1, only storey_preference is ADDED)
Turn 3: "Within last 6 months" → {"town":"Bedok","flat_type":"4 ROOM","storey_preference":"high","months_back":6,...}
  (Notice: all previous fields PRESERVED, only months_back is UPDATED)

CRITICAL RULES:
1. TOWN: Extract the town name exactly as mentioned (e.g., "Sengkang", "Ang Mo Kio", "Bedok"). Look for Singapore HDB town names.
2. FLAT_TYPE: Extract flat type as "X ROOM" format (e.g., "4 ROOM", "3 ROOM", "5 ROOM"). Handle variants like "4-room" → "4 ROOM". Also handle "Executive" / "exec" / "executive flat" → "EXECUTIVE".
3. STOREY_PREFERENCE: If user mentions floor level, set as:
   - "low" for: low floor, ground floor, lower levels
   - "mid" for: mid floor, middle floor, mid-level
   - "high" for: high floor, top floor, upper levels
4. FLOOR_AREA: Extract sqm values. Default tolerance is 5.0 if not specified.
5. MONTHS_BACK: Extract time window (e.g., "last 12 months" → 12, "last 6 months" → 6). Default is 12.
6. Set all other fields (street_hint, flat_model_hint, min_remaining_lease_years, price_budget_max) only if explicitly mentioned.
7. Use null for missing fields, never leave them out.
8. In multi-turn conversations, PRESERVE previous fields and UPDATE only what's mentioned in the current query.

EXAMPLES:

Query: "Find a 4-room in Sengkang, 92 sqm, mid floor, last 12 months"
→ town: "Sengkang" (town name extracted)
→ flat_type: "4 ROOM" (normalized format)
→ storey_preference: "mid" (mid floor mentioned)
→ floor_area_target: 92
→ months_back: 12
Output: {"town":"Sengkang","flat_type":"4 ROOM","street_hint":null,"flat_model_hint":null,"floor_area_target":92,"floor_area_tolerance":5.0,"storey_preference":"mid","min_remaining_lease_years":null,"months_back":12,"price_budget_max":null,"enforce_street_hint":false,"enforce_price_budget":false}

Query: "Ang Mo Kio 3-room near Ave 8, 85 sqm, high floor, last 6 months"
→ town: "Ang Mo Kio" (multi-word town name)
→ flat_type: "3 ROOM"
→ street_hint: "Ave 8" (street mentioned)
→ storey_preference: "high" (high floor mentioned)
→ floor_area_target: 85
→ months_back: 6
Output: {"town":"Ang Mo Kio","flat_type":"3 ROOM","street_hint":"Ave 8","flat_model_hint":null,"floor_area_target":85,"floor_area_tolerance":5.0,"storey_preference":"high","min_remaining_lease_years":null,"months_back":6,"price_budget_max":null,"enforce_street_hint":false,"enforce_price_budget":false}

Query: "4 ROOM in Bedok, 100 sqm, min 80 years lease"
→ town: "Bedok"
→ flat_type: "4 ROOM"
→ floor_area_target: 100
→ min_remaining_lease_years: 80 (lease requirement mentioned)
→ months_back: 12 (default)
Output: {"town":"Bedok","flat_type":"4 ROOM","street_hint":null,"flat_model_hint":null,"floor_area_target":100,"floor_area_tolerance":5.0,"storey_preference":null,"min_remaining_lease_years":80,"months_back":12,"price_budget_max":null,"enforce_street_hint":false,"enforce_price_budget":false}

Query: "5-room flat in Tampines, low floor"
→ town: "Tampines"
→ flat_type: "5 ROOM" (normalized from "5-room")
→ storey_preference: "low" (low floor mentioned)
Output: {"town":"Tampines","flat_type":"5 ROOM","street_hint":null,"flat_model_hint":null,"floor_area_target":null,"floor_area_tolerance":5.0,"storey_preference":"low","min_remaining_lease_years":null,"months_back":12,"price_budget_max":null,"enforce_street_hint":false,"enforce_price_budget":false}

Query: "Executive flat in Pasir Ris, last 12 months"
→ town: "Pasir Ris"
→ flat_type: "EXECUTIVE" ("Executive" / "exec" / "executive flat" maps to "EXECUTIVE")
→ months_back: 12
Output: {"town":"Pasir Ris","flat_type":"EXECUTIVE","street_hint":null,"flat_model_hint":null,"floor_area_target":null,"floor_area_tolerance":5.0,"storey_preference":null,"min_remaining_lease_years":null,"months_back":12,"price_budget_max":null,"enforce_street_hint":false,"enforce_price_budget":false}

Query: "4-room near Compassvale"
→ flat_type: "4 ROOM"
→ street_hint: "Compassvale" (area/street mentioned after "near")
→ town: null (town not mentioned - will ask user)
→ months_back: 12 (default)
Output: {"town":null,"flat_type":"4 ROOM","street_hint":"Compassvale","flat_model_hint":null,"floor_area_target":null,"floor_area_tolerance":5.0,"storey_preference":null,"min_remaining_lease_years":null,"months_back":12,"price_budget_max":null,"enforce_street_hint":false,"enforce_price_budget":false}

Query: "Sengkang 4-room, premium apartment-ish, mid floor, long lease"
→ town: "Sengkang"
→ flat_type: "4 ROOM"
→ flat_model_hint: "Premium Apartment" (premium apartment mentioned)
→ storey_preference: "mid"
→ min_remaining_lease_years: 80 (long lease implied)
Output: {"town":"Sengkang","flat_type":"4 ROOM","street_hint":null,"flat_model_hint":"Premium Apartment","floor_area_target":null,"floor_area_tolerance":5.0,"storey_preference":"mid","min_remaining_lease_years":80,"months_back":12,"price_budget_max":null,"enforce_street_hint":false,"enforce_price_budget":false}

IMPORTANT:
- Always extract town and storey_preference when they are mentioned in the query!
- When "near", "around", or "in the area of" is followed by a PLACE NAME or street name, extract it as street_hint.
- Do NOT extract street_hint when "near" / "around" is followed by a number and a unit (sqm, sq m, sqft). That is a floor area measurement, not a location. Example: "around 95 sqm" → floor_area_target=95, street_hint=null.
""".strip()

PLANNER_PROMPT = """
You are the query-planning agent for HDB resale search.
You receive a JSON payload with query, target, filters, conflicts, count, stats, history, thresholds,
and available adjustments.

Decide the next action: relax, tighten, accept, or clarify.

Guidelines:
- If required fields are missing or conflicts are present, choose "clarify".
- If count < min_count and relax_steps < max_relax_steps and available_relax_adjustments is not empty,
  choose "relax".
- If count > max_count and tighten_steps < max_tighten_steps and available_tighten_adjustments is not empty,
  choose "tighten".
- If count < min_count and no relax adjustments are available, choose "clarify".
- If count > max_count and no tighten adjustments are available, choose "clarify".
- Otherwise choose "accept".

If action is "relax" or "tighten", choose one adjustment:
- Relax: widen_time_window, widen_sqm_tolerance, drop_storey_preference.
- Tighten: narrow_time_window, narrow_sqm_tolerance, raise_min_lease_years, require_street_hint,
  cap_price_budget.
Prefer adjustments listed in the available_* arrays and avoid repeating history unless needed.
If no adjustments are available and the results are already useful, choose "accept".
Use null for adjustment when action is accept/clarify.

Output format:
- Return ONLY a JSON object that matches the PlannerDecision schema.
- Do not include prose, markdown, or tool calls.
- When action is "relax" or "tighten", adjustment MUST be a non-null string from the available list.
- Examples:
  {"action":"relax","adjustment":"widen_time_window","reason":"count below minimum"}
  {"action":"tighten","adjustment":"narrow_time_window","reason":"count above maximum"}
  {"action":"accept","adjustment":null,"reason":"count in range"}

Keep the reason short and specific.
""".strip()

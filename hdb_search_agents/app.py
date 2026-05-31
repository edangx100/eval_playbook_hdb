from __future__ import annotations

import pandas as pd
import gradio as gr

from contextlib import nullcontext

from hdb_search_agents.agent.orchestrator import run_iterative_search
from hdb_search_agents.agent.tracing import get_tracer, initialize_braintrust_tracing
from settings import Settings
from hdb_search_agents.viz.plots import plot_price_hist

# Initialize BrainTrust tracing if configured
# This must happen before any agents are created to capture all spans
settings = Settings()
initialize_braintrust_tracing(
    api_key=settings.braintrust_api_key,
    parent=settings.braintrust_parent,
)

# Column order for the Gradio results table.
DISPLAY_COLUMNS = [
    "month",
    "town",
    "flat_type",
    "block",
    "street_name",
    "storey_range",
    "floor_area_sqm",
    "remaining_lease",
    "resale_price",
]

# Example prompts to help users get started.
EXAMPLE_QUERIES = [
    "Find 4-room in Ang Mo Kio, around 95 sqm, last 12 months",
    "Find comps for 4-room near Compassvale, ~95 sqm, last 12 months",
    "Sengkang 4-room, premium apartment-ish, mid floor, long lease",
    "3-room in Bedok, max 80 sqm, high floor, last 6 months",
]

# Flat type options (clickable to populate query input)
FLAT_TYPES = [
    "5 ROOM",
    "4 ROOM",
    "3 ROOM",
    "2 ROOM",
    "1 ROOM",
    "EXECUTIVE",
    "MULTI-GENERATION",
]


def detect_new_search(query: str, conversation_state: dict) -> bool:
    """
    Detect if the query appears to be a new search vs a follow-up.

    Returns True if:
    - Query contains both town and flat_type indicators
    - AND we have existing conversation history

    This suggests the user might be starting a new search rather than refining.
    """
    if not conversation_state.get("messages"):
        return False  # No history, so nothing to detect

    query_lower = query.lower()

    # Common town names (subset for detection)
    TOWNS = [
        "ang mo kio", "bedok", "bishan", "bukit batok", "bukit merah",
        "bukit panjang", "bukit timah", "choa chu kang", "clementi",
        "geylang", "hougang", "jurong east", "jurong west", "kallang",
        "marine parade", "pasir ris", "punggol", "queenstown",
        "sembawang", "sengkang", "serangoon", "tampines", "toa payoh",
        "woodlands", "yishun"
    ]

    # Flat type patterns
    FLAT_PATTERNS = [
        "1-room", "1 room", "2-room", "2 room", "3-room", "3 room",
        "4-room", "4 room", "5-room", "5 room", "executive", "multi-generation"
    ]

    has_town = any(town in query_lower for town in TOWNS)
    has_flat_type = any(pattern in query_lower for pattern in FLAT_PATTERNS)

    return has_town and has_flat_type


def reset_conversation(conversation_state: dict) -> tuple[str, str, str, pd.DataFrame, str, object, dict]:
    """
    Reset conversation state and clear all results.

    Returns a fresh state with empty message history and cleared UI elements.
    """
    # Close the open conversation span so Braintrust finalises the grouped trace.
    span = (conversation_state or {}).get("conv_span")
    if span is not None:
        try:
            span.end()
        except Exception:
            pass

    empty_df = pd.DataFrame(columns=DISPLAY_COLUMNS)
    # "target" starts empty so the first query after a reset can't inherit anything.
    fresh_state = {"messages": [], "target": None, "conv_span": None, "turn_count": 0}

    return (
        "✅ Conversation reset. Ready for a new search!",
        "Results: 0",
        "Price stats: n/a",
        empty_df,
        "",
        None,
        fresh_state,
    )


async def run_agent_search(
    query: str, strictness: float, conversation_state: dict
) -> tuple[str, str, str, pd.DataFrame, str, object, dict]:
    """
    Run the agentic search and return results for Gradio display.

    Args:
        query: Natural language query from user
        strictness: Slider value (0-1) - higher means stricter tolerances
        conversation_state: Dict tracking conversation history for context

    Returns:
        Tuple of (status_message, count_message, stats_message, results_df, trace_html, histogram_fig, updated_state)
    """
    if not query.strip():
        empty_df = pd.DataFrame(columns=DISPLAY_COLUMNS)
        return (
            "Please enter a search query.",
            "Results: 0",
            "Price stats: n/a",
            empty_df,
            "",
            None,  # No histogram
            conversation_state,
        )

    # Get previous message history from conversation state
    # PydanticAI's message history preserves full conversation context including
    # system prompts, tool calls, and model responses - no manual concatenation needed
    previous_messages = conversation_state.get("messages", [])

    # Detect if this looks like a new search vs a follow-up
    # If both town and flat_type are present in the query and we have history,
    # the user might be starting a new search rather than refining the previous one
    new_search_detected = detect_new_search(query, conversation_state)

    # Multi-turn fix (mirrors the eval harness): hand the *previous turn's*
    # extracted Target to the orchestrator as `prior_target`, so a follow-up like
    # "actually, mid floor" inherits the flat_type / town it leaves blank instead
    # of dropping them. Skip it when a brand-new search is detected, so a fresh
    # query doesn't silently inherit stale context (the Reset button clears it too).
    previous_target = None if new_search_detected else conversation_state.get("target")

    # Log the query for debugging.
    print(f"Agent search query={query!r} strictness={strictness}", flush=True)
    if previous_messages:
        print(f"Using {len(previous_messages)} previous messages for context", flush=True)
        if new_search_detected:
            print("⚠️  New search detected (has town + flat_type). Consider clicking 'Reset' to clear previous context.", flush=True)

    # Placeholder for strictness adjustment (to be implemented).
    # TODO: Use strictness to adjust Target.floor_area_tolerance and other tolerances
    # For now, we just pass the query as-is.

    # --- Braintrust conversation-level span grouping (OTel) ---
    # We use the same OTel TracerProvider that Agent.instrument_all() uses, so any
    # span we make active here becomes the automatic parent for PydanticAI's spans.
    # BraintrustSpanProcessor then exports the full conversation→turn→agent hierarchy.
    tracer = get_tracer()
    conv_span = (conversation_state or {}).get("conv_span")
    turn_count = (conversation_state or {}).get("turn_count", 0) + 1
    _conv_token = None

    if tracer is not None:
        from opentelemetry import context as otel_context, trace as otel_trace

        # Open the conversation span once (first turn); keep it open across turns.
        if conv_span is None:
            conv_span = tracer.start_span(
                "conversation",
                attributes={"first_query": query},
            )

        # Attach the conversation span as the active OTel context so the turn span
        # (and everything inside it) is recorded as its child.
        _conv_token = otel_context.attach(otel_trace.set_span_in_context(conv_span))

    # turn_cm is either a real "turn_N" span or a no-op when tracing is off.
    turn_cm = (
        tracer.start_as_current_span(
            f"turn_{turn_count}",
            attributes={"query": query, "turn": turn_count},
        )
        if tracer is not None
        else nullcontext()
    )

    try:
      with turn_cm:
        try:
            # Pass message history (for the LLM's own multi-turn reasoning) AND
            # prior_target (the deterministic field-inheritance safety net) so
            # earlier-turn fields like flat_type survive a follow-up query.
            response = await run_iterative_search(
                query,
                message_history=previous_messages,
                prior_target=previous_target,
                verbose=True,
            )
        except RuntimeError as exc:
            # Catch specific runtime errors with user-friendly messages
            empty_df = pd.DataFrame(columns=DISPLAY_COLUMNS)
            error_str = str(exc)

            # Provide user-friendly error messages for common failure modes
            if "Database" in error_str or "database" in error_str:
                error_message = (
                    "## ⚠️ Database Connection Error\n\n"
                    "Unable to connect to the database. Please ensure:\n"
                    "- PostgreSQL is running (`docker compose up -d`)\n"
                    "- Database credentials in `.env` are correct\n\n"
                    f"Technical details: {exc}"
                )
            elif "Embedding" in error_str or "embedding" in error_str:
                error_message = (
                    "## ⚠️ Embedding Service Error\n\n"
                    "Unable to generate embeddings for semantic search. Please check:\n"
                    "- JINA_API_KEY is set in `.env`\n"
                    "- Jina API service is accessible\n\n"
                    "The system will attempt to use structured search only.\n\n"
                    f"Technical details: {exc}"
                )
            elif "API" in error_str:
                error_message = (
                    "## ⚠️ API Error\n\n"
                    "An external API call failed. This could be:\n"
                    "- OpenRouter API (LLM inference)\n"
                    "- Jina API (embeddings)\n\n"
                    "Please check your API keys and network connection.\n\n"
                    f"Technical details: {exc}"
                )
            else:
                error_message = (
                    f"## ⚠️ Search Error\n\n"
                    f"An error occurred while processing your search:\n\n"
                    f"{exc}\n\n"
                    f"Please try rephrasing your query or contact support if the issue persists."
                )

            return error_message, "Results: 0", "Price stats: n/a", empty_df, "", None, {
                "messages": conversation_state.get("messages", []),
                # Preserve the prior Target across a failed turn so a follow-up
                # afterwards can still inherit earlier-turn fields.
                "target": conversation_state.get("target"),
                "conv_span": conv_span,
                "turn_count": turn_count,
            }
        except ValueError as exc:
            # Validation errors (e.g., missing API keys, invalid input)
            empty_df = pd.DataFrame(columns=DISPLAY_COLUMNS)
            error_message = (
                f"## ⚠️ Invalid Input\n\n"
                f"{exc}\n\n"
                f"Please check your query and try again."
            )
            return error_message, "Results: 0", "Price stats: n/a", empty_df, "", None, {
                "messages": conversation_state.get("messages", []),
                # Preserve the prior Target across a failed turn so a follow-up
                # afterwards can still inherit earlier-turn fields.
                "target": conversation_state.get("target"),
                "conv_span": conv_span,
                "turn_count": turn_count,
            }
        except Exception as exc:
            # Catch-all for unexpected errors
            empty_df = pd.DataFrame(columns=DISPLAY_COLUMNS)
            error_message = (
                f"## ⚠️ Unexpected Error\n\n"
                f"An unexpected error occurred: {type(exc).__name__}\n\n"
                f"{exc}\n\n"
                f"Please try again or contact support if the issue persists."
            )
            return error_message, "Results: 0", "Price stats: n/a", empty_df, "", None, {
                "messages": conversation_state.get("messages", []),
                # Preserve the prior Target across a failed turn so a follow-up
                # afterwards can still inherit earlier-turn fields.
                "target": conversation_state.get("target"),
                "conv_span": conv_span,
                "turn_count": turn_count,
            }

        # Build status message from response note.
        # Make clarifying questions highly visible with larger text and prominent styling.
        if response.note and (response.count == 0 or "?" in response.note):
            # This is likely a clarifying question - make it stand out!
            status_message = f"## ⚠️ {response.note}"
        else:
            status_message = response.note or "Search complete."

        # Add notification if a new search was detected with existing conversation history
        # This helps users understand they might want to reset the conversation for a fresh start
        if new_search_detected:
            status_message = f"💡 **New search detected.** Click 'Reset Conversation' to start fresh.\n\n{status_message}"

        # Build count message.
        count_message = f"Results: {response.count}"

        # Build stats message from response stats.
        if response.stats.count > 0:
            stats_message = (
                f"Price stats (SGD): "
                f"min {response.stats.min:,.0f}, "
                f"median {response.stats.median:,.0f}, "
                f"max {response.stats.max:,.0f}"
            )
        else:
            stats_message = "Price stats: n/a"

        # Convert results to DataFrame.
        if response.results:
            results_df = pd.DataFrame([r.model_dump() for r in response.results])
            # Ensure columns are in the correct order.
            results_df = results_df[DISPLAY_COLUMNS]
        else:
            results_df = pd.DataFrame(columns=DISPLAY_COLUMNS)

        # Build trace HTML for display.
        trace_html = _format_trace(response.trace)

        # Generate price histogram if we have results
        histogram_fig = None
        if response.results and response.stats.count > 0:
            # Extract prices from results
            prices = [r.resale_price for r in response.results if r.resale_price is not None]

            if prices:
                # Generate histogram with context from target
                histogram_fig = plot_price_hist(
                    prices=prices,
                    stats=response.stats,
                    town=response.target.town,
                    flat_type=response.target.flat_type,
                    months_back=response.target.months_back,
                )

        # Update conversation state: carry conv_span forward so the next turn is
        # grouped under the same parent trace in Braintrust.
        updated_state = {
            "messages": response.messages,
            # Remember this turn's Target so the next turn can inherit any field
            # the follow-up query leaves blank (consumed as prior_target above).
            "target": response.target,
            "conv_span": conv_span,
            "turn_count": turn_count,
        }

        return status_message, count_message, stats_message, results_df, trace_html, histogram_fig, updated_state

    finally:
        # Detach the conversation context so it doesn't leak into unrelated OTel spans.
        # The conversation span itself stays open until reset_conversation() calls end().
        if _conv_token is not None:
            try:
                otel_context.detach(_conv_token)
            except Exception:
                pass


def _format_trace(trace_steps) -> str:
    """
    Format trace steps as HTML for display in the UI.

    Each trace step shows the iterative search process:
    - Step number and descriptive name (e.g., "Initial count", "Relax: widened time window")
    - Count of matching candidates at this step
    - Action taken (initial/relax/tighten/accept/clarify) with color coding
    - Retrieval mode (structured SQL or hybrid vector+SQL)
    - Embedding query text, top-k, and BM25 details (hybrid mode only)
    - Adjustment note explaining what changed
    - Collapsible filters section showing all SQL filters applied

    Returns an HTML string with styled boxes and collapsible details.
    """
    if not trace_steps:
        return "<p>No trace available.</p>"

    html_parts = [
        "<div style='font-family: monospace; font-size: 13px; line-height: 1.6;'>"
    ]

    for step in trace_steps:
        # Build a structured trace entry for each step with all key information.
        # Each step is rendered as a card with left border color matching the action type.
        step_html = f"<div style='margin-bottom: 16px; padding: 12px; background-color: #f8f9fa; border-left: 3px solid #007bff; border-radius: 4px;'>"

        # Header: Step number and name
        # The step name is generated by the orchestrator (e.g., "Initial count", "Relax: widened time window").
        step_html += f"<div style='margin-bottom: 8px;'>"
        step_html += f"<strong style='color: #007bff;'>Step {step.step_number}:</strong> "
        step_html += f"<span style='font-weight: 500;'>{step.step_name}</span>"
        step_html += "</div>"

        # Details row: count, action, retrieval mode
        # These are displayed inline separated by pipes for compactness.
        details = []

        # Count shows how many candidates matched the filters at this step.
        if step.count is not None:
            details.append(f"<strong>Count:</strong> {step.count}")

        # Action shows what the planner decided to do at this step.
        # Color-coded for quick visual scanning:
        # - gray (initial) = first observation
        # - yellow (relax) = loosening constraints
        # - cyan (tighten) = narrowing constraints
        # - green (accept) = final acceptance
        # - red (clarify) = needs user input
        if step.action:
            action_color = {
                "initial": "#6c757d",
                "relax": "#ffc107",
                "tighten": "#17a2b8",
                "accept": "#28a745",
                "clarify": "#dc3545",
            }.get(step.action, "#6c757d")
            details.append(
                f"<strong>Action:</strong> <span style='color: {action_color};'>{step.action}</span>"
            )

        # Retrieval mode shows whether this used pure SQL (structured) or vector search (hybrid).
        details.append(f"<strong>Mode:</strong> {step.retrieval_mode}")

        # Display the text query used for vector similarity search (hybrid mode only)
        if step.query_text_used_for_embedding:
            details.append(
                f"<strong>Embed query:</strong> {step.query_text_used_for_embedding}"
            )

        if step.topk is not None:
            details.append(f"<strong>TopK:</strong> {step.topk}")

        if step.bm25_used is not None:
            bm25_status = "yes" if step.bm25_used else "no"
            if step.bm25_boost_reason:
                bm25_status = f"{bm25_status} ({step.bm25_boost_reason})"
            details.append(f"<strong>BM25:</strong> {bm25_status}")

        if details:
            step_html += f"<div style='margin-bottom: 6px;'>{' | '.join(details)}</div>"

        # Adjustment note if present
        # Explains what changed in human-readable form (e.g., "widened time window to last 12 months").
        if step.adjustment_note:
            step_html += f"<div style='margin-bottom: 6px; font-style: italic; color: #495057;'>"
            step_html += f"→ {step.adjustment_note}"
            step_html += "</div>"

        # Filters in a collapsible details element
        # Shows all SQL filters applied at this step (town, flat_type, sqm range, etc.).
        # Collapsed by default to avoid clutter; users can expand to see full filter details.
        if step.filters:
            filter_summary = _format_filter_summary(step.filters)
            step_html += f"<details style='margin-top: 8px;'>"
            step_html += f"<summary style='cursor: pointer; color: #495057; font-size: 12px;'>Show filters ({len(step.filters)} applied)</summary>"
            step_html += f"<div style='margin-top: 6px; padding-left: 12px; font-size: 12px; color: #6c757d;'>"
            step_html += filter_summary
            step_html += "</div>"
            step_html += "</details>"

        step_html += "</div>"
        html_parts.append(step_html)

    html_parts.append("</div>")
    return "".join(html_parts)


def _format_filter_summary(filters: dict) -> str:
    """
    Format the filters dictionary as readable HTML for display in trace details.

    Converts the raw filter dict from the orchestrator into a human-readable format.
    Each filter key-value pair is displayed on a separate line.

    Examples of filters:
    - town: "ANG MO KIO"
    - flat_type: "4 ROOM"
    - months_back: 12
    - sqm_min: 90.0, sqm_max: 100.0
    - remaining_lease_months_min: 1140 (95 years)
    - storey_min: 11 (high floor)
    - enforce_street_hint: False

    Returns an HTML string with filter key-value pairs separated by line breaks.
    """
    if not filters:
        return "<em>No filters</em>"

    lines = []
    for key, value in filters.items():
        if value is not None:
            # Format boolean values more clearly (Yes/No instead of True/False).
            # This makes enforcement flags easier to read.
            if isinstance(value, bool):
                value_str = "Yes" if value else "No"
            else:
                value_str = str(value)
            # Use a monospace style with clear key-value formatting
            lines.append(f'<div style="margin-bottom: 4px;"><strong style="color: #495057;">{key}:</strong> <span style="color: #212529;">{value_str}</span></div>')

    return "".join(lines) if lines else "<em>No active filters</em>"


# Build the Gradio UI: chat input on the left, results + trace on the right.
with gr.Blocks(title="HDB Compare Flats Agent") as demo:
    gr.Markdown("# HDB Compare Flats Agent")
    gr.Markdown("Ask for comparable HDB resale flats.")

    # Conversation state to track context across multiple turns
    # Stores PydanticAI Message objects to preserve full conversation history
    # This enables the agent to understand multi-turn conversations without manual string concatenation
    # "target" holds the previous turn's extracted Target so a follow-up can inherit
    # fields it leaves blank (multi-turn fix); starts None for a clean first turn.
    conversation_state = gr.State(value={"messages": [], "target": None, "conv_span": None, "turn_count": 0})

    with gr.Row():
        # Left column: chat input + controls.
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="e.g., Find 4-room in Ang Mo Kio, around 95 sqm, last 12 months",
                lines=3,
            )

            strictness_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.5,
                label="Strictness (0=loose, 1=strict)",
                info="Adjusts search tolerances. Higher = stricter constraints.",
            )

            with gr.Row():
                search_btn = gr.Button("Search", variant="primary", scale=3)
                reset_btn = gr.Button("Reset Conversation", variant="secondary", scale=1)

            gr.Markdown("### Example Queries")
            # Clicking an example query will populate the textbox above.
            # The 'inputs' parameter connects the examples to the query_input textbox.
            examples = gr.Examples(
                examples=EXAMPLE_QUERIES,
                inputs=query_input,
                label="",
            )

            gr.Markdown("### Flat Types Available")
            # Clicking a flat type will populate the textbox (useful for answering clarifying questions)
            flat_type_examples = gr.Examples(
                examples=FLAT_TYPES,
                inputs=query_input,
                label="",
            )

        # Right column: results display.
        with gr.Column(scale=3):
            status = gr.Markdown("Ready.")

            with gr.Row():
                count = gr.Markdown("Results: 0")
                stats = gr.Markdown("Price stats: n/a")

            results = gr.Dataframe(
                headers=DISPLAY_COLUMNS,
                label="Comparable Transactions",
                wrap=True,
            )

            # Price distribution histogram
            histogram_plot = gr.Plot(
                label="Price Distribution",
                show_label=True,
            )

            # Trace panel (initially collapsed).
            with gr.Accordion("Agent Trace", open=False):
                trace_display = gr.HTML(label="")

    # Wire up the search button to the agent.
    # conversation_state is both an input (to read previous message history) and output (to update for next turn)
    # This enables multi-turn conversations using PydanticAI's native message history
    search_btn.click(
        run_agent_search,
        inputs=[query_input, strictness_slider, conversation_state],
        outputs=[status, count, stats, results, trace_display, histogram_plot, conversation_state],
    )

    # Wire up the reset button to clear conversation history and UI state
    # This allows users to start a fresh conversation without the context of previous searches
    reset_btn.click(
        reset_conversation,
        inputs=[conversation_state],
        outputs=[status, count, stats, results, trace_display, histogram_plot, conversation_state],
    )


if __name__ == "__main__":
    demo.launch()

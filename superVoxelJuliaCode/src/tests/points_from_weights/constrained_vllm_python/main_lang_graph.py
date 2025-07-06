import uuid
import random
import operator
import asyncio
import httpx
import re
import os
import json
from pathlib import Path
from typing import TypedDict, List, Annotated, Optional, Dict, Any

from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
try:
    # This logic attempts to import 'Send' from its various possible locations
    # across different versions of the langgraph library to ensure compatibility.
    from langgraph.constants import Send
except ImportError:
    try:
        from langgraph.graph import Send
    except ImportError:
        from langgraph.prebuilt import Send

# --- Custom Algorithm Validation and Execution Imports ---
from prompts import MASTER_ALGORITHM_PROMPT, INITIAL_ALGORITHM, INITIAL_ALGORITHM_CRITIQUE
from check_algo_string_validity import validate_algorithm
from algo_validation_utils import check_tags_presence, check_s5_point_usage, check_statement_termination, extract_step_blocks
from verify_is_string_ok_with_grammar import validate_algorithm_syntax, validate_algorithm_syntax_main
from execute_grid_algorithm_full import execute_grid_algorithm_full


# --- Import for NATIVE ASYNC checkpointer ---
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


# --- Configuration ---
VLLM_API_URL = "http://localhost:5000/v1/completions"
VLLM_MODEL_NAME = "nvidia/OpenMath-Nemotron-7B"

# --- Genetic Algorithm Parameters ---
TOP_K_BRANCHES = 4
PROPOSALS_PER_BRANCH = 3
MAX_GENERATIONS = 10000

# --- Debugging, Logging, and State Persistence ---
MAIN_DEBUG_FOLDER = Path("main_debug")
MAIN_DEBUG_FOLDER_jsons = MAIN_DEBUG_FOLDER / "jsons"
VLLM_LOG_FILE = MAIN_DEBUG_FOLDER / "vllm_api_log.json"
TEMP_DIR = Path("./temp_debug")

# --- Logging Helper ---
async def _log_to_markdown(log_file_path, content, mode='a'):
    """Appends content to the specified markdown log file."""
    try:
        with open(log_file_path, mode, encoding="utf-8") as f:
            f.write(content + "\n\n")
    except Exception as e:
        print(f"  ❌ Failed to write to log file {log_file_path}. Error: {e}")

# --- API Call with Retry Logic ---
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def call_vllm_api(prompt: str, context: str = "", call_name: str = "Unnamed Call") -> str:
    """
    Calls the vLLM OpenAI-compatible server with robust retry logic.
    Logs every prompt call to a JSONL file for debugging.
    """
    print(f"\n📡 Calling vLLM API for: {call_name}")

    # --- Logging Logic ---
    full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt
    log_entry = {
        "call_name": call_name,
        "prompt": prompt,
        "context": context,
        "full_prompt_sent": full_prompt
    }
    # This logging is fire-and-forget; we don't use a try/except block
    # to avoid masking potential file system errors during development.
    with open(VLLM_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # The try/except for the API call itself is critical for retry logic.
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "model": VLLM_MODEL_NAME, "prompt": full_prompt,
                "max_tokens": 4096, "temperature": 0.7,
            }
            response = await client.post(VLLM_API_URL, json=payload)
            response.raise_for_status()
            result_text = response.json()["choices"][0]["text"]
            # Ensure we don't return None, which can cause downstream errors
            return result_text if result_text is not None else ""
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"  -> API call failed with error: {e}. Retrying...")
        raise
    except RetryError as e:
        print(f"  -> API call failed after multiple retries: {e}. Returning error message.")
        return "ERROR: LLM API call failed."

# --- State Definitions ---
class BranchState(TypedDict):
    """Represents the state for a single algorithmic gene."""
    branch_id: str
    generation: int
    parent_branch_id: Optional[str]
    status: str
    algorithm_code: str
    modified_code_proposal: Optional[str]
    last_successful_score: float
    critique_and_ideas: str 
    event_log: Annotated[List[Dict[str, Any]], operator.add]
    plan_history_for_task: Optional[str]
    updated_plan_summary: Optional[str]

class GlobalState(TypedDict):
    """Manages the overall optimization process."""
    generation_count: int
    top_k_branches: int
    active_branches: List[BranchState]
    branches: Annotated[List[BranchState], operator.add]
    current_generation_results: Annotated[List[BranchState], operator.add]
    proposal_plans_history: Dict[str, str]
    
# --- Proposal Subgraph Nodes ---
# The subgraph operates on a temporary, in-memory copy of the branch state.
# It does not need to be checkpointed itself, as it's treated as an atomic unit.

async def plan_and_implement_one(state: BranchState) -> BranchState:
    """Generates a plan and new code for a single branch and logs it to a markdown file."""
    parent_id = state['parent_branch_id']
    branch_id = state['branch_id']
    generation = state['generation']
    history_for_task = state['plan_history_for_task']
    critique_and_ideas = state['critique_and_ideas']
    
    print(f"\n💡 Planning and Implementing for branch {branch_id} (Parent: {parent_id})...")

    plan_prompt = "Based on the algorithm and critique, propose a new, distinct high-level plan."
    plan_context = (f"CURRENT ALGORITHM:\n```{state['algorithm_code']}```\n\n"
                    f"CRITIQUE & IDEAS:\n{critique_and_ideas}\n\n"
                    f"HISTORY OF PLANS FOR THIS BRANCH:\n{history_for_task}")
    new_plan = await call_vllm_api(plan_prompt, plan_context, f"Generate Plan (Parent: {parent_id})")
    
    summary_prompt = "Summarize the key ideas from the old plan summary and the new plan."
    summary_context = f"Old Plan Summary:\n{history_for_task}\n\nNew Plan:\n{new_plan}"
    updated_summary = await call_vllm_api(summary_prompt, summary_context, f"Summarize Plans (Parent: {parent_id})")
    
    impl_prompt = "Based on the plan, write a complete new algorithm code."
    impl_context = f"CURRENT ALGORITHM:\n```{state['algorithm_code']}```\n\n PLAN TO IMPLEMENT:\n{new_plan}"
    proposal_code = await call_vllm_api(impl_prompt, impl_context, f"Implement Plan (Parent: {parent_id})")

    # --- Detailed Markdown Logging for Planning Stage ---
    log_content = (
        f"# Branch: {branch_id}\n\n"
        f"## Generation: {generation}\n\n"
        f"### History of Plans\n"
        f"```\n{history_for_task}\n```\n\n"
        f"### Critique & Ideas for Parent ({parent_id})\n"
        f"```\n{critique_and_ideas}\n```\n\n"
        f"### New Plan\n"
        f"```\n{new_plan}\n```\n\n"
        f"### Proposed Code\n"
        f"```python\n{proposal_code.strip() if proposal_code else 'ERROR: No code was generated.'}\n```\n"
    )

    try:
        log_file_path = MAIN_DEBUG_FOLDER / f"{branch_id}_plan_log.md"
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(log_content)
        print(f"  📝 Log saved to: {log_file_path}")
    except Exception as e:
        print(f"  ❌ Failed to write log file for branch {branch_id}. Error: {e}")




    state.update({
        "status": 'pending_validation',
        "modified_code_proposal": proposal_code.strip(),
        "critique_and_ideas": new_plan,
        "event_log": [{"type": "creation", "parent": parent_id, "plan": new_plan}],
        "updated_plan_summary": updated_summary
    })
    return state

async def validate_syntax(state: BranchState) -> BranchState:
    """Performs a robust, multi-stage validation of a proposed algorithm with detailed logging."""
    branch_id = state['branch_id']
    log_file_path = MAIN_DEBUG_FOLDER / f"{branch_id}_syntax_validation_log.md"
    
    # Initialize Log File
    await _log_to_markdown(log_file_path, f"# Branch: {branch_id} - Syntax Validation Log", mode='w')
    
    print(f"\n🔬 Validating Syntax for Branch {branch_id}...")
    
    # --- Stage 1: Holistic Validation ---
    await _log_to_markdown(log_file_path, "## Stage 1: Holistic Validation - Attempt 1")
    print("  -> Stage 1: Holistic Validation")
    current_code = state['modified_code_proposal']
    is_valid, suggestions = validate_algorithm(current_code)

    if is_valid:
        success_msg = "✅ Holistic validation passed on first attempt."
        print(f"    {success_msg}")
        await _log_to_markdown(log_file_path, f"### Status: Success\n{success_msg}")
        state['status'] = 'pending_execution'
        return state

    failure_msg = "Holistic validation failed."
    print(f"    ⚠️ {failure_msg} Suggestions: {suggestions}. Attempting LLM correction...")
    await _log_to_markdown(log_file_path, f"### Status: Failed\n**Suggestions:**\n```\n{suggestions}\n```")
    
    await _log_to_markdown(log_file_path, "## Stage 1: Holistic Validation - Attempt 2 (After LLM Correction)")
    correction_prompt = "The following algorithm failed validation. Please correct it based on the suggestions."
    correction_context = f"FAILED ALGORITHM:\n```\n{current_code}\n```\n\nSUGGESTIONS:\n{suggestions}"
    corrected_code = await call_vllm_api(correction_prompt, correction_context, f"Correct Algorithm (Branch: {branch_id})")
    
    is_valid, suggestions = validate_algorithm(corrected_code)
    if is_valid:
        success_msg = "✅ Holistic validation passed on second attempt after LLM correction."
        print(f"    {success_msg}")
        await _log_to_markdown(log_file_path, f"### Status: Success\n{success_msg}")
        state['status'] = 'pending_execution'
        state['modified_code_proposal'] = corrected_code
        state['event_log'].append({"type": "syntax_correction", "stage": 1, "outcome": "success"})
        return state

    failure_msg = "❌ Holistic validation failed twice. Proceeding to Stage 2."
    print(f"    {failure_msg}")
    await _log_to_markdown(log_file_path, f"### Status: Failed\n{failure_msg}\n**Final Suggestions:**\n```\n{suggestions}\n```")
    
    # --- Stage 2: Step-by-Step Validation ---
    await _log_to_markdown(log_file_path, "## Stage 2: Step-by-Step Validation")
    print("  -> Stage 2: Step-by-Step Validation")
    stage_2_retries = 30
    corrected_steps = []
    current_full_code = corrected_code  # Start with the last code we tried

    for i in range(1, 6):
        step_name = f"S{i}"
        per_step_correction_loop = 5
        step_validated = False
        await _log_to_markdown(log_file_path, f"### Validating Step: {step_name}")

        while per_step_correction_loop > 0:
            iteration_msg = f"Attempt {6 - per_step_correction_loop}/5 for step {step_name}. Total retries left: {stage_2_retries}"
            await _log_to_markdown(log_file_path, f"**{iteration_msg}**")
            if stage_2_retries <= 0:
                final_fail_msg = "❌ Stage 2 failed: Ran out of total retries. Culling branch."
                print(f"    {final_fail_msg}")
                await _log_to_markdown(log_file_path, f"#### FINAL STATUS: CULLED\n{final_fail_msg}")
                state['status'] = 'culled_syntax_failure'
                state['event_log'].append({"type": "syntax_failure", "reason": "Max retries exceeded"})
                return state
            
            # Extraction using the imported function
            all_steps = extract_step_blocks(current_full_code)
            extracted_content = all_steps.get(step_name)
            
            if not extracted_content:
                error_msg = f"Could not extract content for {step_name}."
                print(f"      -> {error_msg} Asking LLM to regenerate step.")
                await _log_to_markdown(log_file_path, f"#### Step {step_name} Issue: Tag Extraction Failed\n`{error_msg}`")
                
                llm_error_msg = f"Could not find the tag for step {step_name} or it was empty. Please provide the content for this step, ensuring it is enclosed in the correct tags, for example: <{step_name}>...code...</{step_name}>."
                correction_prompt = "Please correct the following algorithm based on the error."
                correction_context = f"ALGORITHM:\n```\n{current_full_code}\n```\n\nERROR:\n{llm_error_msg}"
                current_full_code = await call_vllm_api(correction_prompt, correction_context, f"Fix Missing Tag {step_name}")
                per_step_correction_loop -= 1
                stage_2_retries -= 1
                continue
            
            # Validation
            is_syntax_valid, syntax_error = validate_algorithm_syntax_main(extracted_content)
            is_usage_correct, usage_error = True, None
            if i == 5:
                is_usage_correct, usage_error = check_s5_point_usage(extracted_content)

            if is_syntax_valid and is_usage_correct:
                success_msg = f"✅ Validation passed for step {step_name}."
                print(f"      {success_msg}")
                await _log_to_markdown(log_file_path, f"#### Step {step_name} Status: Success")
                corrected_steps.append(extracted_content)
                step_validated = True
                break
            
            # Correction
            error_msg = f"⚠️ Validation failed for step {step_name}. Asking LLM to correct step."
            print(f"      {error_msg}")
            
            combined_errors = []
            if syntax_error: combined_errors.append(f"Syntax Error: {syntax_error}")
            if usage_error: combined_errors.append(f"Usage Error: {usage_error}")
            error_string = "\n".join(combined_errors)
            await _log_to_markdown(log_file_path, f"#### Step {step_name} Issue: Validation Failed\n**Errors:**\n```\n{error_string}\n```")

            correction_prompt = f"The following algorithm has an error in step {step_name}. Please correct ONLY the content within the <{step_name}>...</{step_name}> tags based on the error, keeping the rest of the algorithm identical."
            correction_context = f"ALGORITHM:\n```\n{current_full_code}\n```\n\nERROR(S) IN {step_name}:\n{error_string}"
            current_full_code = await call_vllm_api(correction_prompt, correction_context, f"Correct Step {step_name}")
            per_step_correction_loop -= 1
            stage_2_retries -= 1

        if not step_validated:
            final_fail_msg = f"❌ Stage 2 failed: Could not validate step {step_name} after 5 attempts. Culling branch."
            print(f"    {final_fail_msg}")
            await _log_to_markdown(log_file_path, f"### FINAL STATUS: CULLED\n{final_fail_msg}")
            state['status'] = 'culled_syntax_failure'
            state['event_log'].append({"type": "syntax_failure", "reason": f"Failed to validate step {step_name}"})
            return state

    # Final Assembly
    success_msg = "✅ All steps passed Stage 2 validation. Assembling final code."
    print(f"    {success_msg}")
    await _log_to_markdown(log_file_path, f"## Stage 2: Success\n{success_msg}")
    final_code = "\n".join([f"<{f'S{i+1}'}>{step}</{f'S{i+1}'}>" for i, step in enumerate(corrected_steps)])
    state['modified_code_proposal'] = final_code
    state['status'] = 'pending_execution'
    state['event_log'].append({"type": "syntax_correction", "stage": 2, "outcome": "success"})
    
    return state


async def execute_and_evaluate(state: BranchState) -> BranchState:
    """Performs execution on a single, validated branch, logging the outcome."""
    branch_id = state['branch_id']
    generation = state['generation']
    log_content = ""
    print(f"\n⚙️  Executing and Evaluating Branch {branch_id}...")
    
    # Create a dedicated output folder for this specific execution run
    output_example_json_folder_path = MAIN_DEBUG_FOLDER / branch_id / str(generation)
    output_example_json_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"  -> Created output directory for execution results: {output_example_json_folder_path}")

    is_correct, score, summary = execute_grid_algorithm_full(
        state['modified_code_proposal'], 
        TEMP_DIR,
        output_example_json_folder_path=output_example_json_folder_path
    )
    is_improvement = score > state['last_successful_score']
    
    if is_correct and is_improvement:
        print(f"  ✅ Execution successful for {branch_id}. New Score: {score}")
        state.update({
            "status": 'successful', 
            "algorithm_code": state['modified_code_proposal'], 
            "last_successful_score": score
        })
        critique = await call_vllm_api("Critique this successful algorithm.", state['algorithm_code'], "Critique Algorithm")
        state['critique_and_ideas'] = critique

        # --- Logging for Successful Execution ---
        log_content = (
            f"# Branch: {branch_id}\n\n"
            f"## Generation: {generation}\n\n"
            f"### Execution Successful\n\n"
            f"**Score:** {score:.4f}\n\n"
            f"**Information:** Algorithm was correct and improved upon the parent's score."
        )
    else:
        failure_reason = f"score {score:.4f} did not improve" if is_correct else f"was incorrect ({summary})"
        print(f"  ❌ Execution failed or did not improve for {branch_id}. Reason: {failure_reason}.")
        state["status"] = 'culled_execution_failure'
        state["event_log"].append({"type": "execution_failure", "reason": failure_reason})

        # --- Logging for Failed Execution ---
        log_content = (
            f"# Branch: {branch_id}\n\n"
            f"## Generation: {generation}\n\n"
            f"### Execution Failed\n\n"
            f"**Reason:**\n```\n{failure_reason}\n```"
        )

    # --- Write the Markdown Log for Validation/Execution Stage ---
    if log_content:
        try:
            log_file_path = MAIN_DEBUG_FOLDER / f"{branch_id}_execution_log.md"
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(log_content)
            print(f"  📝 Execution log saved to: {log_file_path}")
        except Exception as e:
            print(f"  ❌ Failed to write execution log file for branch {branch_id}. Error: {e}")

    return state

# --- Graph Definition ---
proposal_workflow = StateGraph(BranchState)
proposal_workflow.add_node("plan_and_implement", plan_and_implement_one)
proposal_workflow.add_node("validate_syntax", validate_syntax)
proposal_workflow.add_node("execute_and_evaluate", execute_and_evaluate)

def route_after_planning(state: BranchState) -> str:
    return "validate_syntax" if state['status'] == 'pending_validation' else END

def route_after_validation(state: BranchState) -> str:
    return "execute_and_evaluate" if state['status'] == 'pending_execution' else END

proposal_workflow.set_entry_point("plan_and_implement")
proposal_workflow.add_conditional_edges("plan_and_implement", route_after_planning)
proposal_workflow.add_conditional_edges("validate_syntax", route_after_validation)
proposal_workflow.add_edge("execute_and_evaluate", END)

proposal_app = proposal_workflow.compile()

# --- Main Graph Nodes (Orchestration Layer) ---

async def manage_and_dispatch(state: GlobalState) -> Dict[str, Any]:
    """
    Prepares and dispatches all parallel subgraph invocations for the current generation.
    This node now acts as the "stable state" before parallel execution begins.
    """
    generation = state['generation_count']
    print("\n" + "="*50 + f"\n🗓️  MANAGING GENERATION {generation}\n" + "="*50)

    tasks = []
    branches_to_evolve = state.get("active_branches", [])
    plan_history = state.get('proposal_plans_history', {})

    for parent_branch in branches_to_evolve:
        for i in range(PROPOSALS_PER_BRANCH):
            parent_id = parent_branch['branch_id']
            print(f"  -> Scheduling proposal {i+1}/{PROPOSALS_PER_BRANCH} for parent {parent_id}")
            
            history_for_task = plan_history.get(parent_id, "No previous plans exist for this branch lineage.")
            new_branch_id = f"b{uuid.uuid4().hex[:4]}_g{generation}"
            
            # Create the initial state for this specific, isolated subgraph run.
            branch_input = BranchState(
                branch_id=new_branch_id,
                generation=generation,
                parent_branch_id=parent_id,
                status="pending_proposal",
                algorithm_code=parent_branch['algorithm_code'],
                critique_and_ideas=parent_branch['critique_and_ideas'],
                last_successful_score=parent_branch['last_successful_score'],
                plan_history_for_task=history_for_task,
                modified_code_proposal=None, event_log=[], updated_plan_summary=None
            )
            # Instead of Send, we now use asyncio.create_task to run the subgraphs concurrently.
            # This is a more standard Python concurrency pattern.
            tasks.append(asyncio.create_task(proposal_app.ainvoke(branch_input)))
    
    # Wait for all parallel branches to complete their execution.
    results = await asyncio.gather(*tasks)
    
    # The results are now available and can be added to the state for the next step.
    return {"current_generation_results": results, "generation_count": generation + 1}


async def aggregate_and_select(state: GlobalState) -> Dict[str, Any]:
    """
    Aggregates results from the completed parallel runs and selects the fittest.
    This node acts as the "stable state" after parallel execution has finished.
    """
    generation = state['generation_count'] - 1 # We incremented it in the previous step
    print("\n" + "="*50 + f"\n🏆 AGGREGATING RESULTS (End of Gen {generation})\n" + "="*50)
    
    results = state['current_generation_results']
    plan_history = state.get('proposal_plans_history', {})

    # Ensure the debug folder for jsons exists
    MAIN_DEBUG_FOLDER_jsons.mkdir(parents=True, exist_ok=True)

    for res in results:
        parent_id = res.get('parent_branch_id')
        updated_summary = res.get('updated_plan_summary')
        if parent_id and updated_summary:
            plan_history[parent_id] = updated_summary

        # Save each successful branch state in a separate file
        if res.get('status') == 'successful':
            score = res.get('last_successful_score', -1.0)
            branch_id = res.get('branch_id', 'unknown')
            # Format score for filename (avoid issues with dots)
            score_str = f"{score:.4f}".replace('.', '_')
            filename = f"{score_str}_{branch_id}.json"
            file_path = MAIN_DEBUG_FOLDER_jsons / filename
            with open(file_path, "w") as f:
                json.dump(res, f, indent=4)
            print(f"  📁 Saved successful branch to: {file_path}")

    successful_branches = [b for b in results if b['status'] == 'successful']
    successful_branches.sort(key=lambda b: b['last_successful_score'], reverse=True)
    
    next_gen_active_branches = successful_branches[:state['top_k_branches']]

    print(f"Processed {len(results)} branches. Found {len(successful_branches)} successful ones.")
    
    if not next_gen_active_branches:
        if state.get('active_branches'):
            print("🛑 No branches survived. Restarting generation with the same parents.")
            next_gen_active_branches = state['active_branches']
        else:
            print("💀 Total extinction event. Re-seeding with the initial progenitor.")
            next_gen_active_branches = [get_initial_branch()]
    else:
        print(f"Selecting top {len(next_gen_active_branches)} for next generation:")
        for b in next_gen_active_branches:
            print(f"  - Branch {b['branch_id']} -> Score: {b['last_successful_score']:.4f}")

    return {
        "active_branches": next_gen_active_branches,
        "branches": state['branches'] + results,
        "current_generation_results": [], # Clear results for the next generation
        "proposal_plans_history": plan_history
    }

def get_initial_branch() -> BranchState:
    """Creates the progenitor branch."""
    print("🌱 Creating the initial progenitor branch...")
    return BranchState(
        branch_id="b0_g0", parent_branch_id=None, status='successful',
        generation=0,
        algorithm_code=INITIAL_ALGORITHM, modified_code_proposal=None,
        last_successful_score=-1.0, 
        critique_and_ideas=INITIAL_ALGORITHM_CRITIQUE,
        event_log=[{"type": "creation", "outcome": "success"}],
        plan_history_for_task=None, updated_plan_summary=None,
    )

async def main():
    """
    Main async function to set up and run the graph with stable checkpointing.
    """
    MAIN_DEBUG_FOLDER.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    
    # The main graph orchestrates the stable steps.
    main_workflow = StateGraph(GlobalState)
    main_workflow.add_node("manage_and_dispatch", manage_and_dispatch)
    main_workflow.add_node("aggregate_and_select", aggregate_and_select)
    main_workflow.set_entry_point("manage_and_dispatch")
    main_workflow.add_edge("manage_and_dispatch", "aggregate_and_select")
    
    def should_continue(state: GlobalState):
        return "manage_and_dispatch" if state['generation_count'] < MAX_GENERATIONS else END

    main_workflow.add_conditional_edges("aggregate_and_select", should_continue)
    
    # The checkpointer is now attached to the main orchestration graph.
    # It will save the state ONLY after 'manage_and_dispatch' and 'aggregate_and_select' complete.
    db_path = MAIN_DEBUG_FOLDER / "main_state.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as memory:
        app = main_workflow.compile(checkpointer=memory)

        config = {"configurable": {"thread_id": "genetic-algo-run-stable-1"}}
        thread_state = None
        try:
            thread_state = await app.aget_state(config)
        except Exception:
            print("Could not retrieve state, starting fresh.")

        if not thread_state or not thread_state.values:
            print("No existing state found. Creating a new one.")
            initial_branch = get_initial_branch()
            # The initial state is now simpler
            initial_state = GlobalState(
                generation_count=0, top_k_branches=TOP_K_BRANCHES,
                active_branches=[initial_branch], current_generation_results=[], 
                branches=[], proposal_plans_history={}
            )
            await app.ainvoke(initial_state, config=config)
        else:
            print(f"Resuming from Generation {thread_state.values.get('generation_count', 0)}.")
            # On resume, invoke with the last known stable state.
            await app.ainvoke(thread_state.values, config=config)

        print("\nWorkflow has completed or reached the max generation limit.")
        
        final_graph_state = await app.aget_state(config)
        if final_graph_state:
            all_branches = final_graph_state.values.get('branches', [])
            successful_branches = [b for b in all_branches if b and b.get('status') == 'successful']
            if successful_branches:
                best_branch = max(successful_branches, key=lambda b: b.get('last_successful_score', -999))
                print("\n👑 Overall Best Performing Branch:")
                print(f"  - Branch ID:    {best_branch['branch_id']}")
                print(f"  - Final Score:  {best_branch['last_successful_score']:.4f}")
                best_branch_file = MAIN_DEBUG_FOLDER / f"BEST_{best_branch['branch_id']}.json"
                with open(best_branch_file, "w") as f:
                    json.dump(best_branch, f, indent=4)
                print(f"\n📋 Best branch state saved to: {best_branch_file}")

if __name__ == "__main__":
    asyncio.run(main())

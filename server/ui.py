from typing import Dict, Any

from models import HiringAction


def _render_observation(obs) -> str:
    if obs is None:
        return "No active episode. Click Reset."
    lines = []
    lines.append(f"Step: {obs.step_num} / {obs.max_steps}")
    lines.append(f"Budget remaining: {obs.budget_remaining:.0f} (interview=10, probe=20-30, hire=50)")
    lines.append(f"Hired: {', '.join(obs.hires_made) if obs.hires_made else 'none'}")
    lines.append(f"Skipped: {', '.join(obs.skipped) if obs.skipped else 'none'}")
    lines.append("")
    lines.append("Candidates:")
    for c in obs.candidates:
        cid = c["candidate_id"]
        if cid in obs.hires_made or cid in obs.skipped:
            continue
        interview_str = ""
        if cid in obs.interviews_done:
            interview_str = f" | INTERVIEWED: {obs.interviews_done[cid]:.3f}"
        if cid in obs.probes_done:
            interview_str += (
                f" | PROBED: {obs.probes_done[cid]:.3f}"
                f" | gap={obs.probe_gaps.get(cid, 0.0):+.3f}"
            )
        lines.append(
            f"{cid}: {c['name']} — resume={c['resume_score']:.2f} exp={c['years_experience']}yrs{interview_str}"
        )
    if obs.last_action_result:
        lines.append("")
        lines.append(f"Last: {obs.last_action_result}")
    return "\n".join(lines)


def mount_ui(app, env):
    """Mounts a Gradio UI under /ui that interacts with the running `env` instance.

    The UI calls `env.reset()` and `env.step()` directly (same process) so it's
    lightweight and works in Spaces where the app runs in a single container.
    """

    try:
        import gradio as gr
    except Exception as e:
        print(f"[WARN] Gradio import failed: {e}")
        return

    # tasks list comes from server.tasks
    try:
        from server.tasks import TASKS
    except Exception:
        TASKS = {}

    with gr.Blocks() as demo:
        gr.Markdown("## AgentHire Arena — Interactive UI")

        with gr.Row():
            task_dropdown = gr.Dropdown(choices=list(TASKS.keys()), value=list(TASKS.keys())[0] if TASKS else None, label="Task")
            reset_btn = gr.Button("Reset")

        obs_md = gr.Markdown("No active episode.")

        with gr.Row():
            candidate_dropdown = gr.Dropdown(choices=[], label="Candidate")
            action_radio = gr.Radio(choices=["interview", "probe", "hire", "skip", "finalize"], value="interview", label="Action")
            step_btn = gr.Button("Step")

        metrics_md = gr.Markdown("")

        def do_reset(task):
            obs = env.reset(task=task)
            # Refresh candidate dropdown
            ids = [c["candidate_id"] for c in obs.candidates if c["candidate_id"] not in obs.hires_made and c["candidate_id"] not in obs.skipped]
            return _render_observation(obs), gr.update(choices=ids, value=ids[0] if ids else None), ""

        def do_step(candidate_id, action):
            if action == "finalize":
                cid = None
            else:
                cid = candidate_id
            try:
                obs, reward = env.step(HiringAction(action=action, candidate_id=cid))
            except Exception as e:
                return _render_observation(env._state), gr.update(choices=[]), f"Error: {e}"
            ids = [c["candidate_id"] for c in obs.candidates if c["candidate_id"] not in obs.hires_made and c["candidate_id"] not in obs.skipped]
            metrics = f"Reward: {reward.step_reward:+.2f} — {reward.reason}\nBudget: {obs.budget_remaining:.0f}\nHires: {', '.join(obs.hires_made) if obs.hires_made else 'none'}"
            return _render_observation(obs), gr.update(choices=ids, value=ids[0] if ids else None), metrics

        reset_btn.click(do_reset, inputs=[task_dropdown], outputs=[obs_md, candidate_dropdown, metrics_md])
        step_btn.click(do_step, inputs=[candidate_dropdown, action_radio], outputs=[obs_md, candidate_dropdown, metrics_md])

    # Mount at /
    return gr.mount_gradio_app(app, demo, path="/")

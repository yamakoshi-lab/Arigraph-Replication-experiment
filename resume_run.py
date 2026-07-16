import sys
import os
import re
import json
import ast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import Logger, observation_processing, find_unexplored_exits, \
    simulate_environment_actions, action_processing, action_deprocessing, \
    find_direction, find_opposite_direction, parse_triplets_removing
from utils.win_cond import win_cond_clean_place, win_cond_clean_take
from utils.textworld_adapter import TextWorldWrapper, graph_from_facts
from graphs.contriever_graph import ContrieverGraph

import pipeline_arigraph as pl  # runs the "changeable part" (agents, env_name, model, api_key, etc.) on import


def extract_json_block(text, start_idx):
    """Find the {...} JSON object starting at or after start_idx, return (obj_text, end_idx)."""
    brace_start = text.index("{", start_idx)
    depth = 0
    i = brace_start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start:i + 1], i + 1
        i += 1
    raise ValueError("Unbalanced JSON block")


def unstr_triplet(s):
    """Reverse ContrieverGraph.str(): 'subj, label, obj' -> [subj, obj, {'label': label}]"""
    parts = s.split(", ")
    if len(parts) < 3:
        return None
    subj = parts[0]
    obj = parts[-1]
    label = ", ".join(parts[1:-1])
    return [subj, obj, {"label": label}]


def parse_log(log_path):
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    steps = []
    step_positions = [m.start() for m in re.finditer(r"\nStep: \d+\n", text)]
    step_positions.append(len(text))

    for idx in range(len(step_positions) - 1):
        block = text[step_positions[idx]:step_positions[idx + 1]]
        step_num = int(re.search(r"Step: (\d+)", block).group(1))

        obs_match = re.search(r"\nObservation: (.*?)\nInventory:", block, re.S)
        observation = obs_match.group(1) if obs_match else None

        new_trip_match = re.search(r"\nNew triplets: (\[.*?\])\n", block)
        new_triplets_str_list = ast.literal_eval(new_trip_match.group(1)) if new_trip_match else []

        outdated_match = re.search(r"\nOutdated triplets: (.*?)\nNUMBER OF REPLACEMENTS:", block, re.S)
        outdated_raw_text = outdated_match.group(1) if outdated_match else "[]"

        action_marker = block.find("\nAction: {")
        if action_marker == -1:
            # Truncated trailing block (process was killed mid-step); drop it.
            continue
        action_json_text, _ = extract_json_block(block, action_marker)
        try:
            action_obj = json.loads(action_json_text)
            action_taken = action_obj.get("action_to_take")
        except Exception:
            m = re.search(r'"action_to_take":\s*"([^"]*)"', action_json_text)
            action_taken = m.group(1) if m else None
        if not action_taken:
            continue

        plan_marker = block.find("\nPlan0: {")
        plan_json_text = None
        if plan_marker != -1:
            plan_json_text, _ = extract_json_block(block, plan_marker)

        reward_match = re.search(r"TOTAL REWARDS: (\[.*?\])\n", block)
        rewards_so_far = ast.literal_eval(reward_match.group(1)) if reward_match else None

        steps.append(dict(
            step_num=step_num,
            observation=observation,
            new_triplets_str_list=new_triplets_str_list,
            outdated_raw_text=outdated_raw_text,
            action_taken=action_taken,
            plan0_text=plan_json_text,
            rewards_so_far=rewards_so_far,
        ))
    return steps


def process_action_get_reward(action, env, info, graph, locations, env_name):
    G_true = graph_from_facts(info)
    full_graph = G_true.edges(data=True)
    step_reward = 0
    is_nav = "go to" in action
    done = False
    if is_nav:
        destination = action.split('go to ')[1]
        path = graph.find_path(observation_processing(env.curr_location).lower(), destination, locations)
        if not isinstance(path, list):
            raise RuntimeError(f"Navigation replay failed: {path}")
        observation = None
        for hidden_action in path:
            observation, reward_, done, info = env.step(hidden_action)
            step_reward += reward_
            if done:
                break
    else:
        observation, reward_, done, info = env.step(action)
        step_reward += reward_

    G_true_new = graph_from_facts(info)
    full_graph_new = G_true_new.edges(data=True)
    step_reward = simulate_environment_actions(full_graph, full_graph_new, win_cond_clean_take, win_cond_clean_place) \
        if env_name == "clean" else step_reward
    return observation, step_reward, done, info


def reconstruct(log_path):
    steps = parse_log(log_path)
    print(f"Parsed {len(steps)} historical steps from {log_path}")

    env = TextWorldWrapper(pl.ENV_NAMES[pl.env_name])
    observation, info = env.reset()
    graph = ContrieverGraph(pl.model, system_prompt="You are a helpful assistant", device=pl.retriever_device,
                             api_key=pl.api_key, base_url=pl.base_url, price_per_1m=pl.price_per_1m)

    observations, history = [], []
    locations = set()
    action = "start"
    plan0 = None
    subgraph = []
    previous_location = observation_processing(env.curr_location).lower()
    reward, rewards = 0, []
    done = False

    for i, s in enumerate(steps):
        locations.add(observation_processing(env.curr_location).lower())

        new_triplets_raw = [unstr_triplet(x) for x in s["new_triplets_str_list"]]
        new_triplets_raw = [t for t in new_triplets_raw if t is not None]

        predicted_outdated = parse_triplets_removing(s["outdated_raw_text"])
        graph.delete_triplets(predicted_outdated, locations)

        # NOTE: 'action' here is the PREVIOUS step's action (matches run()'s loop variable
        # lifecycle: graph.update() uses the action that led to the CURRENT location, and
        # only after planning/action-choice does 'action' get reassigned to the new one).
        curr_location = observation_processing(env.curr_location).lower()
        if "go to" not in action:
            if curr_location != previous_location:
                new_triplets_raw.append([curr_location, previous_location, {"label": find_direction(action)}])
                new_triplets_raw.append([previous_location, curr_location, {"label": find_opposite_direction(action)}])
        graph.add_triplets(new_triplets_raw)

        if s["observation"] is not None:
            obs_embedding = graph.retriever.embed(s["observation"])
            new_triplets_str = graph.convert(new_triplets_raw)
            graph.obs_episodic[s["observation"]] = [new_triplets_str, obs_embedding]

        if s["plan0_text"]:
            plan0 = s["plan0_text"]

        observations.append(s["observation"])
        observations = observations[-pl.n_prev:]
        history.append(f"Observation: {s['observation']}\nAction taken: {s['action_taken']}")
        history = history[-pl.n_prev:]
        previous_location = observation_processing(env.curr_location).lower()

        action = s["action_taken"]
        try:
            observation, step_reward, done, info = process_action_get_reward(action, env, info, graph, locations, pl.env_name)
        except RuntimeError as e:
            print(f"FAILED at replay index {i}, logged step_num={s['step_num']}, action={action!r}: {e}")
            raise
        reward += step_reward
        rewards.append(reward)

        if s["rewards_so_far"] is not None:
            expected = s["rewards_so_far"][-1]
            if abs(expected - reward) > 1e-6:
                print(f"WARNING: reward drift at step {s['step_num']}: replayed={reward} logged={expected}")

        if done:
            print(f"Game already finished during replay at logged step {s['step_num']}!")
            break

    print(f"Reconstruction complete. Resuming from step {len(steps)}, reward={reward}, locations={sorted(locations)}")
    return env, graph, observations, history, locations, plan0, previous_location, reward, rewards, action, done, info, len(steps), observation


if __name__ == "__main__":
    log_path = sys.argv[1]
    extra_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    env, graph, observations, history, locations, plan0, previous_location, reward, rewards, action, done, info, start_step, observation = reconstruct(log_path)

    if done:
        print("Nothing to resume, game already concluded.")
        sys.exit(0)

    log = Logger(pl.log_file + "_resumed")

    for step in range(start_step, start_step + extra_steps):
        observation = observation.split("$$$")[-1] if isinstance(observation, str) else ""
        observation = observation_processing(observation)
        observation = "Game step #" + str(step + 1) + "\n" + observation
        inventory = env.get_inventory()

        if done:
            log("Game itog: " + observation)
            break

        log("Observation: " + observation)
        log("Inventory: " + str(inventory))
        locations.add(observation_processing(env.curr_location).lower())

        observed_items, _ = pl.agent.item_processing_scores(observation, plan0)
        items = {key.lower(): value for key, value in observed_items.items()}
        log("Crucial items: " + str(items))

        subgraph, top_episodic = graph.update(observation, observations, plan=plan0, prev_subgraph=[],
                                               locations=list(locations), curr_location=observation_processing(env.curr_location).lower(),
                                               previous_location=previous_location, action=action, log=log, items1=items, topk_episodic=pl.topk_episodic)
        observation += f"\nInventory: {inventory}"
        log("Length of subgraph: " + str(len(subgraph)))
        log("Associated triplets: " + str(subgraph))
        log("Episodic memory: " + str(top_episodic))

        if_explore, _ = pl.agent_if_expl.generate(prompt=f"Plan: \n{plan0}", t=0.2) if pl.need_exp else ("False", 0)
        if_explore = "True" in if_explore
        log('If explore: ' + str(if_explore))

        all_unexpl_exits = pl.get_unexpl_exits(locations, graph) if if_explore else ""
        if if_explore:
            log(all_unexpl_exits)

        valid_actions = [action_processing(a) for a in env.get_valid_actions()] + env.expand_action_space() if "cook" in pl.env_name else env.get_valid_actions()
        valid_actions += [f"go to {loc}" for loc in locations]
        log("Valid actions: " + str(valid_actions))
        hist_obs = "\n".join([h for h in history if h])

        plan0 = pl.planning(hist_obs, observation, plan0, subgraph, top_episodic, if_explore, all_unexpl_exits)
        action = pl.choose_action(hist_obs, observation, subgraph, top_episodic, plan0, all_unexpl_exits, valid_actions, if_explore)

        observations.append(observation)
        observations = observations[-pl.n_prev:]
        history.append(f"Observation: {observation}\nAction taken: {action}")
        history = history[-pl.n_prev:]
        previous_location = observation_processing(env.curr_location).lower()

        observation, step_reward, done, info = process_action_get_reward(action, env, info, graph, locations, pl.env_name)
        reward += step_reward
        rewards.append(reward)
        print(f"Step {step + 1}: action={action!r} step_reward={step_reward} total_reward={reward}")
        log(f"\n\nTOTAL REWARDS: {rewards}\n\n")

    print(f"DONE resuming. final_reward={reward} rewards={rewards}")

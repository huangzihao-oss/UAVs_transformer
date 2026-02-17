import os
from datetime import datetime
import argparse

import numpy as np

from PPO import PPO
from Env import MultiDroneAoIEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for asynchronous multi-UAV PPO Transformer")

    # Environment hyperparameters
    parser.add_argument("--env_name", type=str, default="A-MAPPO", help="Name of the gym environment")
    parser.add_argument("--max_ep_len", type=int, default=1000, help="Maximum timesteps in one episode")
    parser.add_argument("--max_training_timesteps", type=int, default=int(3e6), help="Maximum total timesteps")
    parser.add_argument("--print_freq", type=int, default=None, help="Print average reward interval")
    parser.add_argument("--log_freq", type=int, default=None, help="Log average reward interval")
    parser.add_argument("--save_model_freq", type=int, default=int(1e5), help="Model saving frequency")

    # PPO hyperparameters
    parser.add_argument("--update_timestep", type=int, default=None, help="Policy update frequency")
    parser.add_argument("--K_epochs", type=int, default=5, help="Number of PPO epochs")
    parser.add_argument("--update_every_episodes", type=int, default=2, help="Update PPO every N episodes")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="Clip parameter for PPO")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr_actor", type=float, default=0.0001, help="Learning rate for actor")
    parser.add_argument("--lr_critic", type=float, default=0.0001, help="Learning rate for critic")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed (0 = no random seed)")
    parser.add_argument("--entropy_ratio", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--gae-lambda", dest="gae_lambda", type=float, default=0.97, help="Lambda for GAE")
    parser.add_argument("--gae_flag", action="store_true", help="Use GAE")

    # Transformer hyperparameters
    parser.add_argument("--history_horizon", type=int, default=20, help="History horizon for [s,a,r] tokens")
    parser.add_argument("--speed_levels", type=str, default="6-20", help="Discrete speed levels, e.g. 6-20 or 6-20:1")
    parser.add_argument("--transformer_dim", type=int, default=128, help="Transformer hidden dim")
    parser.add_argument("--transformer_heads", type=int, default=2, help="Transformer attention heads")
    parser.add_argument("--transformer_layers", type=int, default=2, help="Transformer encoder/decoder layers")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="Transformer dropout")

    # Logging and checkpointing
    parser.add_argument("--log_dir", type=str, default="PPO_logs", help="Directory for log files")
    parser.add_argument("--checkpoint_dir", type=str, default="PPO_preTrained", help="Directory for checkpoints")
    parser.add_argument("--run_num_pretrained", type=int, default=0, help="Run number for pretrained model")

    # Environment parameters
    parser.add_argument("--M", type=int, default=2, help="Num of UAVs")
    parser.add_argument("--N", type=int, default=1, help="Num of Bases")
    parser.add_argument("--K", type=int, default=20, help="Num of PoIs")
    parser.add_argument("--position_file", type=str, default=None, help="Path to .npy map file")
    parser.add_argument("--T", type=int, default=1800, help="Time limits")
    parser.add_argument("--BS_back_times", type=int, default=5, help="Buffer threshold for return-to-BS penalty")
    parser.add_argument("--map_size", type=int, default=1000, help="Map size")
    parser.add_argument("--init_uav_energy", type=float, default=3.0e5, help="Initial energy budget per UAV")
    parser.add_argument("--pre_reward_ratio", type=float, default=3.0, help="Ratio of pre reward")
    parser.add_argument("--buffer_punishment", action="store_true", help="Enable buffer punishment")
    parser.add_argument("--punishment_value", type=float, default=4.0, help="Punishment scale")
    parser.add_argument("--reward_scale_size", type=float, default=10000.0, help="Reward scale")
    parser.add_argument("--reward_divisor", type=float, default=10.0, help="Additional reward divisor")
    parser.add_argument("--print_info", action="store_true", help="Print env step details")

    args = parser.parse_args()
    args.print_freq = args.print_freq if args.print_freq is not None else args.max_ep_len * 10
    args.log_freq = args.log_freq if args.log_freq is not None else args.max_ep_len * 2
    args.update_timestep = args.update_timestep if args.update_timestep is not None else args.max_ep_len * 4
    return args


def train():
    args = parse_args()
    print("============================================================================================")

    env_name = args.env_name
    max_ep_len = args.max_ep_len
    max_training_timesteps = args.max_training_timesteps
    print_freq = args.print_freq
    log_freq = args.log_freq
    save_model_freq = args.save_model_freq

    K_epochs = args.K_epochs
    eps_clip = args.eps_clip
    gamma = args.gamma
    lr_actor = args.lr_actor
    lr_critic = args.lr_critic
    random_seed = args.random_seed

    tensorboard_dir = os.path.join("logs", env_name)

    env = MultiDroneAoIEnv(
        args.M, args.N, args.K, args.T, args.map_size, args=args, position_file=args.position_file
    )

    token_dim = env.token_dim
    target_action_dim = env.target_action_dim
    speed_action_dim = env.speed_action_dim

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, env_name)
    os.makedirs(log_dir, exist_ok=True)

    run_num = len(next(os.walk(log_dir))[2])
    log_f_name = os.path.join(log_dir, f"PPO_{env_name}_log_{run_num}.csv")
    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    directory = args.checkpoint_dir
    os.makedirs(directory, exist_ok=True)
    directory = os.path.join(directory, env_name)
    os.makedirs(directory, exist_ok=True)
    checkpoint_prefix = os.path.join(directory, f"PPO_{env_name}_{random_seed}_{args.run_num_pretrained}")
    print("save checkpoint prefix : " + checkpoint_prefix)

    ppo_agent = []
    for i in range(args.M):
        ppo_agent.append(
            PPO(
                agent_id=i,
                token_dim=token_dim,
                target_action_dim=target_action_dim,
                speed_action_dim=speed_action_dim,
                max_encoder_len=env.encoder_token_len,
                max_decoder_len=env.decoder_token_len,
                max_critic_len=env.critic_token_len,
                max_other_agents=env.max_other_agents,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                K_epochs=K_epochs,
                eps_clip=eps_clip,
                summary_dir=tensorboard_dir,
                entropy_ratio=args.entropy_ratio,
                gae_lambda=args.gae_lambda,
                gae_flag=args.gae_flag,
                d_model=args.transformer_dim,
                nhead=args.transformer_heads,
                num_layers=args.transformer_layers,
                dropout=args.transformer_dropout,
            )
        )

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    log_f = open(log_f_name, "w+")
    log_f.write("episode,timestep,reward\n")

    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    episodes_since_update = 0
    speed_segment_step = 0

    def log_avg_speed_segment():
        speed_values = []
        for agent in ppo_agent:
            for speed_idx in agent.buffer.speed_actions:
                idx = int(speed_idx.item()) if hasattr(speed_idx, "item") else int(speed_idx)
                idx = int(np.clip(idx, 0, speed_action_dim - 1))
                speed_values.append(float(env.speed_levels[idx]))
        if len(speed_values) == 0:
            return None
        return float(np.mean(speed_values))

    while time_step <= max_training_timesteps:
        env.reset()
        current_ep_reward = [0.0] * args.M
        done_tag = 0

        uav_actions_queue = [None for _ in range(args.M)]
        target_masks = np.ones((args.M, target_action_dim), dtype=np.float32)
        speed_masks = np.ones((args.M, speed_action_dim), dtype=np.float32)

        actions_set = [[] for _ in range(args.M)]
        positions_set = [[] for _ in range(args.M)]
        done_bool = np.zeros(args.M, dtype=np.int32)

        non_base_count = 0

        for _ in range(1, max_ep_len + 1):
            candidate_times = np.full(args.M, np.inf, dtype=np.float32)

            for i in range(args.M):
                if done_bool[i]:
                    continue

                if uav_actions_queue[i] is None:
                    obs_pack = env.get_transformer_inputs(i)
                    mask_pack = {"target": target_masks[i], "speed": speed_masks[i]}

                    if non_base_count < 100:
                        uav_actions_queue[i] = ppo_agent[i].select_action(obs_pack, mask_pack)
                        if uav_actions_queue[i][0] < env.K:
                            non_base_count += 1
                        else:
                            non_base_count = 0
                    else:
                        forced_action = (env.K + env.N - 1, env.default_speed_idx)
                        uav_actions_queue[i] = ppo_agent[i].select_action(
                            obs_pack, mask_pack, deter_action=forced_action
                        )
                        non_base_count = 0

                candidate_times[i] = env.drone_timing_now[i] + env.time_cost(i, uav_actions_queue[i])

            if np.isinf(candidate_times).all():
                break

            action_uav = int(np.argmin(candidate_times))

            _, reward, done, masked = env.step(action_uav, uav_actions_queue[action_uav])

            ppo_agent[action_uav].buffer.rewards.append(reward)
            ppo_agent[action_uav].buffer.is_terminals.append(done)

            actions_set[action_uav].append(uav_actions_queue[action_uav])
            positions_set[action_uav].append(env.drone_position_now[action_uav].copy())

            target_masks[action_uav] = masked["target"]
            speed_masks[action_uav] = masked["speed"]
            uav_actions_queue[action_uav] = None

            time_step += 1
            current_ep_reward[action_uav] += reward

            if done:
                done_tag += 1
                done_bool[action_uav] = 1

            if done_tag == args.M:
                format_string = "Episode: {}"
                values = [i_episode]
                rewards_sum = 0
                for idx, agent in enumerate(ppo_agent, 1):
                    agent_reward = current_ep_reward[idx - 1]
                    format_string += ", Agent {} len: {}, rewards {}: {}"
                    values.extend([idx, len(agent.buffer.rewards), idx, round(agent_reward, 6)])
                    rewards_sum += agent_reward
                format_string += " Sum of rewards: {}"
                values.append(round(rewards_sum, 6))
                print(format_string.format(*values))
                break

            if time_step % log_freq == 0 and log_running_episodes > 0:
                log_avg_reward = round(log_running_reward / log_running_episodes, 4)
                log_f.write("{},{},{}\n".format(i_episode, time_step, log_avg_reward))
                log_f.flush()
                log_running_reward = 0
                log_running_episodes = 0

            if time_step % print_freq == 0 and print_running_episodes > 0:
                print_avg_reward = round(print_running_reward / print_running_episodes, 2)
                print(
                    "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                        i_episode, time_step, print_avg_reward
                    )
                )
                print(
                    "Episode : {} \t\t Timestep : {} \t\t Current epi reward : {}".format(
                        i_episode, time_step, current_ep_reward
                    )
                )
                print_running_reward = 0
                print_running_episodes = 0

            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at prefix : " + checkpoint_prefix)
                for i in range(args.M):
                    ckpt = f"{checkpoint_prefix}_agent{i}.pth"
                    ppo_agent[i].save(ckpt)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if time_step > max_training_timesteps:
                break

        print_running_reward += current_ep_reward[0]
        print_running_episodes += 1
        log_running_reward += current_ep_reward[0]
        log_running_episodes += 1

        i_episode += 1
        episodes_since_update += 1

        if episodes_since_update >= max(1, args.update_every_episodes):
            avg_speed_segment = log_avg_speed_segment()
            if avg_speed_segment is not None:
                ppo_agent[0].writer.add_scalar("stats/avg_speed_segment", avg_speed_segment, speed_segment_step)
                speed_segment_step += 1
            for i in range(args.M):
                ppo_agent[i].update()
            episodes_since_update = 0

        if i_episode % 10 == 0:
            env.reset()
            target_masks = np.ones((args.M, target_action_dim), dtype=np.float32)
            speed_masks = np.ones((args.M, speed_action_dim), dtype=np.float32)
            uav_actions_queue = [None for _ in range(args.M)]

            done_tag = 0
            done_bool = np.zeros(args.M, dtype=np.int32)
            test_rewards = [0.0] * args.M
            test_positions = [[] for _ in range(args.M)]

            finish_step = max_ep_len
            test_logged = False
            for _ in range(1, finish_step + 1):
                candidate_times = np.full(args.M, np.inf, dtype=np.float32)

                for i in range(args.M):
                    if done_bool[i]:
                        continue
                    if uav_actions_queue[i] is None:
                        obs_pack = env.get_transformer_inputs(i)
                        mask_pack = {"target": target_masks[i], "speed": speed_masks[i]}
                        uav_actions_queue[i] = ppo_agent[i].action_test(obs_pack, mask_pack)
                    candidate_times[i] = env.drone_timing_now[i] + env.time_cost(i, uav_actions_queue[i])

                if np.isinf(candidate_times).all():
                    break

                action_uav = int(np.argmin(candidate_times))
                _, reward, done, masked = env.step(action_uav, uav_actions_queue[action_uav])

                target_masks[action_uav] = masked["target"]
                speed_masks[action_uav] = masked["speed"]
                test_rewards[action_uav] += reward
                test_positions[action_uav].append(env.drone_position_now[action_uav].copy())
                uav_actions_queue[action_uav] = None

                if done:
                    done_tag += 1
                    done_bool[action_uav] = 1

                if done_tag == args.M:
                    ppo_agent[0].call_2_record(i_episode // 10, float(sum(test_rewards)))
                    env.visualize_routes(test_positions, "./test_figs")
                    test_logged = True
                    break

            # 即使测试没有在finish_step内全部结束，也记录当前累计团队测试奖励
            if not test_logged:
                ppo_agent[0].call_2_record(i_episode // 10, float(sum(test_rewards)))

    # 训练结束前，若未到更新周期，仍执行一次更新，避免缓冲区样本丢失
    if episodes_since_update > 0:
        avg_speed_segment = log_avg_speed_segment()
        if avg_speed_segment is not None:
            ppo_agent[0].writer.add_scalar("stats/avg_speed_segment", avg_speed_segment, speed_segment_step)
            speed_segment_step += 1
        for i in range(args.M):
            ppo_agent[i].update()

    log_f.close()
    env.close()

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


# if __name__ == "__main__":
train()

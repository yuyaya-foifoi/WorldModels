import os
import sys

sys.path.append("./")

import gc
import time

import numpy as np
import pandas as pd
import pybullet_envs
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from configs import TRANSFERED_SINGLE_ENV_CONFIG_DICT as CONFIG_DICT
from src.agent import Agent
from src.buffer import ReplayBuffer
from src.loss import lambda_target
from src.model import RSSM, ActionModel, Encoder, ValueModel
from src.utils import make_env, preprocess_obs
from src.utils.date import get_str_currentdate
from src.utils.save import shutil_copy

seed_episodes = CONFIG_DICT["experiment"]["train"]["seed_episodes"]

all_episodes = CONFIG_DICT["experiment"]["train"]["all_episodes"]
action_noise_var = CONFIG_DICT["experiment"]["train"]["action_noise_var"]
collect_interval = CONFIG_DICT["experiment"]["train"]["collect_interval"]
batch_size = CONFIG_DICT["experiment"]["train"]["batch_size"]
chunk_length = CONFIG_DICT["experiment"]["train"]["chunk_length"]
batch_size = CONFIG_DICT["experiment"]["train"]["batch_size"]

test_interval = CONFIG_DICT["experiment"]["train"]["test_interval"]
model_save_interval = CONFIG_DICT["experiment"]["train"]["model_save_interval"]
collect_interval = CONFIG_DICT["experiment"]["train"]["collect_interval"]
action_noise_var = CONFIG_DICT["experiment"]["train"]["action_noise_var"]

batch_size = CONFIG_DICT["experiment"]["train"]["batch_size"]
chunk_length = CONFIG_DICT["experiment"]["train"]["chunk_length"]
imagination_horizon = CONFIG_DICT["experiment"]["train"]["imagination_horizon"]
gamma = CONFIG_DICT["experiment"]["train"]["gamma"]
lambda_ = CONFIG_DICT["experiment"]["train"]["lambda_"]
clip_grad_norm = CONFIG_DICT["experiment"]["train"]["clip_grad_norm"]
free_nats = CONFIG_DICT["experiment"]["train"]["free_nats"]

state_dim = CONFIG_DICT["model"]["state_dim"]
rnn_hidden_dim = CONFIG_DICT["model"]["rnn_hidden_dim"]

device = CONFIG_DICT["device"]

model_lr = CONFIG_DICT["experiment"]["train"]["model_lr"]
eps = CONFIG_DICT["experiment"]["train"]["eps"]
value_lr = CONFIG_DICT["experiment"]["train"]["value_lr"]
action_lr = CONFIG_DICT["experiment"]["train"]["action_lr"]

env_list = CONFIG_DICT["experiment"]["env_list"]
env_name = CONFIG_DICT["experiment"]["env_name"]
action_space_list = [
    make_env(env_name).action_space.shape[0] for env_name in env_list
]
observation_space_list = [
    make_env(env_name).observation_space.shape[0] for env_name in env_list
]
action_space = max(action_space_list)
observation_space = max(observation_space_list)

encoder = Encoder().to(device)
rssm = RSSM(state_dim, action_space, rnn_hidden_dim, device)
value_model = ValueModel(state_dim, rnn_hidden_dim).to(device)
action_model = ActionModel(state_dim, rnn_hidden_dim, action_space).to(device)


def get_fractional(saved_state, weight_key: str, ratio: float):
    fractional = saved_state[weight_key] * ratio
    return fractional


transfer_type = CONFIG_DICT["experiment"]["transfer_type"]
log_dir = CONFIG_DICT["experiment"]["transfer_path"]

if transfer_type == "fractional":

    encoder.load_state_dict(torch.load(os.path.join(log_dir, "encoder.pth")))

    action_model.load_state_dict(
        torch.load(os.path.join(log_dir, "action_model.pth"))
    )
    action_model.fc4.weight = nn.Parameter(
        torch.Tensor(action_model.fc4.weight.size())
    )
    action_model.fc4.bias = nn.Parameter(
        torch.Tensor(action_model.fc4.bias.size())
    )

    value_fc4_init_weight = value_model.fc4.weight
    value_fc4_init_bias = value_model.fc4.bias
    value_model.load_state_dict(
        torch.load(os.path.join(log_dir, "value_model.pth"))
    )
    value_model.fc4.weight = nn.Parameter(
        value_fc4_init_weight
        + get_fractional(value_model.state_dict(), "fc4.weight", 0.2)
    )
    value_model.fc4.bias = nn.Parameter(
        value_fc4_init_bias
        + get_fractional(value_model.state_dict(), "fc4.bias", 0.2)
    )

    reward_fc4_init_weight = rssm.reward.fc4.weight
    reward_fc4_init_bias = rssm.reward.fc4.bias
    rssm.reward.load_state_dict(
        torch.load(os.path.join(log_dir, "reward_model.pth"))
    )
    rssm.reward.fc4.weight = nn.Parameter(
        reward_fc4_init_weight
        + get_fractional(rssm.reward.state_dict(), "fc4.weight", 0.2)
    )
    rssm.reward.fc4.bias = nn.Parameter(
        reward_fc4_init_bias
        + get_fractional(rssm.reward.state_dict(), "fc4.bias", 0.2)
    )

    rssm.transition.load_state_dict(
        torch.load(os.path.join(log_dir, "rssm.pth"))
    )
    rssm.observation.load_state_dict(
        torch.load(os.path.join(log_dir, "obs_model.pth"))
    )

    action_model = action_model.to(device)
    value_model = value_model.to(device)
    rssm.reward = rssm.reward.to(device)

elif transfer_type == "full_transfer":
    encoder.load_state_dict(torch.load(os.path.join(log_dir, "encoder.pth")))

    action_model.load_state_dict(
        torch.load(os.path.join(log_dir, "action_model.pth"))
    )
    value_model.load_state_dict(
        torch.load(os.path.join(log_dir, "value_model.pth"))
    )

    rssm.reward.load_state_dict(
        torch.load(os.path.join(log_dir, "reward_model.pth"))
    )
    rssm.transition.load_state_dict(
        torch.load(os.path.join(log_dir, "rssm.pth"))
    )
    rssm.observation.load_state_dict(
        torch.load(os.path.join(log_dir, "obs_model.pth"))
    )


model_params = (
    list(encoder.parameters())
    + list(rssm.transition.parameters())
    + list(rssm.observation.parameters())
    + list(rssm.reward.parameters())
)
model_optimizer = torch.optim.Adam(model_params, lr=model_lr, eps=eps)
value_optimizer = torch.optim.Adam(
    value_model.parameters(), lr=value_lr, eps=eps
)
action_optimizer = torch.optim.Adam(
    action_model.parameters(), lr=action_lr, eps=eps
)

replay_buffer = ReplayBuffer(
    capacity=CONFIG_DICT["buffer"]["buffer_capacity"],
    observation_shape=[64, 64, 3],
    action_dim=action_space,
)


def main():

    env = make_env(env_name)
    for episode in range(CONFIG_DICT["experiment"]["train"]["seed_episodes"]):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)

            if action.shape[0] < action_space:
                padded_action = np.pad(
                    action, ((0, action_space - action.shape[0]))
                )
                replay_buffer.push(obs, padded_action, reward, done)

            else:
                replay_buffer.push(obs, action, reward, done)
            obs = next_obs
    del env
    gc.collect()

    test_rewards = []
    episodes = []

    for episode in range(seed_episodes, all_episodes):
        # -----------------------------
        #      経験を集める
        # -----------------------------
        start = time.time()
        # 行動を決定するためのエージェントを宣言
        policy = Agent(encoder, rssm.transition, action_model)

        env = make_env(env_name)
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = policy(obs)
            action = action[: env.action_space.shape[0]]

            # 探索のためにガウス分布に従うノイズを加える(explaration noise)
            action += np.random.normal(
                0, np.sqrt(action_noise_var), env.action_space.shape[0]
            )
            next_obs, reward, done, _ = env.step(action)

            if action.shape[0] < action_space:
                action = np.pad(action, ((0, action_space - action.shape[0])))

            # リプレイバッファに観測, 行動, 報酬, doneを格納
            replay_buffer.push(obs, action, reward, done)

            obs = next_obs
            total_reward += reward
            # 訓練時の報酬と経過時間をログとして表示
        print(
            "env : {} episode [{}/{}] is collected. Total reward is {}".format(
                env_name, episode + 1, all_episodes, total_reward
            )
        )
        print("elasped time for interaction: %.2fs" % (time.time() - start))
        del env
        gc.collect()

        # NNのパラメータを更新する
        start = time.time()
        for update_step in range(collect_interval):
            # -------------------------------------------------------------------------------------
            #  RSSM(trainsition_model, obs_model, reward_model)の更新 - Dynamics learning
            # -------------------------------------------------------------------------------------
            observations, actions, rewards, _ = replay_buffer.sample(
                batch_size, chunk_length
            )

            # 観測を前処理し, RNNを用いたPyTorchでの学習のためにTensorの次元を調整
            observations = preprocess_obs(observations)
            observations = torch.as_tensor(observations, device=device)
            observations = observations.transpose(3, 4).transpose(2, 3)
            observations = observations.transpose(0, 1)
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)

            # 観測をエンコーダで低次元のベクトルに変換
            embedded_observations = encoder(
                observations.reshape(-1, 3, 64, 64)
            ).view(chunk_length, batch_size, -1)

            # 低次元の状態表現を保持しておくためのTensorを定義
            states = torch.zeros(
                chunk_length, batch_size, state_dim, device=device
            )
            rnn_hiddens = torch.zeros(
                chunk_length, batch_size, rnn_hidden_dim, device=device
            )

            # 低次元の状態表現は最初はゼロ初期化（timestep１つ分）
            state = torch.zeros(batch_size, state_dim, device=device)
            rnn_hidden = torch.zeros(batch_size, rnn_hidden_dim, device=device)

            # 状態s_tの予測を行ってそのロスを計算する（priorとposteriorの間のKLダイバージェンス）
            kl_loss = 0
            for l in range(chunk_length - 1):
                (
                    next_state_prior,
                    next_state_posterior,
                    rnn_hidden,
                ) = rssm.transition(
                    state, actions[l], rnn_hidden, embedded_observations[l + 1]
                )
                state = next_state_posterior.rsample()
                states[l + 1] = state
                rnn_hiddens[l + 1] = rnn_hidden
                kl = kl_divergence(next_state_prior, next_state_posterior).sum(
                    dim=1
                )
                kl_loss += kl.clamp(
                    min=free_nats
                ).mean()  # 原論文通り, KL誤差がfree_nats以下の時は無視
            kl_loss /= chunk_length - 1

            # states[0] and rnn_hiddens[0]はゼロ初期化なので以降では使わない
            # states, rnn_hiddensは低次元の状態表現
            states = states[1:]
            rnn_hiddens = rnn_hiddens[1:]

            # 観測を再構成, また, 報酬を予測
            flatten_states = states.view(-1, state_dim)
            flatten_rnn_hiddens = rnn_hiddens.view(-1, rnn_hidden_dim)
            recon_observations = rssm.observation(
                flatten_states, flatten_rnn_hiddens
            ).view(chunk_length - 1, batch_size, 3, 64, 64)
            predicted_rewards = rssm.reward(
                flatten_states, flatten_rnn_hiddens
            ).view(chunk_length - 1, batch_size, 1)

            # 観測と報酬の予測誤差を計算
            obs_loss = (
                0.5
                * F.mse_loss(
                    recon_observations, observations[1:], reduction="none"
                )
                .mean([0, 1])
                .sum()
            )
            reward_loss = 0.5 * F.mse_loss(predicted_rewards, rewards[:-1])

            # 以上のロスを合わせて勾配降下で更新する
            model_loss = kl_loss + obs_loss + reward_loss
            model_optimizer.zero_grad()
            model_loss.backward()
            clip_grad_norm_(model_params, clip_grad_norm)
            model_optimizer.step()

            # --------------------------------------------------
            #  Action Model, Value　Modelの更新　- Behavior leaning
            # --------------------------------------------------
            # Actor-Criticのロスで他のモデルを更新することはないので勾配の流れを一度遮断
            # flatten_states, flatten_rnn_hiddensは RSSMから得られた低次元の状態表現を平坦化した値
            flatten_states = flatten_states.detach()
            flatten_rnn_hiddens = flatten_rnn_hiddens.detach()

            # DreamerにおけるActor-Criticの更新のために, 現在のモデルを用いた
            # 数ステップ先の未来の状態予測を保持するためのTensorを用意
            imaginated_states = torch.zeros(
                imagination_horizon + 1,
                *flatten_states.shape,
                device=flatten_states.device
            )
            imaginated_rnn_hiddens = torch.zeros(
                imagination_horizon + 1,
                *flatten_rnn_hiddens.shape,
                device=flatten_rnn_hiddens.device
            )

            # 　未来予測をして想像上の軌道を作る前に, 最初の状態としては先ほどモデルの更新で使っていた
            # リプレイバッファからサンプルされた観測データを取り込んだ上で推論した状態表現を使う
            imaginated_states[0] = flatten_states
            imaginated_rnn_hiddens[0] = flatten_rnn_hiddens

            # open-loopで未来の状態予測を使い, 想像上の軌道を作る
            for h in range(1, imagination_horizon + 1):
                # 行動はActionModelで決定. この行動はモデルのパラメータに対して微分可能で,
                # 　これを介してActionModelは更新される
                actions = action_model(flatten_states, flatten_rnn_hiddens)
                (
                    flatten_states_prior,
                    flatten_rnn_hiddens,
                ) = rssm.transition.prior(
                    rssm.transition.reccurent(
                        flatten_states, actions, flatten_rnn_hiddens
                    )
                )
                flatten_states = flatten_states_prior.rsample()
                imaginated_states[h] = flatten_states
                imaginated_rnn_hiddens[h] = flatten_rnn_hiddens

            # RSSMのreward_modelにより予測された架空の軌道に対する報酬を計算
            flatten_imaginated_states = imaginated_states.view(-1, state_dim)
            flatten_imaginated_rnn_hiddens = imaginated_rnn_hiddens.view(
                -1, rnn_hidden_dim
            )
            imaginated_rewards = rssm.reward(
                flatten_imaginated_states, flatten_imaginated_rnn_hiddens
            ).view(imagination_horizon + 1, -1)
            imaginated_values = value_model(
                flatten_imaginated_states, flatten_imaginated_rnn_hiddens
            ).view(imagination_horizon + 1, -1)

            # λ-returnのターゲットを計算(V_{\lambda}(s_{\tau})
            lambda_target_values = lambda_target(
                imaginated_rewards, imaginated_values, gamma, lambda_
            )

            # 価値関数の予測した価値が大きくなるようにActionModelを更新
            # PyTorchの基本は勾配降下だが, 今回は大きくしたいので-1をかける
            action_loss = -lambda_target_values.mean()
            action_optimizer.zero_grad()
            action_loss.backward()
            clip_grad_norm_(action_model.parameters(), clip_grad_norm)
            action_optimizer.step()

            # TD(λ)ベースの目的関数で価値関数を更新（価値関数のみを学習するため，学習しない変数のグラフは切っている. )
            imaginated_values = value_model(
                flatten_imaginated_states.detach(),
                flatten_imaginated_rnn_hiddens.detach(),
            ).view(imagination_horizon + 1, -1)
            value_loss = 0.5 * F.mse_loss(
                imaginated_values, lambda_target_values.detach()
            )
            value_optimizer.zero_grad()
            value_loss.backward()
            clip_grad_norm_(value_model.parameters(), clip_grad_norm)
            value_optimizer.step()

            # ログをTensorBoardに出力
            print(
                "update_step: %3d model loss: %.5f, kl_loss: %.5f, "
                "obs_loss: %.5f, reward_loss: %.5f, "
                "value_loss: %.5f action_loss: %.5f"
                % (
                    update_step + 1,
                    model_loss.item(),
                    kl_loss.item(),
                    obs_loss.item(),
                    reward_loss.item(),
                    value_loss.item(),
                    action_loss.item(),
                )
            )

        print("elasped time for update: %.2fs" % (time.time() - start))
        # --------------------------------------------------------------
        #    テストフェーズ. 探索ノイズなしでの性能を評価する
        # --------------------------------------------------------------
        if (episode + 1) % test_interval == 0:
            policy = Agent(encoder, rssm.transition, action_model)
            start = time.time()
            env = make_env(env_name)
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = policy(obs, training=False)
                action = action[: env.action_space.shape[0]]
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            print(
                "Total test reward at episode [%4d/%4d] is %f"
                % (episode + 1, all_episodes, total_reward)
            )
            print("elasped time for test: %.2fs" % (time.time() - start))
            test_rewards.append(total_reward)
            episodes.append(episode)

            df = pd.DataFrame(
                list(zip(episodes, test_rewards)),
                columns=["episodes", "test_rewards"],
            )
            df.to_csv(os.path.join(log_dir, "test_reward.csv"))

            del env
            gc.collect()

        if (episode + 1) % model_save_interval == 0:
            # 定期的に学習済みモデルのパラメータを保存する
            model_log_dir = os.path.join(
                log_dir, "episode_%04d" % (episode + 1)
            )
            os.makedirs(model_log_dir)
            torch.save(
                encoder.state_dict(),
                os.path.join(model_log_dir, "encoder.pth"),
            )
            torch.save(
                rssm.transition.state_dict(),
                os.path.join(model_log_dir, "rssm.pth"),
            )
            torch.save(
                rssm.observation.state_dict(),
                os.path.join(model_log_dir, "obs_model.pth"),
            )
            torch.save(
                rssm.reward.state_dict(),
                os.path.join(model_log_dir, "reward_model.pth"),
            )
            torch.save(
                value_model.state_dict(),
                os.path.join(model_log_dir, "value_model.pth"),
            )
            torch.save(
                action_model.state_dict(),
                os.path.join(model_log_dir, "action_model.pth"),
            )


if __name__ == "__main__":
    folder_name = (
        "transfered_"
        + CONFIG_DICT["experiment"]["transfer_type"]
        + "_"
        + CONFIG_DICT["experiment"]["env_name"]
        + "_"
        + get_str_currentdate()
    )
    log_dir = os.path.join(CONFIG_DICT["logs"]["log_dir"], folder_name)
    os.makedirs(log_dir, exist_ok=True)
    shutil_copy("./configs/transfer_single_env_config.py", log_dir)
    main()

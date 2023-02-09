import os
import sys

sys.path.append("./")

import gc
import time

import numpy as np
import pandas as pd
import pybullet_envs
import torch
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from configs import SINGLE_SHOT_SINGLE_ENV_CONFIG_DICT as CONFIG_DICT
from src.agent import Agent
from src.buffer import ReplayBuffer
from src.loss import lambda_target
from src.model import RSSM, ActionModel, Encoder, ValueModel
from src.prune import apply_masks, calculate_grad, get_masks, get_score
from src.slth.utils import modify_module_for_slth
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

_env = make_env(CONFIG_DICT["experiment"]["env_name"])
encoder_init = Encoder().to(device)
rssm_init = RSSM(state_dim, _env.action_space.shape[0], rnn_hidden_dim, device)
value_model_init = ValueModel(state_dim, rnn_hidden_dim).to(device)
action_model_init = ActionModel(
    state_dim, rnn_hidden_dim, _env.action_space.shape[0]
).to(device)


replay_buffer = ReplayBuffer(
    capacity=CONFIG_DICT["buffer"]["buffer_capacity"],
    observation_shape=_env.observation_space.shape,
    action_dim=_env.action_space.shape[0],
)


def main():

    models = (encoder_init, rssm_init, value_model_init, action_model_init)

    env = make_env(CONFIG_DICT["experiment"]["env_name"])
    for episode in range(CONFIG_DICT["experiment"]["train"]["seed_episodes"]):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs
    del env
    gc.collect()

    observations, actions, rewards, _ = replay_buffer.sample(
        batch_size, chunk_length
    )

    encoder, rssm, value_model, action_model = calculate_grad(
        (observations, actions, rewards),
        models,
        CONFIG_DICT["single_shot"]["method"],
        device,
        (
            imagination_horizon,
            gamma,
            lambda_,
            clip_grad_norm,
            free_nats,
            chunk_length,
            batch_size,
            state_dim,
            rnn_hidden_dim,
        ),
    )

    encoder_scores = get_score(encoder)
    rssm_transition_scores = get_score(rssm.transition)  # observation, reward
    rssm_observation_scores = get_score(rssm.observation)
    rssm_reward_scores = get_score(rssm.reward)
    value_model_scores = get_score(value_model)
    action_model_scores = get_score(action_model)

    keep_ratio = CONFIG_DICT["single_shot"]["keep_ratio"]
    encoder_masks = get_masks(encoder_scores, keep_ratio)
    rssm_transition_masks = get_masks(rssm_transition_scores, keep_ratio)
    rssm_observation_masks = get_masks(rssm_observation_scores, keep_ratio)
    rssm_reward_masks = get_masks(rssm_reward_scores, keep_ratio)
    value_model_masks = get_masks(value_model_scores, keep_ratio)
    action_model_masks = get_masks(action_model_scores, keep_ratio)

    apply_masks(encoder, encoder_masks)
    apply_masks(rssm.transition, rssm_transition_masks)
    apply_masks(rssm.observation, rssm_observation_masks)
    apply_masks(rssm.reward, rssm_reward_masks)
    apply_masks(value_model, value_model_masks)
    apply_masks(action_model, action_model_masks)

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

    torch.save(
        {
            "encoder_scores": encoder_scores,
            "rssm_transition_scores": rssm_transition_scores,
            "rssm_observation_scores": rssm_observation_scores,
            "rssm_reward_scores": rssm_reward_scores,
            "value_model_scores": value_model_scores,
            "action_model_scores": action_model_scores,
        },
        os.path.join(log_dir, "scores.pkl"),
    )

    test_rewards = []
    episodes = []

    for episode in range(seed_episodes, all_episodes):
        # -----------------------------
        #      経験を集める
        # -----------------------------
        start = time.time()
        # 行動を決定するためのエージェントを宣言
        policy = Agent(encoder, rssm.transition, action_model)

        env = make_env(CONFIG_DICT["experiment"]["env_name"])
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = policy(obs)
            # 探索のためにガウス分布に従うノイズを加える(explaration noise)
            action += np.random.normal(
                0, np.sqrt(action_noise_var), env.action_space.shape[0]
            )
            next_obs, reward, done, _ = env.step(action)

            # リプレイバッファに観測, 行動, 報酬, doneを格納
            replay_buffer.push(obs, action, reward, done)

            obs = next_obs
            total_reward += reward

        # 訓練時の報酬と経過時間をログとして表示
        print(
            "episode [%4d/%4d] is collected. Total reward is %f"
            % (episode + 1, all_episodes, total_reward)
        )
        print("elasped time for interaction: %.2fs" % (time.time() - start))

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

        del env
        gc.collect()

        # --------------------------------------------------------------
        #    テストフェーズ. 探索ノイズなしでの性能を評価する
        # --------------------------------------------------------------
        if (episode + 1) % test_interval == 0:
            env = make_env(CONFIG_DICT["experiment"]["env_name"])
            policy = Agent(encoder, rssm.transition, action_model)
            start = time.time()
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = policy(obs, training=False)
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
    log_dir = os.path.join(
        CONFIG_DICT["logs"]["log_dir"],
        "SingleShot_"
        + CONFIG_DICT["experiment"]["env_name"]
        + "/"
        + CONFIG_DICT["single_shot"]["method"]
        + "/"
        + get_str_currentdate(),
    )
    os.makedirs(log_dir, exist_ok=True)
    shutil_copy(
        "./configs/single_shot/single_shot_single_env_config.py", log_dir
    )
    main()

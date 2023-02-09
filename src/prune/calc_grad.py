import copy

import torch
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from src.loss import lambda_target
from src.utils import preprocess_obs


def calculate_grad(
    inputs: tuple, models: tuple, prune_type: str, device: str, params: tuple
) -> tuple:

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
    ) = params

    observations, actions, rewards = inputs
    if prune_type == "SNIP":
        pass
    elif prune_type == "synflow":
        observations = observations * 0 + 1
        actions = actions * 0 + 1
        rewards = rewards * 0 + 1

    encoder, rssm, value_model, action_model = models
    encoder = copy.deepcopy(encoder)
    rssm = copy.deepcopy(rssm)
    value_model = copy.deepcopy(value_model)
    action_model = copy.deepcopy(action_model)

    model_params = (
        list(encoder.parameters())
        + list(rssm.transition.parameters())
        + list(rssm.observation.parameters())
        + list(rssm.reward.parameters())
    )

    # 観測を前処理し, RNNを用いたPyTorchでの学習のためにTensorの次元を調整
    observations = preprocess_obs(observations)
    observations = torch.as_tensor(observations, device=device)
    observations = observations.transpose(3, 4).transpose(2, 3)
    observations = observations.transpose(0, 1)
    actions = torch.as_tensor(actions, device=device).transpose(0, 1)
    rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)

    # 観測をエンコーダで低次元のベクトルに変換
    embedded_observations = encoder(observations.reshape(-1, 3, 64, 64)).view(
        chunk_length, batch_size, -1
    )

    # 低次元の状態表現を保持しておくためのTensorを定義
    states = torch.zeros(chunk_length, batch_size, state_dim, device=device)
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
        kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
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
    predicted_rewards = rssm.reward(flatten_states, flatten_rnn_hiddens).view(
        chunk_length - 1, batch_size, 1
    )

    # 観測と報酬の予測誤差を計算
    obs_loss = (
        0.5
        * F.mse_loss(recon_observations, observations[1:], reduction="none")
        .mean([0, 1])
        .sum()
    )
    reward_loss = 0.5 * F.mse_loss(predicted_rewards, rewards[:-1])

    # 以上のロスを合わせて勾配降下で更新する
    model_loss = kl_loss + obs_loss + reward_loss
    model_loss.backward()
    clip_grad_norm_(model_params, clip_grad_norm)

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
        (flatten_states_prior, flatten_rnn_hiddens,) = rssm.transition.prior(
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
    action_loss.backward()
    clip_grad_norm_(action_model.parameters(), clip_grad_norm)

    # TD(λ)ベースの目的関数で価値関数を更新（価値関数のみを学習するため，学習しない変数のグラフは切っている. )
    imaginated_values = value_model(
        flatten_imaginated_states.detach(),
        flatten_imaginated_rnn_hiddens.detach(),
    ).view(imagination_horizon + 1, -1)
    value_loss = 0.5 * F.mse_loss(
        imaginated_values, lambda_target_values.detach()
    )

    value_loss.backward()
    clip_grad_norm_(value_model.parameters(), clip_grad_norm)

    return (encoder, rssm, value_model, action_model)

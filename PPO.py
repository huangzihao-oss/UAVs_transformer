import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


print("============================================================================================")
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class RolloutBuffer:
    def __init__(self):
        self.encoder_tokens = []
        self.encoder_pad = []
        self.encoder_segment_ids = []
        self.decoder_tokens = []
        self.decoder_pad = []
        self.critic_tokens = []
        self.critic_pad = []
        self.target_masks = []

        self.target_actions = []
        self.speed_actions = []
        self.logprobs = []

        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        self.encoder_tokens.clear()
        self.encoder_pad.clear()
        self.encoder_segment_ids.clear()
        self.decoder_tokens.clear()
        self.decoder_pad.clear()
        self.critic_tokens.clear()
        self.critic_pad.clear()
        self.target_masks.clear()

        self.target_actions.clear()
        self.speed_actions.clear()
        self.logprobs.clear()

        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


class TransformerActorCritic(nn.Module):
    def __init__(
        self,
        token_dim,
        target_action_dim,
        speed_action_dim,
        max_encoder_len,
        max_decoder_len,
        max_critic_len,
        max_other_agents=0,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.target_action_dim = target_action_dim
        self.speed_action_dim = speed_action_dim
        self.max_other_agents = max(0, int(max_other_agents))

        self.token_proj = nn.Linear(token_dim, d_model)

        self.encoder_pos = nn.Parameter(torch.zeros(max_encoder_len, d_model))
        self.decoder_pos = nn.Parameter(torch.zeros(max_decoder_len, d_model))
        self.critic_pos = nn.Parameter(torch.zeros(max_critic_len, d_model))
        self.encoder_segment_embedding = nn.Embedding(self.max_other_agents + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.actor_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.actor_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.critic_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.target_head = nn.Linear(d_model, target_action_dim)
        self.speed_head = nn.Linear(d_model, speed_action_dim)
        self.critic_head = nn.Linear(d_model, 1)

    def _add_positional(self, x, pos_table):
        return x + pos_table[: x.size(1)].unsqueeze(0)

    def _last_valid(self, h, pad_mask):
        if pad_mask is None:
            return h[:, -1, :]
        valid_len = (~pad_mask).long().sum(dim=1) - 1
        valid_len = torch.clamp(valid_len, min=0)
        batch_idx = torch.arange(h.size(0), device=h.device)
        return h[batch_idx, valid_len, :]

    def actor_forward(self, encoder_tokens, decoder_tokens, encoder_pad=None, decoder_pad=None, encoder_segment_ids=None):
        src = self._add_positional(self.token_proj(encoder_tokens), self.encoder_pos)
        if encoder_segment_ids is not None:
            encoder_segment_ids = torch.clamp(encoder_segment_ids.long(), min=0, max=self.max_other_agents)
            src = src + self.encoder_segment_embedding(encoder_segment_ids)
        tgt = self._add_positional(self.token_proj(decoder_tokens), self.decoder_pos)

        memory = self.actor_encoder(src, src_key_padding_mask=encoder_pad)
        dec_h = self.actor_decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=decoder_pad,
            memory_key_padding_mask=encoder_pad,
        )
        actor_h = self._last_valid(dec_h, decoder_pad)

        target_logits = self.target_head(actor_h)
        speed_logits = self.speed_head(actor_h)
        return target_logits, speed_logits

    def critic_forward(self, critic_tokens, critic_pad=None):
        critic_x = self._add_positional(self.token_proj(critic_tokens), self.critic_pos)
        critic_h = self.critic_encoder(critic_x, src_key_padding_mask=critic_pad)

        if critic_pad is None:
            pooled = critic_h.mean(dim=1)
        else:
            valid = (~critic_pad).float().unsqueeze(-1)
            denom = torch.clamp(valid.sum(dim=1), min=1.0)
            pooled = (critic_h * valid).sum(dim=1) / denom
        return self.critic_head(pooled)

    def _masked_target_logits(self, target_logits, target_mask):
        target_mask = torch.as_tensor(target_mask, dtype=torch.float32, device=target_logits.device)
        if target_mask.dim() == 1:
            target_mask = target_mask.unsqueeze(0)
        if torch.any(target_mask.sum(dim=1) <= 0):
            raise ValueError("Target mask cannot be all zeros.")
        return target_logits.masked_fill(target_mask <= 0, -1e9)

    def act(self, inputs, masks, deter_action=None):
        target_logits, speed_logits = self.actor_forward(
            inputs["encoder_tokens"],
            inputs["decoder_tokens"],
            inputs.get("encoder_pad"),
            inputs.get("decoder_pad"),
            inputs.get("encoder_segment_ids"),
        )
        target_logits = self._masked_target_logits(target_logits, masks["target"])

        dist_target = Categorical(logits=target_logits)
        dist_speed = Categorical(logits=speed_logits)

        if deter_action is not None:
            target_action = torch.tensor([int(deter_action[0])], dtype=torch.long, device=target_logits.device)
            speed_action = torch.tensor([int(deter_action[1])], dtype=torch.long, device=target_logits.device)
        else:
            target_action = dist_target.sample()
            speed_action = dist_speed.sample()

        logprob = dist_target.log_prob(target_action) + dist_speed.log_prob(speed_action)
        state_val = self.critic_forward(inputs["critic_tokens"], inputs.get("critic_pad"))

        return target_action.detach(), speed_action.detach(), logprob.detach(), state_val.detach()

    def act_test(self, inputs, masks):
        with torch.no_grad():
            target_logits, speed_logits = self.actor_forward(
                inputs["encoder_tokens"],
                inputs["decoder_tokens"],
                inputs.get("encoder_pad"),
                inputs.get("decoder_pad"),
                inputs.get("encoder_segment_ids"),
            )
            target_logits = self._masked_target_logits(target_logits, masks["target"])

            target_action = torch.argmax(target_logits, dim=-1)
            speed_action = torch.argmax(speed_logits, dim=-1)

        return int(target_action.item()), int(speed_action.item())

    def evaluate(self, inputs, target_actions, speed_actions, target_mask):
        target_logits, speed_logits = self.actor_forward(
            inputs["encoder_tokens"],
            inputs["decoder_tokens"],
            inputs.get("encoder_pad"),
            inputs.get("decoder_pad"),
            inputs.get("encoder_segment_ids"),
        )
        target_logits = self._masked_target_logits(target_logits, target_mask)

        dist_target = Categorical(logits=target_logits)
        dist_speed = Categorical(logits=speed_logits)

        action_logprobs = dist_target.log_prob(target_actions) + dist_speed.log_prob(speed_actions)
        dist_entropy = dist_target.entropy() + dist_speed.entropy()

        state_values = self.critic_forward(inputs["critic_tokens"], inputs.get("critic_pad"))
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        agent_id,
        token_dim,
        target_action_dim,
        speed_action_dim,
        max_encoder_len,
        max_decoder_len,
        max_critic_len,
        max_other_agents,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        summary_dir,
        entropy_ratio,
        gae_lambda,
        gae_flag,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        self.id = agent_id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_ratio = entropy_ratio
        self.gae_lambda = gae_lambda
        self.gae_flag = gae_flag

        self.buffer = RolloutBuffer()

        self.policy = TransformerActorCritic(
            token_dim=token_dim,
            target_action_dim=target_action_dim,
            speed_action_dim=speed_action_dim,
            max_encoder_len=max_encoder_len,
            max_decoder_len=max_decoder_len,
            max_critic_len=max_critic_len,
            max_other_agents=max_other_agents,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        actor_params = []
        actor_params += list(self.policy.token_proj.parameters())
        actor_params += list(self.policy.encoder_segment_embedding.parameters())
        actor_params += list(self.policy.actor_encoder.parameters())
        actor_params += list(self.policy.actor_decoder.parameters())
        actor_params += list(self.policy.target_head.parameters())
        actor_params += list(self.policy.speed_head.parameters())

        critic_params = []
        critic_params += list(self.policy.critic_encoder.parameters())
        critic_params += list(self.policy.critic_head.parameters())

        self.optimizer = torch.optim.Adam(
            [
                {"params": actor_params, "lr": lr_actor},
                {"params": critic_params, "lr": lr_critic},
            ]
        )

        self.policy_old = TransformerActorCritic(
            token_dim=token_dim,
            target_action_dim=target_action_dim,
            speed_action_dim=speed_action_dim,
            max_encoder_len=max_encoder_len,
            max_decoder_len=max_decoder_len,
            max_critic_len=max_critic_len,
            max_other_agents=max_other_agents,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.update_times = 0

        self.summary_dir = summary_dir
        self.writer = SummaryWriter(log_dir=self.summary_dir)

    def _to_policy_inputs(self, obs_pack):
        encoder_tokens = torch.as_tensor(obs_pack["encoder_tokens"], dtype=torch.float32, device=device).unsqueeze(0)
        encoder_segment_ids = obs_pack.get("encoder_segment_ids", None)
        if encoder_segment_ids is None:
            encoder_segment_ids = torch.zeros(
                encoder_tokens.size(1), dtype=torch.long, device=device
            ).unsqueeze(0)
        else:
            encoder_segment_ids = torch.as_tensor(
                encoder_segment_ids, dtype=torch.long, device=device
            ).unsqueeze(0)
        return {
            "encoder_tokens": encoder_tokens,
            "encoder_pad": torch.as_tensor(obs_pack["encoder_pad"], dtype=torch.bool, device=device).unsqueeze(0),
            "encoder_segment_ids": encoder_segment_ids,
            "decoder_tokens": torch.as_tensor(obs_pack["decoder_tokens"], dtype=torch.float32, device=device).unsqueeze(0),
            "decoder_pad": torch.as_tensor(obs_pack["decoder_pad"], dtype=torch.bool, device=device).unsqueeze(0),
            "critic_tokens": torch.as_tensor(obs_pack["critic_tokens"], dtype=torch.float32, device=device).unsqueeze(0),
            "critic_pad": torch.as_tensor(obs_pack["critic_pad"], dtype=torch.bool, device=device).unsqueeze(0),
        }

    def select_action(self, obs_pack, masks, deter_action=None):
        inputs = self._to_policy_inputs(obs_pack)
        target_mask = torch.as_tensor(masks["target"], dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            target_action, speed_action, action_logprob, state_val = self.policy_old.act(
                inputs,
                {"target": target_mask},
                deter_action=deter_action,
            )

        self.buffer.encoder_tokens.append(inputs["encoder_tokens"].squeeze(0).detach())
        self.buffer.encoder_pad.append(inputs["encoder_pad"].squeeze(0).detach())
        self.buffer.encoder_segment_ids.append(inputs["encoder_segment_ids"].squeeze(0).detach())
        self.buffer.decoder_tokens.append(inputs["decoder_tokens"].squeeze(0).detach())
        self.buffer.decoder_pad.append(inputs["decoder_pad"].squeeze(0).detach())
        self.buffer.critic_tokens.append(inputs["critic_tokens"].squeeze(0).detach())
        self.buffer.critic_pad.append(inputs["critic_pad"].squeeze(0).detach())
        self.buffer.target_masks.append(target_mask.squeeze(0).detach())

        self.buffer.target_actions.append(target_action.squeeze(0).detach())
        self.buffer.speed_actions.append(speed_action.squeeze(0).detach())
        self.buffer.logprobs.append(action_logprob.squeeze(0).detach())
        self.buffer.state_values.append(state_val.squeeze(0).detach())

        return int(target_action.item()), int(speed_action.item())

    def action_test(self, obs_pack, masks):
        inputs = self._to_policy_inputs(obs_pack)
        target_mask = torch.as_tensor(masks["target"], dtype=torch.float32, device=device).unsqueeze(0)
        return self.policy_old.act_test(inputs, {"target": target_mask})

    def update(self):
        if len(self.buffer.rewards) == 0:
            return

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        old_inputs = {
            "encoder_tokens": torch.stack(self.buffer.encoder_tokens, dim=0).detach().to(device),
            "encoder_pad": torch.stack(self.buffer.encoder_pad, dim=0).detach().to(device),
            "encoder_segment_ids": torch.stack(self.buffer.encoder_segment_ids, dim=0).detach().to(device),
            "decoder_tokens": torch.stack(self.buffer.decoder_tokens, dim=0).detach().to(device),
            "decoder_pad": torch.stack(self.buffer.decoder_pad, dim=0).detach().to(device),
            "critic_tokens": torch.stack(self.buffer.critic_tokens, dim=0).detach().to(device),
            "critic_pad": torch.stack(self.buffer.critic_pad, dim=0).detach().to(device),
        }

        old_target_masks = torch.stack(self.buffer.target_masks, dim=0).detach().to(device)
        old_target_actions = torch.stack(self.buffer.target_actions, dim=0).detach().to(device)
        old_speed_actions = torch.stack(self.buffer.speed_actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach().to(device).view(-1)

        if self.gae_flag:
            advantages = self.compute_gae(self.buffer.rewards, old_state_values, self.buffer.is_terminals)
        else:
            advantages = rewards - old_state_values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_inputs,
                old_target_actions,
                old_speed_actions,
                old_target_masks,
            )
            state_values = state_values.view(-1)

            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, rewards)
            entropy_loss = dist_entropy.mean()
            loss = policy_loss + 0.5 * critic_loss - self.entropy_ratio * entropy_loss

            if self.id == 0:
                self.writer.add_scalar("loss/policy", policy_loss.detach().item(), self.update_times)
                self.writer.add_scalar("loss/critic", critic_loss.detach().item(), self.update_times)
                self.writer.add_scalar("stats/critic", state_values.mean().detach().item(), self.update_times)
                self.writer.add_scalar("stats/entropy", entropy_loss.detach().item(), self.update_times)
                self.writer.add_scalar("reward/train", float(sum(self.buffer.rewards)), self.update_times)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.update_times += 1

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def compute_gae(self, rewards, state_values, is_terminals):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(is_terminals, dtype=torch.float32, device=device)
        values = state_values.detach()

        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, dtype=torch.float32, device=device)
        next_value = torch.tensor(0.0, dtype=torch.float32, device=device)

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]

        return advantages

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def call_2_record(self, steps, value):
        self.writer.add_scalar("reward/test", value, steps)

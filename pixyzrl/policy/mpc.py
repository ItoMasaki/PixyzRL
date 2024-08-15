import torch


# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner:
    __constants__ = ["action_size", "planning_horizon", "optimisation_iters", "candidates", "top_candidates"]

    def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, stoch_model, reward_model):
        super().__init__()
        self.transition_model, self.stoch_model, self.reward_model = transition_model, stoch_model, reward_model
        self.action_size = action_size
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates

    def forward(self, belief, state):
        B, H, Z = belief.size(0), belief.size(1), state.size(1)
        belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean, action_std_dev = torch.zeros(self.planning_horizon, self.candidates, self.action_size, device=belief.device), torch.ones(self.planning_horizon, self.candidates, self.action_size, device=belief.device)

        for _ in range(self.optimisation_iters):
            beliefs = []
            states = []
            actions = action_mean + action_std_dev * torch.randn(self.planning_horizon, self.candidates, self.action_size, device=belief.device)
            actions[:, :, 1] = torch.clamp(actions[:, :, 1], 0, 1)
            actions[:, :, 2] = torch.clamp(actions[:, :, 2], 0, 1)
            for step in range(self.planning_horizon):
                belief = self.transition_model(state, actions[step], belief, torch.ones((1, 1), device=belief.device))["z_tp1"]
                state = self.stoch_model.sample({"z_t": belief})["s_t"]

                beliefs.append(belief)
                states.append(state)

            beliefs, states = torch.stack(beliefs, dim=0), torch.stack(states, dim=0)

            returns = self.reward_model.sample_mean({"s_t": states, "z_t": beliefs}).sum(dim=0)

            # # Re-fit belief to the K best action sequences
            _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
            # # Fix indices for unrolled actions
            topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)
            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size).squeeze()
            # # Update belief with new means and standard deviations
            action_mean, action_std_dev = best_actions.mean(dim=1, keepdim=True), best_actions.std(dim=1, unbiased=False, keepdim=True)
        # Return first action mean µ_t
        return action_mean[0].squeeze()

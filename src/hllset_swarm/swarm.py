# src/hllset_swarm/swarm.py
class SwarmState:
    """GPU resident vector + sparse matrices"""
    def __init__(self, n: int, alpha: float, beta: float, gamma: float):
        self.s = torch.full((n,), 0.5, device="cuda")
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def step(self, teacher: torch.Tensor | None = None):
        cognitive = torch.sparse.mm(Wτ, self.s.unsqueeze(1)).squeeze()
        exclusion = torch.sparse.mm(Wρ, self.s.unsqueeze(1)).squeeze()
        self.s += self.alpha * cognitive - self.beta * exclusion
        if teacher is not None:
            social = torch.sparse.mm(Wτ, teacher.unsqueeze(1)).squeeze()
            self.s += self.gamma * social
        self.s = torch.clamp(self.s, 0.0, 1.0)

# src/hllset_swarm/trajectory.py
class TrajectoryCompiler:
    """Compile YAML list → GPU kernel sequence"""
    def compile(self, steps: list[dict]) -> list[Callable]:
        return [self._compile_op(op) for op in steps]

    def _compile_op(self, op: dict) -> Callable:
        if op["op"] == "cover":
            return lambda env: env.cover_mask()
        if op["op"] == "converge":
            return lambda env: self._converge(env, **op)
        if op["op"] == "feedback":
            return lambda env: env.write_back(self.swarm.s)
        # ... more ops
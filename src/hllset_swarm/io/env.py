class Environment:
    def __init__(self, text: str, adapter: Adapter):
        self.text = text
        self.adapter = adapter          # LLM, DB, robot, etc.

    def cover_mask(self) -> torch.Tensor:
        chars = {c for c in self.text if c in kernel}
        t = torch.zeros(len(kernel), device="cuda")
        t[[idx[c] for c in chars]] = 1.0
        return t

    def write_back(self, s: torch.Tensor):
        """adapter-specific: REST, SQL, ROS, …"""
        self.adapter.update_embedding(s.cpu().numpy())
def commit_layer(self, repo: str, token: str):
    layer = {
        "swarm_state": self.swarm.s.half(),
        "Wτ_coo": self.Wτ.coalesce(),
        "Wρ_coo": self.Wρ.coalesce(),
        "meta": {"parent": self.parent_sha, "entry": self.env.text}
    }
    buf = io.BytesIO()
    torch.save(layer, buf)
    compressed = zstd.compress(buf.getvalue())
    sha = hashlib.sha256(compressed).hexdigest()[:16]
    github_upload(repo, f"layers/{sha}.pt.zst", compressed, token)
    self.parent_sha = sha
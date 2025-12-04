from __future__ import annotations
import io
import hashlib
import torch
import zstandard as zstd
import requests
from typing import Optional

__all__ = ["github_upload"]


def github_upload(repo: str, path: str, data: bytes, token: str) -> str:
    """
    Upload binary data to a GitHub repository.
    
    Args:
        repo: Repository in format "owner/repo"
        path: File path within the repo
        data: Binary data to upload
        token: GitHub personal access token
        
    Returns:
        SHA hash of the uploaded content
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    import base64
    content_b64 = base64.b64encode(data).decode()
    
    payload = {
        "message": f"Add layer {path}",
        "content": content_b64
    }
    
    response = requests.put(url, json=payload, headers=headers)
    response.raise_for_status()
    
    return response.json()["content"]["sha"]


def commit_layer(self, repo: str, token: str) -> str:
    """
    Commit current swarm state as a layer to GitHub.
    
    Args:
        self: SwarmProgram instance
        repo: Repository in format "owner/repo"
        token: GitHub personal access token
        
    Returns:
        SHA hash of the committed layer
    """
    layer = {
        "swarm_state": self.swarm.s.half(),
        "Wτ_coo": self.lattice.Wτ.coalesce(),
        "Wρ_coo": self.lattice.Wρ.coalesce(),
        "meta": {"parent": self.parent_sha, "entry": self.env.text}
    }
    buf = io.BytesIO()
    torch.save(layer, buf)
    compressed = zstd.compress(buf.getvalue())
    sha = hashlib.sha256(compressed).hexdigest()[:16]
    github_upload(repo, f"layers/{sha}.pt.zst", compressed, token)
    self.parent_sha = sha
    return sha
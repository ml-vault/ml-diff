import requests

data = {"input": {"upload_dir": "/runpod-volume/_tmp/v3/output", "repo_id": "togoron/gulora2", "repo_path": "v2", "write_token": "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"}}

resp = requests.post("https://api.runpod.ai/v2/kccm083lu9qq4d/run", json=data, headers={"Authorization": f"Bearer YH09L6LL5NRZTF294BFBIAUS0W5IC6VXH3Z0JWGH"})
print(resp)


Download the repo in this directory
```bash
# (skip this if you are already in the right directory)
cd datasets/MMTBench/

# download from Hugging Face (you can leave out your token, but then AFAIK your download speed is slower(?))
hf download MMTBench/MMTabReal --repo-type=dataset --local-dir=/tmp/dsa --token=<YOUR_HF_TOKEN> --max-workers <YOUR_NUMBER_OF_CPUS>
```
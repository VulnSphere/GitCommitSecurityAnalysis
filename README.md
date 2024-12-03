# GitCommitSecurityAnalysis
This repo uses a large model to analyze the last n commits of a repository for potential security issues.

### Default. Analyze the last 10 commits.
```
python main.py /path/to/git/repo
```
### Analyze the last 3 commits and save the output in json format.
```
python main.py --commits 3 /path/to/git/repo --output results.json
```

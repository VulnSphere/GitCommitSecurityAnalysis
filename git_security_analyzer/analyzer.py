import os
from typing import List, Dict, Set
from git import Repo, Commit
import logging
from .models import CommitAnalysis, SecurityIssue
from .llm import initialize_llm

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a security expert analyzing Git commits for potential security vulnerabilities.
Focus on identifying security issues in code changes while considering the context.
For each commit:
1. Determine if it's a documentation change, dependency update, or code change
2. For code changes, analyze new/modified code for security vulnerabilities
3. Consider the context and surrounding code when assessing security impact
4. Provide specific recommendations for fixing identified issues

Respond in JSON format matching the CommitAnalysis model."""

class GitSecurityAnalyzer:
    def __init__(self, repo_path: str, llm_type: str = "claude", llm_kwargs: Dict = None):
        self.repo_path = repo_path
        self.repo = Repo(repo_path)
        self.llm = initialize_llm(llm_type, system_prompt=SYSTEM_PROMPT)
        self.analyzed_files: Set[str] = set()
        
    def _is_documentation_file(self, file_path: str) -> bool:
        """Check if the file is documentation."""
        doc_patterns = {'.md', '.rst', '.txt', '.docx', '.pdf', 'docs/', 'documentation/'}
        return any(pattern in file_path.lower() for pattern in doc_patterns)
    
    def _is_dependency_file(self, file_path: str) -> bool:
        """Check if the file is a dependency manifest."""
        dep_patterns = {
            'requirements.txt', 'setup.py', 'package.json', 'pom.xml',
            'build.gradle', 'Gemfile', 'Cargo.toml', 'go.mod'
        }
        return os.path.basename(file_path) in dep_patterns
    
    def _get_file_content(self, commit: Commit, file_path: str) -> str:
        """Get file content at specific commit."""
        try:
            return commit.tree / file_path.encode()
        except:
            return ""

    #def _get_file_diff(self, commit: Commit, file_path: str) -> str:
        """Get the diff for a specific file in a commit."""
        '''
        if not commit.parents:
            return ""
        
        parent = commit.parents[0]
        diffs = parent.diff(commit, paths=[file_path])
        
        if not diffs:
            return ""
        
        return diffs[0].diff if isinstance(diffs[0].diff, str) else diffs[0].diff.decode('utf-8', errors='ignore') #diffs[0].diff.decode('utf-8', errors='ignore')
        '''

    def _get_file_diff(self, commit: Commit, file_path: str) -> str:
        """Get the diff for a specific file in a commit."""
        logging.info(f"Getting diff for commit {commit.hexsha[:8]} file: {file_path}")
        
        if not commit.parents:
            logging.info("No parent commits found - initial commit")
            try:
                blob = commit.tree / file_path
                content = blob.data_stream.read().decode('utf-8', errors='ignore')
                return f"[Initial commit] Added file {file_path}:\n{content}"
            except Exception as e:
                logging.error(f"Error getting initial commit content: {e}")
                return ""
        
        parent = commit.parents[0]
        import pdb
        #pdb.set_trace()
        logging.info(f"Parent commit: {parent.hexsha[:8]}")

        import subprocess
        try:
            cmd_args = ["git", "show", f"{commit.hexsha}", "--", file_path]
            process = subprocess.Popen(
                cmd_args,
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
                )
            stdout, stderr = process.communicate()
            #input(stdout)
            return stdout
        except Exception as e:
            logging.info(f"Error getting diff: {e}")
        '''

        try:
            # 获取文件在当前commit和父commit中的状态
            try:
                current_blob = commit.tree / file_path
                current_content = current_blob.data_stream.read().decode('utf-8', errors='ignore')
            except:
                current_content = None

            try:
                parent_blob = parent.tree / file_path
                parent_content = parent_blob.data_stream.read().decode('utf-8', errors='ignore')
            except:
                parent_content = None

            # 根据文件状态返回相应的diff
            if parent_content is None and current_content is not None:
                # 新增文件
                logging.info(f"Diff type: Add")
                return f"[Added] {file_path}:\n{current_content}"
            elif parent_content is not None and current_content is None:
                # 删除文件
                logging.info(f"Diff type: Delete")
                return f"[Deleted] {file_path}:\n{parent_content}"
            elif parent_content is not None and current_content is not None:
                # 修改文件 - 使用git的diff
                logging.info(f"Diff type: Modify")
                import difflib
                diff_lines = difflib.unified_diff(
                    current_content,
                    parent_content,
                    fromfile=f'a/{file_path}',
                    tofile=f'b/{file_path}'
                )
                diff_content = diff_lines
                input(diff_content)


                diffs = parent.diff(commit, paths=[file_path])
                if diffs:
                    diff = diffs[0]
                    # 检查diff.diff是bytes还是str类型
                    diff_content = diff.diff.decode('utf-8', errors='ignore') if isinstance(diff.diff, bytes) else diff.diff
                    input(diff.diff)
                    return f"[Modified] {file_path}:\n{diff_content}"
                else:
                    return f"[No changes detected] {file_path}"
            else:
                return ""

        except Exception as e:
            logging.error(f"Error getting diff for {file_path}: {e}")
            return ""
        '''
        '''
        # Try different diff methods
        # 1. Standard diff
        diffs = list(parent.diff(commit))
        logging.info(f"Found {len(diffs)} total diffs")
        
        # Log all changed files
        for d in diffs:
            logging.info(f"Diff found for: {d.a_path} -> {d.b_path}")
        
        # Filter for our specific file
        file_diffs = [d for d in diffs if (d.a_path == file_path or d.b_path == file_path)]
        logging.info(f"Found {len(file_diffs)} diffs for {file_path}")
        
        if file_diffs:
            diff = file_diffs[0]
            logging.info(f"Diff type: {diff.change_type}")
            logging.info(f"a_path: {diff.a_path}, b_path: {diff.b_path}")
            
            # Get the actual diff content
            try:
                if diff.diff:
                    diff_content = diff.diff if isinstance(diff.diff, str) else diff.diff.decode('utf-8', errors='ignore')
                    return diff_content
            except Exception as e:
                logging.error(f"Error getting diff content: {e}")
        
        # 2. Check commit stats
        logging.info("Checking commit stats")
        logging.info(f"Changed files in commit: {list(commit.stats.files.keys())}")
        
        try:
            if file_path in commit.stats.files:
                logging.info(f"File found in commit stats: {file_path}")
                blob = commit.tree / file_path
                content = blob.data_stream.read().decode('utf-8', errors='ignore')
                input(content)
                return f"New file: {file_path}\n\n{content}"
        except Exception as e:
            logging.error(f"Error checking commit stats: {e}")
        
        # 3. Check for deletion
        try:
            in_current = file_path in commit.tree
            in_parent = file_path in parent.tree
            logging.info(f"File in current tree: {in_current}, in parent tree: {in_parent}")
            
            if not in_current and in_parent:
                old_content = (parent.tree / file_path).data_stream.read().decode('utf-8', errors='ignore')
                return f"Deleted file: {file_path}\n\nPrevious content:\n{old_content}"
        except Exception as e:
            logging.error(f"Error checking file deletion: {e}")
        
        logging.warning(f"No diff found for {file_path} in commit {commit.hexsha[:8]}")
        return ""
        '''

    def _prepare_analysis_prompt(self, commit: Commit, file_path: str) -> str:
        """Prepare the prompt for LLM analysis."""
        file_content = self._get_file_content(commit, file_path)
        import pdb
        #pdb.set_trace()

        file_diff = self._get_file_diff(commit, file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Determine language-specific security considerations
        language_context = {
            '.py': 'Python-specific vulnerabilities like input validation, command injection, unsafe deserialization, path traversal, code execution, XSS, authentication bypass',
            '.js': 'JavaScript-specific issues like XSS, prototype pollution, insecure dependencies',
            '.java': 'Java security concerns like unsafe deserialization, SQL injection, improper error handling',
            '.go': 'Go-specific issues like race conditions, memory safety, error handling',
            '.php': 'PHP security issues like file inclusion vulnerabilities, SQL injection, command execution',
            '.rb': 'Ruby-specific concerns like command injection, unsafe deserialization, SQL injection',
        }.get(file_extension, 'General code security issues')
        
        prompt = f"""You are a security expert analyzing code changes for potential vulnerabilities. Please analyze this commit with special attention to {language_context}.

COMMIT INFORMATION:
- Hash: {commit.hexsha}
- Author: {commit.author.name} <{commit.author.email}>
- Date: {commit.authored_datetime}
- Message: {commit.message}

FILE INFORMATION:
- Path: {file_path}
- Type: {file_extension} file
- Changes: Diff shown below

DIFF:
{file_diff}

CURRENT FILE CONTENT:
{file_content}

ANALYSIS INSTRUCTIONS:
1. Code Change Classification:
   - Is this primarily a documentation change?
   - Is this a dependency update?
   - What is the main purpose of these changes?

2. Security Analysis:
   a) Check for common vulnerabilities:
      - Input validation issues
      - Authentication/authorization flaws
      - Cryptographic problems
      - Unsafe data handling
      - Resource management issues
      - Error handling vulnerabilities
   
   b) Language-specific concerns:
      - Review based on {language_context}
      - Check for unsafe language features
      - Identify risky API usage

3. Context Analysis:
   - How do these changes affect the overall security posture?
   - Are there any indirect security implications?
   - Consider interaction with existing code

4. For each identified issue:
   - Describe the vulnerability
   - Assess severity (Critical/High/Medium/Low)
   - Explain potential impact
   - Provide specific fix recommendations
   - Include relevant code snippets

Please format your response as a CommitAnalysis JSON object with the following structure:
{{
    "commit_hash": string,
    "commit_message": string,
    "is_security_related": "False",
    "is_documentation": "False",
    "is_dependency_update": "False",
    "analysis_summary": string,
    "security_issues": [
        {{
            "severity": string,
            "description": string,
            "code_snippet": string,
            "file_path": string,
            "line_number": number=0,
            "recommendation": string
        }}
    ]
}}

Focus on accuracy and actionable insights. Avoid false positives by considering the full context of changes.
"""

        return prompt

    def analyze_commit(self, commit: Commit) -> CommitAnalysis:
        """Analyze a single commit for security issues."""
        log.info(f"Analyzing commit: {commit.hexsha}")
        
        all_files = set()
        if commit.parents:
            # Get changed files
            for diff in commit.parents[0].diff(commit):
                if diff.a_path:
                    all_files.add(diff.a_path)
                if diff.b_path:
                    all_files.add(diff.b_path)
        else:
            # Initial commit - get all files
            for blob in commit.tree.traverse():
                all_files.add(blob.path)

        # Filter out documentation and dependency files
        code_files = {f for f in all_files 
                     if not self._is_documentation_file(f) 
                     and not self._is_dependency_file(f)}

        if not code_files:
            # If no code files were changed, return basic analysis
            return CommitAnalysis(
                commit_hash=commit.hexsha,
                commit_message=commit.message,
                is_security_related=False,
                is_documentation=True,
                is_dependency_update=True,
                security_issues=[],
                analysis_summary="No code files were modified in this commit."
            )

        # Analyze each code file
        security_issues = []
        for file_path in code_files:
            prompt = self._prepare_analysis_prompt(commit, file_path)
            try:
                analysis = self.llm.chat(prompt, response_model=CommitAnalysis)
                security_issues.extend(analysis.security_issues)
            except Exception as e:
                log.error(f"Error analyzing file {file_path}: {str(e)}")

        return CommitAnalysis(
            commit_hash=commit.hexsha,
            commit_message=commit.message,
            is_security_related=len(security_issues) > 0,
            is_documentation=False,
            is_dependency_update=False,
            security_issues=security_issues,
            analysis_summary=f"Analyzed {len(code_files)} files. Found {len(security_issues)} security issues."
        )

    def analyze_commits(self, num_commits: int = 10) -> List[CommitAnalysis]:
        """Analyze the most recent commits."""
        results = []
        for commit in list(self.repo.iter_commits())[:num_commits]:
            try:
                analysis = self.analyze_commit(commit)
                results.append(analysis)
            except Exception as e:
                log.error(f"Error analyzing commit {commit.hexsha}: {str(e)}")
                continue
        return results

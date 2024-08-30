from github import Github
import base64
import logging

class GitHubRepo:
    def __init__(self, token, repo_name):
        self.g = Github(token)
        self.repo = self.g.get_repo(repo_name)

    def get_file_content(self, file_path):
        try:
            file_content = self.repo.get_contents(file_path)
            if file_content.size > 1000000:  # 1MB limit
                return "File is too large to fetch content directly."
            content = base64.b64decode(file_content.content).decode('utf-8')
            return content
        except Exception as e:
            return f"Error fetching file: {str(e)}"

    def get_directory_structure(self, path="", prefix="", max_depth=2, current_depth=0):
        if current_depth > max_depth:
            return []

        contents = self.repo.get_contents(path)
        structure = []
        for content in contents:
            if content.type == "dir":
                structure.append(f"{prefix}{content.name}/")
                if current_depth < max_depth:
                    structure.extend(self.get_directory_structure(
                        content.path, 
                        prefix + "  ", 
                        max_depth, 
                        current_depth + 1
                    ))
            else:
                structure.append(f"{prefix}{content.name}")
        return structure


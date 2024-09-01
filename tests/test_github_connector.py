import unittest
import os
from dotenv import load_dotenv
from tars_bot.github_utils import GithubConnector

class TestGithubConnector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()  # Load environment variables from .env file
        cls.connector = GithubConnector(
            os.getenv('GITHUB_TOKEN'), 
            os.getenv('GITHUB_REPO')
        )

    def test_get_file_content(self):
        # Use an actual file in the repo
        content = self.connector.get_file_content("README.md")  # Replace with the actual path in your repo
        print(content[:42])
        self.assertIsNotNone(content)
        self.assertIsInstance(content, str)

    def test_get_dir_structure(self):
        structure = self.connector.get_dir_structure(max_depth=1)
        print(structure)
        self.assertIsInstance(structure, list)

    def test_get_local_file_content(self):
        # Use the actual principles.md file in the repo
        with open("principles.md", "r") as f:  # Ensure this file exists in your local repo
            expected_content = f.read()
        
        content = self.connector.get_local_file_content("principles.md")
        print(content[:42])
        self.assertEqual(content, expected_content)

    def test_extract_principles(self):
        markdown_content = "# Header 1\n## Header 2\n- List item"
        expected_output = "# 1 Header 1\n## 1.1 Header 2\n  - List item"
        output = self.connector.extract_principles(markdown_content)
        print(output[:42])
        self.assertEqual(output.strip(), expected_output.strip())

    def test_generate_prompt(self):
        file_path = "path/to/file.py"
        repo_code = "print('Hello World')"
        principles = "# 1 Principle"
        task_description = "Explain the code."
        code_type = "python"
        
        prompt = self.connector.generate_prompt(file_path, repo_code, principles, task_description, code_type)
        self.assertIn("Context", prompt)
        self.assertIn("Core Principles", prompt)
        self.assertIn("Task", prompt)

if __name__ == '__main__':
    unittest.main()
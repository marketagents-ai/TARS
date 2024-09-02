import unittest
from unittest.mock import patch, MagicMock
from TARS.tars_bot.github_tools import GitHubRepo

class TestGitHubRepo(unittest.TestCase):

    @patch('TARS.tars_bot.github_tools.Github')
    def setUp(self, MockGithub):
        self.mock_github = MockGithub.return_value
        self.mock_repo = self.mock_github.get_repo.return_value
        self.github_repo = GitHubRepo('fake_token', 'fake_repo')

    def test_get_file_content_success(self):
        mock_file = MagicMock()
        mock_file.size = 500
        mock_file.content = base64.b64encode(b'file content').decode('utf-8')
        self.mock_repo.get_contents.return_value = mock_file

        content = self.github_repo.get_file_content('path/to/file')
        self.assertEqual(content, 'file content')

    def test_get_file_content_large_file(self):
        mock_file = MagicMock()
        mock_file.size = 2000000
        self.mock_repo.get_contents.return_value = mock_file

        content = self.github_repo.get_file_content('path/to/file')
        self.assertEqual(content, 'File is too large to fetch content directly.')

    def test_get_file_content_error(self):
        self.mock_repo.get_contents.side_effect = Exception('Error')

        content = self.github_repo.get_file_content('path/to/file')
        self.assertEqual(content, 'Error fetching file: Error')

    def test_get_directory_structure(self):
        mock_dir = MagicMock()
        mock_dir.type = 'dir'
        mock_dir.name = 'dir'
        mock_dir.path = 'dir'

        mock_file = MagicMock()
        mock_file.type = 'file'
        mock_file.name = 'file'

        self.mock_repo.get_contents.side_effect = [[mock_dir, mock_file], []]

        structure = self.github_repo.get_directory_structure()
        self.assertEqual(structure, ['dir/', '  file'])

if __name__ == '__main__':
    unittest.main()
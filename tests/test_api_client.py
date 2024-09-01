import unittest
import asyncio
from tars_bot.api_client import call_api, initialize_api_client
from dotenv import load_dotenv

load_dotenv()

class TestAPIClient(unittest.TestCase):

    def setUp(self):
        # Initialize with real environment variables
        args = type('Args', (object,), {})()  # Create a simple object to hold attributes
        args.api = 'openrouter'  # Change to 'ollama' or 'openrouter' as needed
        args.model = None  # You can specify a model if needed
        initialize_api_client(args)

    def test_call_api(self):
        # Run the async function in the event loop
        response = asyncio.run(call_api("Hey TARS, what's on the other side of the event horizon?"))
        print(response)
        self.assertIsNotNone(response)  # Check if response is not None
        self.assertIsInstance(response, str)  # Check if response is a string # Check if the response contains an error key

if __name__ == '__main__':
    unittest.main()
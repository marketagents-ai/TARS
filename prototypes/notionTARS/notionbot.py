import os
import aiohttp
import json
import yaml
import logging
import asyncio
import discord
from discord.ext import commands, tasks
from discord import Intents
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

NOTION_API_KEY = os.getenv('NOTION_API_KEY')
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
USER_MAPPING_FILE = 'user_mapping.yaml'
KANBAN_DB_ID = os.getenv('KANBAN_DB_ID')
DISCORD_CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID'))
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', 120))

# Set up logging
logging.basicConfig(level=os.getenv('LOGLEVEL', 'INFO'), 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KanbanCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.last_checked = None
        self.item_states = {}
        self.ensure_cache_dir()
        self.load_cache()

    def ensure_cache_dir(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                self.last_checked = datetime.fromisoformat(data['last_checked'])
                self.item_states = data['item_states']
        else:
            self.last_checked = datetime.now(timezone.utc)
            self.item_states = {}

    def save_cache(self):
        data = {
            'last_checked': self.last_checked.isoformat(),
            'item_states': self.item_states
        }
        with open(self.cache_file, 'w') as f:
            json.dump(data, f)

    def update_item_state(self, item_id, state):
        self.item_states[item_id] = state
        self.save_cache()

    def remove_item_state(self, item_id):
        if item_id in self.item_states:
            del self.item_states[item_id]
            self.save_cache()

    def update_last_checked(self):
        self.last_checked = datetime.now(timezone.utc)
        self.save_cache()

class NotionAPI:
    BASE_URL = 'https://api.notion.com/v1'

    def __init__(self, token):
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Notion-Version': '2022-06-28',
            'Content-Type': 'application/json'
        }

    async def test_connection(self):
        url = f'{self.BASE_URL}/users/me'
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Successfully connected to Notion API. User: {data['name']}")
                else:
                    error_body = await response.text()
                    logger.error(f"Error connecting to Notion API: Status {response.status}, Body: {error_body}")

    async def query_database(self, database_id, filter_params=None, sorts=None):
        url = f'{self.BASE_URL}/databases/{database_id}/query'
        payload = {}
        if filter_params:
            payload['filter'] = filter_params
        if sorts:
            payload['sorts'] = sorts

        logger.debug(f"Querying database: {database_id}")
        logger.debug(f"Payload: {payload}")

        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_body = await response.text()
                    logger.error(f"Error querying database: Status {response.status}, Body: {error_body}")
                    return None

    async def get_kanban_updates(self, database_id, last_checked):
        # Remove the filter to fetch all items
        return await self.query_database(database_id)

    async def get_board_details(self, database_id):
        url = f'{self.BASE_URL}/databases/{database_id}'
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    description = 'No description'
                    if 'description' in data:
                        if isinstance(data['description'], list) and len(data['description']) > 0:
                            description = data['description'][0].get('plain_text', 'No description')
                        elif isinstance(data['description'], dict):
                            description = data['description'].get('plain_text', 'No description')
                    return {
                        'title': data['title'][0]['plain_text'] if data['title'] else 'Untitled',
                        'description': description,
                        'created_time': data['created_time'],
                        'last_edited_time': data['last_edited_time']
                    }
                else:
                    error_body = await response.text()
                    logger.error(f"Error fetching board details: Status {response.status}, Body: {error_body}")
                    return None

    async def get_database_properties(self, database_id):
        url = f'{self.BASE_URL}/databases/{database_id}'
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('properties', {})
                else:
                    error_body = await response.text()
                    logger.error(f"Error fetching database properties: Status {response.status}, Body: {error_body}")
                    return None

    async def get_users(self):
        url = f'{self.BASE_URL}/users'
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', [])
                else:
                    error_body = await response.text()
                    logger.error(f"Error fetching users: Status {response.status}, Body: {error_body}")
                    return None

    async def check_user_permission(self, database_id, user_id):
        url = f'{self.BASE_URL}/databases/{database_id}'
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Database access response: {json.dumps(data, indent=2)}")
                    
                    # Check if the user is the owner of the database
                    if data.get('created_by', {}).get('id') == user_id:
                        logger.info(f"User {user_id} is the owner of the database {database_id}")
                        return True

                    # Check user permissions
                    permissions = data.get('permissions', [])
                    for permission in permissions:
                        logger.debug(f"Checking permission: {json.dumps(permission, indent=2)}")
                        if permission['type'] == 'user_permission' and permission['user']['id'] == user_id:
                            logger.info(f"User {user_id} has explicit permission for database {database_id}")
                            return True
                    
                    logger.warning(f"User {user_id} does not have explicit permission for database {database_id}")
                    return False
                else:
                    error_body = await response.text()
                    logger.error(f"Error checking user permission: Status {response.status}, Body: {error_body}")
                    return None

    async def get_user_info(self, user_id):
        url = f'{self.BASE_URL}/users/{user_id}'
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"User info for {user_id}: {json.dumps(data, indent=2)}")
                    return data
                else:
                    error_body = await response.text()
                    logger.error(f"Error fetching user info: Status {response.status}, Body: {error_body}")
                    return None

    async def print_all_users(self):
        users = await self.get_users()
        if users:
            logger.info("Notion Users:")
            for user in users:
                logger.info(f"Name: {user['name']}, ID: {user['id']}")
        else:
            logger.warning("No users found or error occurred while fetching users.")

    async def get_databases(self):
        url = f'{self.BASE_URL}/search'
        payload = {
            "filter": {
                "value": "database",
                "property": "object"
            }
        }
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', [])
                else:
                    error_body = await response.text()
                    logger.error(f"Error fetching databases: Status {response.status}, Body: {error_body}")
                    return None

    async def print_all_databases(self):
        databases = await self.get_databases()
        if databases:
            logger.info("Available Notion Databases:")
            for db in databases:
                logger.info(f"Name: {db['title'][0]['plain_text']}, ID: {db['id']}")
        else:
            logger.warning("No databases found or error occurred while fetching databases.")

notion_api = NotionAPI(NOTION_API_KEY)

# Initialize the cache
cache_dir = 'cache'
cache_file = os.path.join(cache_dir, 'kanban_cache.json')
kanban_cache = KanbanCache(cache_file)

# Set up Discord bot
intents = Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

def load_user_mapping():
    try:
        with open(USER_MAPPING_FILE, 'r') as file:
            return yaml.safe_load(file) or {}
    except Exception as e:
        logger.error(f"Error loading user mapping: {e}")
        return {}

def save_user_mapping(mapping):
    try:
        with open(USER_MAPPING_FILE, 'w') as file:
            yaml.dump(mapping, file)
        logger.info("User mapping saved successfully.")
    except Exception as e:
        logger.error(f"Error saving user mapping: {e}")

user_mapping = load_user_mapping()

def get_notion_user_id(discord_id):
    notion_id = user_mapping.get(str(discord_id))
    if not notion_id:
        logger.warning(f"No Notion user ID found for Discord ID: {discord_id}")
    return notion_id

def get_item_state(item):
    # Create a hashable representation of the item's current state
    return json.dumps(item, sort_keys=True)

async def format_kanban_update(item, board_name, properties, update_type, notion_api):
    title = item['properties'].get('Name', {}).get('title', [{}])[0].get('text', {}).get('content', 'Untitled')
    status = 'Unknown'
    for prop_name, prop_value in item['properties'].items():
        if properties[prop_name]['type'] == 'status':
            status = prop_value['status']['name']
            break
    last_edited_time = datetime.fromisoformat(item['last_edited_time'].rstrip('Z')).strftime("%Y-%m-%d %H:%M:%S")
    
    editor_id = item.get('last_edited_by', {}).get('id', 'Unknown')
    editor_info = await notion_api.get_user_info(editor_id)
    editor_name = editor_info.get('name', 'Unknown User') if editor_info else 'Unknown User'
    
    return f"**{update_type} in {board_name}:**\n• Task: {title}\n• Status: {status}\n• Last Edited: {last_edited_time}\n• Edited by: {editor_name}\n"


@tasks.loop(seconds=POLL_INTERVAL)
async def poll_notion_database():
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if not channel:
        logger.error(f"Could not find channel with id {DISCORD_CHANNEL_ID}")
        return

    try:
        board_details = await notion_api.get_board_details(KANBAN_DB_ID)
        if not board_details:
            logger.error("Failed to fetch board details")
            return

        properties = await notion_api.get_database_properties(KANBAN_DB_ID)
        if not properties:
            logger.error("Failed to fetch database properties")
            return

        kanban_data = await notion_api.get_kanban_updates(KANBAN_DB_ID, kanban_cache.last_checked)
        if kanban_data and kanban_data.get('results'):
            for item in kanban_data['results']:
                item_id = item['id']
                current_state = get_item_state(item)
                
                if item_id not in kanban_cache.item_states:
                    update_type = "New item added"
                elif current_state != kanban_cache.item_states[item_id]:
                    update_type = "Item updated"
                else:
                    continue  # No changes, skip this item
                
                message = await format_kanban_update(item, board_details['title'], properties, update_type, notion_api)
                await channel.send(message)
                kanban_cache.update_item_state(item_id, current_state)
            
            # Remove items that no longer exist
            for old_item_id in list(kanban_cache.item_states.keys()):
                if old_item_id not in [item['id'] for item in kanban_data['results']]:
                    kanban_cache.remove_item_state(old_item_id)
                    await channel.send(f"Item with ID {old_item_id} has been removed from the Kanban board.")
        else:
            logger.info("No Kanban data found")

        kanban_cache.update_last_checked()
    except Exception as e:
        logger.error(f"Error polling Notion database: {e}", exc_info=True)

@bot.event
async def on_ready():
    logger.info(f"{bot.user} is now online!")
    await notion_api.test_connection()
    await notion_api.print_all_users()
    await notion_api.print_all_databases()
    poll_notion_database.start()

async def check_board_access(ctx, board_id):
    notion_user_id = get_notion_user_id(ctx.author.id)
    if not notion_user_id:
        await ctx.send("Your Discord account is not linked to a Notion user. Use `!link_user <notion_user_id>` to link your account.")
        return False

    has_permission = await notion_api.check_user_permission(board_id, notion_user_id)
    if has_permission is None:
        await ctx.send("An error occurred while checking your permissions. Please try again later.")
        return False
    elif not has_permission:
        await ctx.send(f"You don't have permission to access the Kanban board with ID: {board_id}")
        return False
    return True

@bot.command(name='notion_kanban')
async def generate_kanban_report(ctx, board_id=None):
    board_id = board_id or KANBAN_DB_ID
    if not await check_board_access(ctx, board_id):
        return

    try:
        board_details = await notion_api.get_board_details(board_id)
        if not board_details:
            await ctx.send("Failed to fetch board details. Please check the board ID.")
            return

        properties = await notion_api.get_database_properties(board_id)
        if not properties:
            await ctx.send("Failed to fetch board properties. Please check the board ID.")
            return

        # Fetch all Kanban data, not just recent updates
        kanban_data = await notion_api.query_database(board_id)

        if not kanban_data or not kanban_data.get('results'):
            await ctx.send(f"No Kanban board data found for board ID: {board_id}. The board might be empty.")
            return

        report = f"# Kanban Board Report for {ctx.author.name}\n\n"
        report += f"## Board: {board_details['title']}\n"
        report += f"Description: {board_details['description']}\n"
        report += f"Last Edited: {board_details['last_edited_time']}\n\n"

        status_groups = {}
        for item in kanban_data['results']:
            status = 'Unknown'
            for prop_name, prop_value in item['properties'].items():
                if properties[prop_name]['type'] == 'status':
                    status = prop_value['status']['name']
                    break
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(item)

        for status, items in status_groups.items():
            report += f"### {status}\n\n"
            for item in items:
                title = item['properties'].get('Name', {}).get('title', [{}])[0].get('text', {}).get('content', 'Untitled')
                report += f"- {title}\n"
                for prop_name, prop_value in item['properties'].items():
                    if prop_name != 'Name' and prop_name != 'Status':
                        value = get_property_value(prop_value)
                        if value:  # Only add the property if it has a value
                            report += f"  - {prop_name}: {value}\n"
                report += "\n"

        if len(report) > 2000:
            parts = [report[i:i+1990] for i in range(0, len(report), 1990)]
            for part in parts:
                await ctx.send(f"```markdown\n{part}\n```")
        else:
            await ctx.send(f"```markdown\n{report}\n```")
    except Exception as e:
        logger.error(f"Error generating Kanban report: {e}", exc_info=True)
        await ctx.send(f"An error occurred while generating the Kanban report: {str(e)}")

def get_property_value(prop):
    prop_type = prop['type']
    if prop_type == 'rich_text':
        return ' '.join([text['plain_text'] for text in prop['rich_text'] if text['plain_text']])
    elif prop_type == 'select':
        return prop['select']['name'] if prop['select'] else None
    elif prop_type == 'multi_select':
        return ', '.join([option['name'] for option in prop['multi_select']])
    elif prop_type == 'date':
        if prop['date']:
            start = prop['date']['start']
            end = prop['date'].get('end')
            return f"{start} - {end}" if end else start
        return None
    elif prop_type == 'people':
        return ', '.join([person['name'] for person in prop['people']])
    elif prop_type == 'files':
        return ', '.join([file['name'] for file in prop['files']])
    elif prop_type == 'checkbox':
        return 'Yes' if prop['checkbox'] else 'No'
    elif prop_type == 'number':
        return str(prop['number']) if prop['number'] is not None else None
    elif prop_type == 'formula':
        return str(prop['formula'].get('string', ''))
    elif prop_type == 'status':
        return prop['status']['name'] if prop['status'] else None
    elif prop_type == 'title':
        return ' '.join([text['plain_text'] for text in prop['title'] if text['plain_text']])
    else:
        return None  # Return None for unsupported types, which will be filtered out

async def debug_board_access(ctx, board_id):
    notion_user_id = get_notion_user_id(ctx.author.id)
    if not notion_user_id:
        await ctx.send("Your Discord account is not linked to a Notion user. Use `!link_user <notion_user_id>` to link your account.")
        return False

    user_info = await notion_api.get_user_info(notion_user_id)
    if user_info:
        await ctx.send(f"Notion user info: Name: {user_info.get('name')}, Type: {user_info.get('type')}")
    else:
        await ctx.send("Failed to fetch Notion user info. The user ID might be incorrect.")
        return False

    has_permission = await notion_api.check_user_permission(board_id, notion_user_id)
    if has_permission is None:
        await ctx.send("An error occurred while checking your permissions. Please try again later.")
        return False
    elif not has_permission:
        await ctx.send(f"You don't have permission to access the Kanban board with ID: {board_id}")
        board_details = await notion_api.get_board_details(board_id)
        if board_details:
            await ctx.send(f"Board details: Title: {board_details['title']}, Created by: {board_details.get('created_by', {}).get('id')}")
        return False
    await ctx.send(f"You have permission to access the Kanban board with ID: {board_id}")
    return True

@bot.command(name='debug_kanban')
async def debug_kanban_access(ctx, board_id=None):
    board_id = board_id or KANBAN_DB_ID
    await debug_board_access(ctx, board_id)

@bot.command(name='list_boards')
async def list_kanban_boards(ctx):
    try:
        databases = await notion_api.get_databases()
        if not databases:
            await ctx.send("No Kanban boards found or error occurred while fetching boards.")
            return

        report = "# Available Kanban Boards\n\n"
        for db in databases:
            report += f"- {db['title'][0]['plain_text']}\n"
            report += f"  ID: {db['id']}\n\n"

        await ctx.send(f"```markdown\n{report}\n```")
    except Exception as e:
        logger.error(f"Error listing Kanban boards: {e}", exc_info=True)
        await ctx.send("An error occurred while listing Kanban boards. Please try again later.")

@bot.command(name='link_user')
async def link_notion_user(ctx, notion_user_id: str = None):
    if notion_user_id is None:
        await ctx.send("Please provide a Notion user ID. Usage: `!link_user <notion_user_id>`")
        return
    
    user_mapping[str(ctx.author.id)] = notion_user_id
    save_user_mapping(user_mapping)
    await ctx.send(f"Your Discord account has been linked to Notion user ID: {notion_user_id}")

@link_notion_user.error
async def link_notion_user_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Please provide a Notion user ID. Usage: `!link_user <notion_user_id>`")

@bot.command(name='unlink_user')
async def unlink_notion_user(ctx):
    if str(ctx.author.id) in user_mapping:
        del user_mapping[str(ctx.author.id)]
        save_user_mapping(user_mapping)
        await ctx.send("Your Discord account has been unlinked from Notion.")
    else:
        await ctx.send("Your Discord account was not linked to any Notion user.")

@bot.command(name='check_mapping')
async def check_user_mapping(ctx):
    notion_user_id = get_notion_user_id(ctx.author.id)
    if notion_user_id:
        await ctx.send(f"Your Discord account is linked to Notion user ID: {notion_user_id}")
    else:
        await ctx.send("Your Discord account is not linked to any Notion user ID.")

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
from collections import defaultdict
import yaml
from cache_manager import CacheManager
from api_client import call_api
import io
import discord

class ChannelSummarizer:
    def __init__(self, bot, cache_manager: CacheManager, max_entries=100):
        self.bot = bot
        self.cache_manager = cache_manager
        self.max_entries = max_entries
        self.load_config()

    def load_config(self):
        with open('prompt_formats.yaml', 'r') as file:
            self.prompt_formats = yaml.safe_load(file)
        with open('system_prompts.yaml', 'r') as file:
            self.system_prompts = yaml.safe_load(file)

    async def summarize_and_send(self, channel_id, user):
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return "Channel not found."

        summary = await self.summarize_channel(channel)
        
        # Create a Markdown file with the summary
        file_content = f"# Summary of #{channel.name} in {channel.guild.name}\n\n"
        file_content += f"Last {self.max_entries} messages\n\n"
        file_content += summary

        # Create a Discord file object
        file = discord.File(io.StringIO(file_content), filename=f"summary_{channel.name}.md")

        # Send the file via DM
        try:
            await user.send(file=file)
            return True
        except discord.Forbidden:
            return False

    async def summarize_channel(self, channel):
        main_messages = []
        threads = defaultdict(list)

        async for message in channel.history(limit=self.max_entries):
            if hasattr(message, 'thread') and message.thread:
                threads[message.thread.id].append(message)
            else:
                main_messages.append(message)

        summary = f"## Main Channel\n\n"
        summary += await self._summarize_messages(main_messages, "Main Channel")

        for thread_id, thread_messages in threads.items():
            thread = channel.get_thread(thread_id)
            if thread:
                thread_summary = await self._summarize_messages(thread_messages, f"Thread: {thread.name}")
                summary += f"\n## Thread: {thread.name}\n\n{thread_summary}"

        return summary

    async def _summarize_messages(self, messages, context):
        user_message_counts = defaultdict(int)
        file_types = defaultdict(int)
        content_chunks = []
        
        for message in messages:
            user_message_counts[message.author.name] += 1
            for attachment in message.attachments:
                file_type = attachment.filename.split('.')[-1].lower()
                file_types[file_type] += 1
            
            content_chunks.append(f"{message.author.name}: {message.content}")

        summary = f"### Participants\n\n"
        for user, count in user_message_counts.items():
            summary += f"- {user}: {count} messages\n"

        if file_types:
            summary += "\n### Shared Files\n\n"
            for file_type, count in file_types.items():
                summary += f"- {file_type}: {count} files\n"

        content_summary = await self._process_chunks(content_chunks, context)
        summary += f"\n### Content Summary\n\n{content_summary}\n"

        return summary

    async def _process_chunks(self, chunks, context):
        prompt = self.prompt_formats['summarize_channel'].format(
            context=context,
            content="\n".join(chunks)
        )
        
        system_prompt = self.system_prompts['channel_summarization']
        
        try:
            return await call_api(prompt, context="", system_prompt=system_prompt)
        except Exception as e:
            return f"Error in generating summary: {str(e)}"
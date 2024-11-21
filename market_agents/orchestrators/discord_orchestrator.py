from datetime import datetime
import os
import yaml
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
import asyncio

from market_agents.agents.market_agent import MarketAgent
from market_agents.inference.message_models import LLMConfig
from market_agents.agents.personas.persona import Persona
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.environments.mechanisms.discord import (
    DiscordMechanism,
    DiscordActionSpace,
    DiscordObservationSpace
)

import discord

# Import UserMemoryIndex to handle memory storage
from agent.memory import UserMemoryIndex

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MessageProcessor:
    def __init__(self, bot):
        self.bot = bot
        self.bot_id = None
        self.agent = None
        self.environment = None

        # Initialize the memory index for storing reflections
        self.memory_index = UserMemoryIndex('agent_memory_index')

    async def initialize_bot_id(self):
        """Initialize bot_id once the bot is ready"""
        if self.bot.user:
            self.bot_id = str(self.bot.user.id)
        else:
            raise ValueError("Bot user is not initialized")

    async def setup_agent(self):
        """Initialize the TARS agent with persona and environment"""
        try:
            # Load personas
            personas_dir = Path("./market_agents/agents/personas/generated_personas")
            existing_personas = []

            for filename in os.listdir(personas_dir):
                if filename.endswith(".yaml"):
                    with open(os.path.join(personas_dir, filename), 'r') as file:
                        persona_data = yaml.safe_load(file)
                        # Create Persona instance using pydantic model
                        try:
                            persona = Persona(
                                name=persona_data['name'],
                                role=persona_data['role'],
                                persona=persona_data['persona'],
                                objectives=persona_data['objectives'],
                                trader_type=persona_data['trader_type'],
                                communication_style=persona_data['communication_style'],
                                routines=persona_data['routines'],
                                skills=persona_data['skills']
                            )
                            existing_personas.append(persona)
                        except Exception as e:
                            logger.error(f"Error creating Persona object from {filename}: {str(e)}")
                            continue

            # Find TARS persona
            agent_persona = next((p for p in existing_personas if p.name == "TARS"), None)
            if not agent_persona:
                raise ValueError("TARS persona not found in generated personas")

            # Create Discord environment
            discord_mechanism = DiscordMechanism()
            self.environment = MultiAgentEnvironment(
                name="DiscordEnvironment",
                action_space=DiscordActionSpace(),
                observation_space=DiscordObservationSpace(),
                mechanism=discord_mechanism,
                max_steps=1000
            )

            # Configure LLM
            llm_config = LLMConfig(
                client="openai",
                model="gpt-4o",
                temperature=0.7,
                max_tokens=1024
            )

            # Create agent
            self.agent = MarketAgent.create(
                agent_id=self.bot_id,
                use_llm=True,
                llm_config=llm_config,
                environments={'discord': self.environment},
                persona=agent_persona,
                econ_agent=None
            )

            logger.info("Agent setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}", exc_info=True)
            return False

    async def process_messages(self, channel_info, messages):
        """Process messages through the agent's cognitive functions"""
        try:
            if not messages:
                logger.warning("No messages to process")
                return

            # Update environment state
            environment_info = {
                "bot_id": self.bot_id,
                "channel_id": channel_info["id"],
                "channel_name": channel_info["name"],
                "messages": messages
            }
            self.environment.mechanism.update_state(environment_info)
            
            # Run cognitive functions
            logger.info("Starting agent cognitive functions")

            # Perception
            perception_result = await self.agent.perceive('discord')
            logger.info("Perception completed")
            print("\nPerception Result:")
            print("\033[94m" + json.dumps(perception_result, indent=2) + "\033[0m")

            # Action Generation
            action_result = await self.agent.generate_action(
                'discord', 
                perception=perception_result
            )
            logger.info("Action generation completed")
            print("\nAction Result:")
            print("\033[92m" + json.dumps(action_result, indent=2) + "\033[0m")

            # Create task for reflection to run in parallel
            reflection_task = asyncio.create_task(self._run_reflection())

            # Return action result immediately
            response = {
                "perception": perception_result,
                "action": action_result,
                "reflection": None
            }

            # Clear messages from mechanism after processing core functions
            self.environment.mechanism.messages = []
            logger.info("Cleared messages from mechanism")

            return response

        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}", exc_info=True)
            return None

    async def _run_reflection(self):
        """Run reflection as a separate async task"""
        try:
            # Reflection
            reflection_result = await self.agent.reflect('discord')
            logger.info("Reflection completed")
            print("\nReflection Result:")
            print("\033[93m" + json.dumps(reflection_result, indent=2) + "\033[0m")

            # Store reflection outputs to memory
            if self.agent.memory:
                last_memory = self.agent.memory[-1]
                if last_memory.get('type') == 'reflection':
                    reflection_content = last_memory.get('content', '')
                    timestamp = last_memory.get('timestamp', datetime.now().isoformat())
                    # Save to memory index
                    user_id = self.agent.id
                    memory_text = f"Reflection at {timestamp}: {reflection_content}"
                    self.memory_index.add_memory(user_id, memory_text)
                    self.memory_index.save_cache()
                    logger.info("Saved reflection to memory index")

            return reflection_result

        except Exception as e:
            logger.error(f"Error during reflection: {str(e)}", exc_info=True)
            return None

async def run_bot():
    """Run the Discord bot and process messages once connected."""
    # Create a Discord client
    intents = discord.Intents.default()
    intents.messages = True
    intents.guilds = True
    client = discord.Client(intents=intents)

    # Initialize the message processor with the client (bot)
    processor = MessageProcessor(client)
    processing_done = asyncio.Event()

    @client.event
    async def on_ready():
        logger.info(f'Logged in as {client.user.name} (ID: {client.user.id})')

        await processor.initialize_bot_id()
        success = await processor.setup_agent()

        if not success:
            logger.error("Failed to set up agent")
            processing_done.set()
            return

        # Get the channel to test with
        channel = client.get_channel(int(os.getenv('DISCORD_CHANNEL_ID')))

        if channel:
            # Get channel info
            channel_info = {
                "id": str(channel.id),
                "name": channel.name
            }
            
            # Fetch messages
            messages = []
            async for msg in channel.history(limit=10):
                messages.append({
                    "content": msg.content,
                    "author_id": str(msg.author.id),
                    "author_name": msg.author.name,
                    "timestamp": msg.created_at.isoformat()
                })
            
            print(f"Discord messages: {messages}")

            # Process the messages
            results = await processor.process_messages(channel_info, messages)
            
            if results:
                logger.info("Message processing test completed successfully")
            else:
                logger.error("Message processing test failed")
        else:
            logger.error("Channel not found")

        # Signal that processing is done
        processing_done.set()

    # Get Discord token
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("Discord token not found in environment variables")
        return

    # Run the bot
    await client.start(token)

    # Wait until processing is done
    await processing_done.wait()

    # Close the bot connection
    await client.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_bot())

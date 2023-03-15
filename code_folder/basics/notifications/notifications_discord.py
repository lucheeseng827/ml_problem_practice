# First, install the library:
# !pip install discord.py

# Import the necessary libraries
import discord

# Set the Discord API token and client ID
TOKEN = 'DISCORD_API_TOKEN_GOES_HERE'
CLIENT_ID = 'DISCORD_CLIENT_ID_GOES_HERE'

# Create a client object
client = discord.Client()

# Set the message that you want to send
message = 'MESSAGE_GOES_HERE'

# Set the ID of the channel that you want to send the message to
channel_id = 'CHANNEL_ID_GOES_HERE'

# Set the name of the server that the channel belongs to
server_name = 'SERVER_NAME_GOES_HERE'

# Use the `on_ready` event to ensure that the client is ready to send the message


@client.event
async def on_ready():
    # Find the server with the specified name
    server = discord.utils.get(client.guilds, name=server_name)
    # Find the channel with the specified ID
    channel = discord.utils.get(server.channels, id=channel_id)
    # Send the message to the channel
    await channel.send(message)

# Run the client
client.run(TOKEN)

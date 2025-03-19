# Gaming Store AI Chatbot

A smart AI chatbot for gaming store websites that can recommend games, assist with navigation, provide technical support, and engage the community through challenges.

## Features

- **Game Recommendations**: Suggests games based on user preferences, including genres, platforms, and price sensitivity.
- **Memory System**: Remembers user preferences and past interactions to provide contextual responses.
- **Context Awareness**: Understands when users change topics or follow up on previous questions.
- **Technical Support**: Provides troubleshooting steps for common gaming issues.
- **FAQ Handling**: Answers frequently asked questions about the store's policies and services.
- **Navigation Assistance**: Helps users find their way around the gaming store website.
- **Community Engagement**: Offers exclusive quests and challenges to engage users.

## Sample Game Data

The chatbot comes preloaded with information about popular games including:
- Elden Ring
- Red Dead Redemption 2
- The Legend of Zelda: Breath of the Wild
- Cyberpunk 2077
- Minecraft
- God of War
- Fortnite
- The Witcher 3: Wild Hunt
- Among Us
- Call of Duty: Modern Warfare II

## Requirements

Python 3.6 or higher

## Installation

1. Clone this repository
2. Navigate to the project directory
3. Run the chatbot:
   ```
   python gaming_chatbot.py
   ```

## Usage

When you run the chatbot, you can interact with it by typing messages. Here are some example interactions:

1. **Game Recommendations**:
   - "Can you recommend some action games?"
   - "I'm looking for multiplayer games on PlayStation"
   - "Suggest some free to play games"

2. **Game Information**:
   - "Tell me about Elden Ring"
   - "What platforms is Minecraft available on?"
   - "How much does Fortnite cost?"

3. **Technical Support**:
   - "I'm having trouble downloading a game"
   - "My game keeps crashing"
   - "How do I fix installation issues?"

4. **Navigation Help**:
   - "How do I find new releases?"
   - "Where is the wishlist?"
   - "How to browse games by genre?"

5. **Community Features**:
   - "Are there any challenges I can participate in?"
   - "Tell me about tournaments"
   - "How can I join the community forums?"

## Customization

You can extend the chatbot by:

1. Adding more games to the `load_game_data` method
2. Expanding the FAQ database in the `__init__` method
3. Creating more intent patterns to recognize different user requests
4. Enhancing the community challenges

## Integration

To integrate this chatbot with your website:

1. Use this as a backend service
2. Create an API endpoint that accepts user messages and returns the chatbot's responses
3. Connect your frontend UI to this API

## License

This project is available for free use and modification. 
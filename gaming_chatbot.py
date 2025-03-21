import re
import random
import json
import time
from datetime import datetime
from collections import defaultdict
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import joblib
import os
import torch
from torch.nn.functional import cosine_similarity as torch_cosine_similarity

# Import data from data.py
from data import (
    games_data,
    game_vocabulary,
    faqs,
    navigation_guides,
    challenges,
    translations,
    reasoning_rules,
    intent_patterns,
    synonyms,
    platform_sections,
    platform_faqs
)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('omw-1.4')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('movie_reviews')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('taggers')
nltk.download('tokenizers')
nltk.download('corpora')

class GamingChatbot:
    def __init__(self):
        """Initialize the chatbot with necessary data"""
        from data import (games_data, challenges, game_vocabulary,
                        navigation_guides, platform_sections, platform_faqs,
                        faqs, translations, reasoning_rules, intent_patterns,
                        synonyms)
        
        self.games_data = games_data
        self.challenges = challenges
        self.game_vocabulary = game_vocabulary
        self.navigation_guides = navigation_guides
        self.platform_sections = platform_sections
        self.platform_faqs = platform_faqs
        self.faqs = faqs
        self.translations = translations
        self.reasoning_rules = reasoning_rules
        self.intent_patterns = intent_patterns
        self.synonyms = synonyms
        
        # Initialize user memory and conversation history
        self.user_memory = {}
        self.conversation_history = {}
        
        # Initialize response cache
        self.response_cache = {}
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')
        self.stop_words = set(stopwords.words('english') + stopwords.words('french'))
        
        # Initialize ML models
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english") 
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load or create training data
        self.training_data = self.load_training_data()
        
        # Initialize and train the classifier
        self.classifier = self.train_classifier()
        
        # User session memory with enhanced reasoning
        self.user_memories = {}
        
        # Conversation history with context window
        self.conversation_history = []
        self.context_window = 5  # Keep last 5 interactions for context
        
        # Initialize TF-IDF vectorizer for semantic matching
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words=list(self.stop_words)
        )
        self.game_descriptions = [game['description'] for game in self.games_data]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.game_descriptions)
        
        # Cache for responses with TTL (Time To Live)
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Load data from data.py
        self.intent_patterns = intent_patterns
        self.synonyms = synonyms
        self.faqs = faqs
        self.challenges = challenges
        self.translations = translations
        self.reasoning_rules = reasoning_rules

    def detect_language(self, message):
        """Detect the language of the user's message"""
        # Check for explicit language requests first and prioritize them
        if any(phrase in message.lower() for phrase in ['in french', 'en français', 'translate to french', 'traduire en français']):
            return 'fr'
        if any(phrase in message.lower() for phrase in ['in english', 'en anglais', 'translate to english', 'traduire en anglais']):
            return 'en'
        
        # If no explicit language request, use keyword detection
        message_lower = message.lower()
        french_indicators = [
            'bonjour', 'salut', 'merci', 'jeu', 'prix', 'jouer', 'recommander',
            'quel', 'est', 'le', 'de', 'la', 'les', 'des', 'que', 'quoi', 'comment',
            'pourquoi', 'où', 'quand', 'qui', 'combien', 'coûte'
        ]
        english_indicators = [
            'hello', 'hi', 'thanks', 'game', 'price', 'play', 'recommend',
            'what', 'is', 'the', 'of', 'how', 'much', 'where', 'when', 'who',
            'why', 'which', 'cost'
        ]
        
        french_count = sum(1 for word in french_indicators if word in message_lower)
        english_count = sum(1 for word in english_indicators if word in message_lower)
        
        # Check for French characters
        french_chars = set('éèêëàâäôöûüçîïœæ')
        has_french_chars = any(char in french_chars for char in message)
        
        # Give extra weight to French characters
        if has_french_chars:
            french_count += 2
            
        # Check for French question patterns
        french_patterns = ["qu'est", "qu'est-ce", "est-ce", "qu'il", "est-il"]
        if any(pattern in message_lower for pattern in french_patterns):
            french_count += 2
            
        return 'fr' if french_count >= english_count else 'en'

    def handle_game_info(self, message, language='en'):
        """Handle requests for information about specific games"""
        message = message.lower()
        
        # Common game title variations
        game_variations = {
            'gta': 'Grand Theft Auto V',
            'gta v': 'Grand Theft Auto V',
            'gta 5': 'Grand Theft Auto V',
            'gtav': 'Grand Theft Auto V',
            'grand theft auto': 'Grand Theft Auto V',
            'minecraft': 'Minecraft',
            'mincraft': 'Minecraft',
            'mine craft': 'Minecraft',
            'fortnite': 'Fortnite'
        }
        
        # First try exact matches with variations
        found_game = None
        for variation, full_title in game_variations.items():
            if variation in message:
                # Find the game in games_data
                for game in self.games_data:
                    if game['title'] == full_title:
                        found_game = game
                        break
                if found_game:
                    break
        
        # If no match found through variations, try regular matching
        if not found_game:
            for game in self.games_data:
                if game['title'].lower() in message:
                    found_game = game
                    break
        
        # Special handling for Fortnite (since it's not in games_data)
        if 'fortnite' in message:
            if language == 'fr':
                return ("Fortnite est un jeu de type Battle Royale, développé par Epic Games. "
                       "C'est un jeu d'action multijoueur où 100 joueurs s'affrontent jusqu'à ce qu'il n'en reste plus qu'un. "
                       "Le jeu propose un système de construction unique et des événements saisonniers réguliers. "
                       "\n\nParmi ses caractéristiques principales, on trouve: Battle Royale, Construction, Multijoueur, "
                       "Événements en direct, et Crossplay. "
                       "\n\nLe jeu est disponible sur PC, PlayStation, Xbox, Nintendo Switch, et Mobile. "
                       "\n\nBonne nouvelle : le jeu est totalement gratuit à jouer, bien qu'il propose des achats "
                       "in-game pour des objets cosmétiques et des pass de combat!")
            else:
                return ("Fortnite is a Battle Royale game developed by Epic Games. "
                       "It's a multiplayer action game where 100 players fight until only one remains. "
                       "The game features a unique building system and regular seasonal events. "
                       "\n\nKey features include: Battle Royale, Building, Multiplayer, "
                       "Live Events, and Crossplay. "
                       "\n\nThe game is available on PC, PlayStation, Xbox, Nintendo Switch, and Mobile. "
                       "\n\nGreat news: the game is completely free to play, although it offers "
                       "in-game purchases for cosmetic items and battle passes!")
        
        if found_game:
            return self.format_game_info(found_game, language)
        
        # If no match found
        if language == 'fr':
            return "Je n'ai pas trouvé ce jeu. Pourriez-vous vérifier l'orthographe ou essayer un autre titre?"
        return "I couldn't find that game. Could you check the spelling or try another title?"

    def format_game_info(self, game, language='en'):
        """Format game information in a conversational way with language support"""
        if language == 'fr':
            response = f"{game['title']} est "
            if len(game['genre']) > 1:
                response += f"un jeu qui mélange les genres {' et '.join(game['genre'])}, "
            else:
                response += f"un jeu de {game['genre'][0]}, "
            response += f"développé par {game['publisher']}. "
            
            # Release and Rating
            response += f"Sorti le {game['release_date']}, "
            if game['rating'] >= 8:
                response += f"il a reçu d'excellentes critiques avec une note de {game['rating']}/10. "
            elif game['rating'] >= 6:
                response += f"il a reçu de bonnes critiques avec une note de {game['rating']}/10. "
            else:
                response += f"il a reçu une note de {game['rating']}/10. "
            
            # Description
            response += f"\n\n{game['description']} "
            
            # Features
            if 'tags' in game and game['tags']:
                response += f"\n\nParmi ses caractéristiques principales, on trouve: "
                response += f"{', '.join(game['tags'][:-1])}"
                if len(game['tags']) > 1:
                    response += f" et {game['tags'][-1]}. "
                else:
                    response += f"{game['tags'][0]}. "
            
            # Platforms
            response += f"\n\nLe jeu est disponible sur {', '.join(game['platforms'][:-1])}"
            if len(game['platforms']) > 1:
                response += f" et {game['platforms'][-1]}. "
            else:
                response += ". "
            
            # Price and Multiplayer
            if game['price'] == 0:
                response += "\nBonne nouvelle : le jeu est totalement gratuit"
            else:
                response += f"\nLe jeu est disponible au prix de {game['price']:.2f}€"
            
            if game['multiplayer']:
                response += " et propose un mode multijoueur pour jouer avec vos amis!"
            else:
                response += " et offre une expérience solo immersive."
        else:
            response = f"{game['title']} is "
            if len(game['genre']) > 1:
                response += f"a game that combines {' and '.join(game['genre'])}, "
            else:
                response += f"a {game['genre'][0]} game, "
            response += f"developed by {game['publisher']}. "
            
            # Release and Rating
            response += f"Released on {game['release_date']}, "
            if game['rating'] >= 8:
                response += f"it has received excellent reviews with a rating of {game['rating']}/10. "
            elif game['rating'] >= 6:
                response += f"it has received good reviews with a rating of {game['rating']}/10. "
            else:
                response += f"it has received a rating of {game['rating']}/10. "
            
            # Description
            response += f"\n\n{game['description']} "
            
            # Features
            if 'tags' in game and game['tags']:
                response += f"\n\nKey features include: "
                response += f"{', '.join(game['tags'][:-1])}"
                if len(game['tags']) > 1:
                    response += f" and {game['tags'][-1]}. "
                else:
                    response += f"{game['tags'][0]}. "
            
            # Platforms
            response += f"\n\nThe game is available on {', '.join(game['platforms'][:-1])}"
            if len(game['platforms']) > 1:
                response += f" and {game['platforms'][-1]}. "
            else:
                response += ". "
            
            # Price and Multiplayer
            if game['price'] == 0:
                response += "\nGreat news: the game is completely free"
            else:
                response += f"\nThe game is available for ${game['price']:.2f}"
            
            if game['multiplayer']:
                response += " and features multiplayer support to play with your friends!"
            else:
                response += " and offers an immersive single-player experience."
        
        return response

    def are_words_similar(self, word1, word2):
        """Check if two words are similar (allowing for minor misspellings)"""
        # Convert both words to lowercase
        word1 = word1.lower()
        word2 = word2.lower()
        
        # If either word is too short, require exact match
        if len(word1) <= 3 or len(word2) <= 3:
            return word1 == word2
        
        # If words are exactly the same, return True
        if word1 == word2:
            return True
        
        # Calculate Levenshtein distance
        distances = [[0 for _ in range(len(word2) + 1)] for _ in range(len(word1) + 1)]
        
        # Initialize first row and column
        for i in range(len(word1) + 1):
            distances[i][0] = i
        for j in range(len(word2) + 1):
            distances[0][j] = j
            
        # Fill the matrix
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i-1] == word2[j-1]:
                    distances[i][j] = distances[i-1][j-1]
                else:
                    distances[i][j] = min(distances[i-1][j], distances[i][j-1], distances[i-1][j-1]) + 1
        
        # Get the final distance
        distance = distances[len(word1)][len(word2)]
        
        # Calculate maximum allowed distance based on word length
        max_distance = min(len(word1), len(word2)) // 3
        
        # Return True if distance is within acceptable range
        return distance <= max_distance 

    def handle_game_recommendation(self, message, language='en'):
        """Handle game recommendation requests"""
        message = message.lower()
        
        # Extract preferences from message
        preferences = self.extract_preferences(message)
        
        # Find matching games
        matching_games = self.find_matching_games(preferences)
        
        if not matching_games:
            if language == 'fr':
                return "Je n'ai pas trouvé de jeux correspondant à vos critères. Pourriez-vous me donner plus de détails sur vos préférences?"
            return "I couldn't find any games matching your criteria. Could you tell me more about your preferences?"
        
        # Format response
        if language == 'fr':
            response = "Voici quelques jeux qui pourraient vous intéresser:\n\n"
            for game in matching_games[:3]:
                response += f"🎮 {game['title']}\n"
                response += f"📝 {game['description']}\n"
                response += f"💰 Prix: {game['price']}€\n"
                response += f"⭐ Note: {game['rating']}/10\n\n"
        else:
            response = "Here are some games you might enjoy:\n\n"
            for game in matching_games[:3]:
                response += f"🎮 {game['title']}\n"
                response += f"📝 {game['description']}\n"
                response += f"💰 Price: ${game['price']}\n"
                response += f"⭐ Rating: {game['rating']}/10\n\n"
        
        return response

    def extract_preferences(self, message):
        """Extract game preferences from user message"""
        preferences = {}
        message = message.lower()
        
        # Extract genre preferences
        genre_keywords = {
            'survival': ['survival', 'survive', 'survie'],
            'action': ['action', 'combat', 'fighting'],
            'rpg': ['rpg', 'role-playing', 'role playing', 'jeu de rôle'],
            'strategy': ['strategy', 'stratégie', 'tactical', 'tactique'],
            'sports': ['sports', 'sport', 'football', 'soccer', 'basketball'],
            'racing': ['racing', 'race', 'course', 'driving', 'conduite'],
            'horror': ['horror', 'horreur', 'scary', 'effrayant'],
            'adventure': ['adventure', 'aventure', 'exploration'],
            'puzzle': ['puzzle', 'puzzles', 'énigmes', 'réflexion'],
            'simulation': ['simulation', 'simulator', 'simulateur']
        }
        
        # Find mentioned genres
        mentioned_genres = []
        for genre, keywords in genre_keywords.items():
            if any(keyword in message for keyword in keywords):
                mentioned_genres.append(genre.title())
        
        if mentioned_genres:
            preferences['genre'] = mentioned_genres
        
        # Extract platform preferences
        platform_keywords = {
            'pc': ['pc', 'computer', 'desktop', 'windows', 'ordinateur'],
            'playstation': ['ps4', 'ps5', 'playstation', 'sony'],
            'xbox': ['xbox', 'microsoft'],
            'switch': ['switch', 'nintendo'],
            'mobile': ['mobile', 'phone', 'android', 'ios', 'iphone', 'smartphone']
        }
        
        mentioned_platforms = []
        for platform, keywords in platform_keywords.items():
            if any(keyword in message for keyword in keywords):
                mentioned_platforms.append(platform)
        
        if mentioned_platforms:
            preferences['platform'] = mentioned_platforms
        
        # Extract price range preferences
        if any(word in message for word in ['free', 'gratuit']):
            preferences['price_range'] = 'free'
        elif any(word in message for word in ['cheap', 'pas cher', 'bon marché']):
            preferences['price_range'] = 'cheap'
        elif any(word in message for word in ['expensive', 'cher', 'premium']):
            preferences['price_range'] = 'expensive'
        
        # Extract multiplayer preference
        if any(word in message for word in ['multiplayer', 'multi', 'coop', 'co-op', 'multijoueur']):
            preferences['multiplayer'] = True
        elif any(word in message for word in ['single player', 'solo', 'alone', 'seul']):
            preferences['multiplayer'] = False
        
        return preferences

    def find_matching_games(self, preferences):
        """Find games matching the given preferences"""
        matching_games = []
        
        for game in self.games_data:
            # Check if game matches all preferences
            matches = True
            
            # Check genre
            if 'genre' in preferences and preferences['genre']:
                game_genres = [g.lower() for g in game['genre']]
                if not any(g.lower() in game_genres for g in preferences['genre']):
                    matches = False
            
            # Check platform
            if 'platform' in preferences and preferences['platform']:
                if not any(p.lower() in [plat.lower() for plat in game['platforms']] 
                          for p in preferences['platform']):
                    matches = False
            
            # Check price range
            if 'price_range' in preferences:
                price = float(game['price'])
                if preferences['price_range'] == 'free' and price > 0:
                    matches = False
                elif preferences['price_range'] == 'cheap' and price > 20:
                    matches = False
                elif preferences['price_range'] == 'medium' and (price < 20 or price > 40):
                    matches = False
                elif preferences['price_range'] == 'expensive' and price < 40:
                    matches = False
            
            # Check multiplayer preference
            if 'multiplayer' in preferences:
                if preferences['multiplayer'] != game['multiplayer']:
                    matches = False
            
            # Add game if it matches all criteria
            if matches:
                matching_games.append(game)
        
        # Sort by rating
        matching_games.sort(key=lambda x: float(x['rating']), reverse=True)
        
        return matching_games

    def update_user_preferences(self, user_id, game_interaction):
        """Update user preferences based on their interactions with games"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'favorite_genres': {},
                'preferred_platforms': {},
                'price_sensitivity': [],
                'multiplayer_preference': 0,
                'favorite_tags': {},
                'interaction_history': []
            }
        
        prefs = self.user_preferences[user_id]
        
        # Update genre preferences
        for genre in game_interaction['genre']:
            prefs['favorite_genres'][genre] = prefs['favorite_genres'].get(genre, 0) + 1
        
        # Update platform preferences
        for platform in game_interaction['platforms']:
            prefs['preferred_platforms'][platform] = prefs['preferred_platforms'].get(platform, 0) + 1
        
        # Update price sensitivity
        prefs['price_sensitivity'].append(game_interaction['price'])
        if len(prefs['price_sensitivity']) > 10:  # Keep only last 10 interactions
            prefs['price_sensitivity'].pop(0)
        
        # Update multiplayer preference
        prefs['multiplayer_preference'] += 1 if game_interaction['multiplayer'] else -1
        
        # Update tag preferences
        if 'tags' in game_interaction:
            for tag in game_interaction['tags']:
                prefs['favorite_tags'][tag] = prefs['favorite_tags'].get(tag, 0) + 1
        
        # Add to interaction history
        prefs['interaction_history'].append({
            'game_id': game_interaction['title'],
            'timestamp': datetime.now().isoformat(),
            'interaction_type': game_interaction.get('interaction_type', 'view')
        })
        
        # Trim history if too long
        if len(prefs['interaction_history']) > 50:
            prefs['interaction_history'] = prefs['interaction_history'][-50:] 

    def handle_faq(self, message, language='en'):
        """Handle frequently asked questions"""
        message = message.lower()
        best_match = None
        highest_score = 0
        
        # Try to find the best matching FAQ
        for category, faqs in self.faqs.items():
            for question, answers in faqs.items():
                # Calculate similarity score
                score = self.calculate_similarity(message, question.lower())
                if score > highest_score and score > 0.6:  # Threshold for matching
                    highest_score = score
                    best_match = (category, question, answers)
        
        if best_match:
            category, question, answers = best_match
            return answers[language] if language in answers else answers['en']
        
        # If no direct match found, try keyword matching
        keywords = self.extract_keywords(message)
        for category, faqs in self.faqs.items():
            for question, answers in faqs.items():
                if any(keyword in question.lower() for keyword in keywords):
                    return answers[language] if language in answers else answers['en']
        
        # No match found
        if language == 'fr':
            return "Je suis désolé, je ne comprends pas complètement votre question. Pourriez-vous la reformuler ou être plus spécifique?"
        return "I'm sorry, I don't fully understand your question. Could you rephrase it or be more specific?"

    def handle_technical_support(self, message, language='en'):
        """Handle technical support inquiries"""
        # Extract the problem type and details
        problem_info = self.extract_technical_problem(message)
        
        # Get relevant support information
        support_info = self.get_support_info(problem_info, language)
        
        # Format and return the response
        if language == 'fr':
            response = "Voici quelques suggestions pour résoudre votre problème:\n\n"
        else:
            response = "Here are some suggestions to resolve your issue:\n\n"
        
        response += support_info
        return response

    def extract_technical_problem(self, message):
        """Extract technical problem information from the message"""
        problem_info = {
            'type': None,
            'game': None,
            'platform': None,
            'symptoms': set(),
            'error_code': None
        }
        
        message = message.lower()
        
        # Extract error codes (common formats)
        error_patterns = [
            r'error[: ]([A-Za-z0-9-]+)',
            r'code[: ]([A-Za-z0-9-]+)',
            r'#([A-Za-z0-9-]+)'
        ]
        for pattern in error_patterns:
            match = re.search(pattern, message)
            if match:
                problem_info['error_code'] = match.group(1)
                break
        
        # Extract game title if mentioned
        for game in self.games_data:
            if game['title'].lower() in message:
                problem_info['game'] = game['title']
                break
        
        # Extract platform if mentioned
        for platform in self.game_vocabulary['platforms']:
            if platform.lower() in message:
                problem_info['platform'] = platform
                break
        
        # Identify problem type and symptoms
        problem_types = {
            'performance': ['lag', 'slow', 'fps', 'freeze', 'stuttering'],
            'connection': ['disconnect', 'network', 'online', 'connection', 'server'],
            'graphics': ['graphics', 'display', 'screen', 'resolution', 'visual'],
            'audio': ['sound', 'audio', 'volume', 'music', 'voice'],
            'controls': ['controller', 'keyboard', 'mouse', 'input', 'buttons'],
            'installation': ['install', 'download', 'update', 'patch', 'version'],
            'account': ['login', 'account', 'password', 'profile', 'save'],
            'crash': ['crash', 'close', 'exit', 'stop working', 'not responding']
        }
        
        for ptype, keywords in problem_types.items():
            if any(keyword in message for keyword in keywords):
                problem_info['type'] = ptype
                problem_info['symptoms'].update(kw for kw in keywords if kw in message)
        
        return problem_info

    def get_support_info(self, problem_info, language='en'):
        """Get support information based on the problem information"""
        # Initialize response based on language
        if language == 'fr':
            response = ""
            if problem_info['game']:
                response += f"Pour {problem_info['game']}:\n"
            if problem_info['platform']:
                response += f"Sur {problem_info['platform']}:\n"
        else:
            response = ""
            if problem_info['game']:
                response += f"For {problem_info['game']}:\n"
            if problem_info['platform']:
                response += f"On {problem_info['platform']}:\n"
        
        # Add specific troubleshooting steps based on problem type
        steps = self.get_troubleshooting_steps(problem_info, language)
        response += steps
        
        # Add error code information if available
        if problem_info['error_code']:
            error_info = self.get_error_code_info(problem_info['error_code'], language)
            response += f"\n\n{error_info}"
        
        # Add general advice
        if language == 'fr':
            response += "\n\nSi ces étapes ne résolvent pas votre problème:"
            response += "\n1. Vérifiez que votre système répond aux exigences minimales"
            response += "\n2. Mettez à jour vos pilotes graphiques"
            response += "\n3. Vérifiez les forums de la communauté pour des solutions"
            response += "\n4. Contactez le support technique"
        else:
            response += "\n\nIf these steps don't resolve your issue:"
            response += "\n1. Verify your system meets the minimum requirements"
            response += "\n2. Update your graphics drivers"
            response += "\n3. Check community forums for solutions"
            response += "\n4. Contact technical support"
        
        return response

    def get_troubleshooting_steps(self, problem_info, language='en'):
        """Get specific troubleshooting steps based on problem type"""
        steps = []
        
        if problem_info['type'] == 'performance':
            if language == 'fr':
                steps = [
                    "1. Fermez les applications en arrière-plan",
                    "2. Réduisez les paramètres graphiques",
                    "3. Vérifiez la température de votre système",
                    "4. Défragmentez vos disques durs",
                    "5. Mettez à jour vos pilotes"
                ]
            else:
                steps = [
                    "1. Close background applications",
                    "2. Lower graphics settings",
                    "3. Check system temperature",
                    "4. Defragment your hard drives",
                    "5. Update your drivers"
                ]
        
        elif problem_info['type'] == 'connection':
            if language == 'fr':
                steps = [
                    "1. Vérifiez votre connexion internet",
                    "2. Redémarrez votre routeur",
                    "3. Vérifiez les pare-feu",
                    "4. Utilisez une connexion filaire si possible",
                    "5. Vérifiez l'état des serveurs du jeu"
                ]
            else:
                steps = [
                    "1. Check your internet connection",
                    "2. Restart your router",
                    "3. Check firewall settings",
                    "4. Use a wired connection if possible",
                    "5. Verify game server status"
                ]
        
        # Add more problem types and their steps...
        
        return "\n".join(steps)

    def get_error_code_info(self, error_code, language='en'):
        """Get information about specific error codes"""
        # Implement error code lookup logic here
        # This could be expanded with a comprehensive error code database
        if language == 'fr':
            return f"Code d'erreur {error_code}: Veuillez consulter la base de connaissances ou contacter le support technique."
        return f"Error code {error_code}: Please consult the knowledge base or contact technical support."

    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts using TF-IDF and cosine similarity"""
        # Vectorize the texts
        vectors = self.tfidf_vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        return similarity

    def extract_keywords(self, text):
        """Extract important keywords from text"""
        # Tokenize and lemmatize
        tokens = word_tokenize(text.lower())
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove stopwords and short words
        keywords = [word for word in lemmatized 
                   if word not in self.stop_words and len(word) > 2]
        
        return keywords 

    def handle_community(self, message, language='en'):
        """Handle community-related inquiries"""
        message = message.lower()
        
        # Check for challenge-related queries
        if any(word in message for word in ['challenge', 'défi', 'achievement', 'mission']):
            return self.handle_challenges(message, language)
        
        # Check for tournament-related queries
        if any(word in message for word in ['tournament', 'tournoi', 'competition', 'compétition']):
            return self.handle_tournaments(message, language)
        
        # Check for forum-related queries
        if any(word in message for word in ['forum', 'discussion', 'community', 'communauté']):
            return self.handle_forum_info(message, language)
        
        # Default community information
        if language == 'fr':
            return ("Notre communauté de joueurs est très active! Vous pouvez:\n"
                   "1. Participer à nos défis hebdomadaires\n"
                   "2. Rejoindre des tournois\n"
                   "3. Discuter sur nos forums\n"
                   "4. Trouver des partenaires de jeu\n"
                   "Que souhaitez-vous faire?")
        return ("Our gaming community is very active! You can:\n"
                "1. Participate in weekly challenges\n"
                "2. Join tournaments\n"
                "3. Discuss on our forums\n"
                "4. Find gaming partners\n"
                "What would you like to do?")

    def handle_challenges(self, message, language='en'):
        """Handle challenge-related inquiries"""
        message = message.lower()
        
        # Check for language override first
        if 'in english' in message or 'en anglais' in message:
            language = 'en'
        elif 'in french' in message or 'en français' in message:
            language = 'fr'
        else:
            # If no explicit override, check message language
            french_indicators = ['défi', 'donnez-moi', 'donner', 'défi']
            english_indicators = ['challenge', 'give me', 'get', 'want']
            
            french_count = sum(1 for word in french_indicators if word in message)
            english_count = sum(1 for word in english_indicators if word in message)
            
            # Override language based on message content
            if french_count > english_count:
                language = 'fr'
            elif english_count > french_count:
                language = 'en'
            # else: keep the provided language parameter
        
        # Game variations mapping
        game_variations = {
            'minecraft': ['minecraft', 'mincraft', 'mine craft'],
            'fortnite': ['fortnite', 'fortnight', 'fort nite'],
            'gta': ['gta', 'grand theft auto', 'gta v', 'gta 5'],
            'cod': ['call of duty', 'cod', 'modern warfare', 'warzone']
        }
        
        # Extract game name if specified
        game_name = None
        game_genre = None
        
        # Check for game variations first
        for game, variations in game_variations.items():
            if any(var in message for var in variations):
                game_name = game.title()  # Capitalize first letter
                break
        
        # If no match found through variations, try direct game matching
        if not game_name:
            for game in self.games_data:
                if game['title'].lower() in message:
                    game_name = game['title']
                    game_genre = game['genre'][0]  # Use the primary genre
                    break
        
        # Get the challenge based on game or genre
        challenge = None
        
        # If we found a specific game with challenges
        if game_name and game_name.lower() in [g.lower() for g in self.challenges['game_specific'].keys()]:
            challenges = self.challenges['game_specific'][game_name]
            challenge = random.choice(challenges)
        
        # Special handling for game genres
        elif game_name:
            # Map games to their genres
            genre_mapping = {
                'Minecraft': 'Survival',
                'Fortnite': 'Battle Royale',
                'GTA': 'Action',
                'COD': 'FPS'
            }
            
            if game_name in genre_mapping:
                genre = genre_mapping[game_name]
                if genre in self.challenges['genre_specific']:
                    challenges = self.challenges['genre_specific'][genre]
                    challenge = random.choice(challenges)
        
        # If no specific game or genre challenge found, get a general challenge
        if not challenge:
            challenge = random.choice(self.challenges['general'])
        
        # Format the response
        if language == 'fr':
            response = "🎯 Voici votre défi:\n\n"
            response += f"📋 {challenge['name']['fr']}\n"
            response += f"ℹ️ {challenge['description']['fr']}\n"
            if 'difficulty' in challenge:
                response += f"⭐ Difficulté: {challenge['difficulty']}\n"
            response += f"🏆 Récompense: {challenge['reward']['fr']}"
        else:
            response = "🎯 Here's your challenge:\n\n"
            response += f"📋 {challenge['name']['en']}\n"
            response += f"ℹ️ {challenge['description']['en']}\n"
            if 'difficulty' in challenge:
                response += f"⭐ Difficulty: {challenge['difficulty']}\n"
            response += f"🏆 Reward: {challenge['reward']['en']}"
        
        return response

    def handle_tournaments(self, message, language='en'):
        """Handle tournament-related inquiries"""
        tournaments = self.get_tournament_info()
        
        if language == 'fr':
            response = "Tournois à venir:\n\n"
            for tournament in tournaments:
                response += f"🏆 {tournament['name']}\n"
                response += f"🎮 Jeu: {tournament['game']}\n"
                response += f"📅 Date: {tournament['date']}\n"
                response += f"🏅 Prix: {tournament['prize']}\n"
                response += f"👥 Places disponibles: {tournament['slots_available']}\n\n"
        else:
            response = "Upcoming tournaments:\n\n"
            for tournament in tournaments:
                response += f"🏆 {tournament['name']}\n"
                response += f"🎮 Game: {tournament['game']}\n"
                response += f"📅 Date: {tournament['date']}\n"
                response += f"🏅 Prize: {tournament['prize']}\n"
                response += f"👥 Available slots: {tournament['slots_available']}\n\n"
        
        return response

    def handle_forum_info(self, message, language='en'):
        """Handle forum-related inquiries"""
        # Extract any specific topics of interest
        topics = self.extract_forum_topics(message)
        
        if language == 'fr':
            if topics:
                response = "Voici les discussions actives sur ces sujets:\n\n"
                for topic in topics:
                    response += f"📌 {topic['title']}\n"
                    response += f"💬 {topic['replies']} réponses\n"
                    response += f"👁️ {topic['views']} vues\n"
                    response += f"⏰ Dernière activité: {topic['last_activity']}\n\n"
            else:
                response = ("Nos forums sont organisés en plusieurs sections:\n"
                          "🎮 Discussion générale\n"
                          "🔧 Support technique\n"
                          "🤝 Recherche de partenaires\n"
                          "📰 Actualités et annonces\n"
                          "💡 Suggestions et feedback\n\n"
                          "Quel sujet vous intéresse?")
        else:
            if topics:
                response = "Here are the active discussions on those topics:\n\n"
                for topic in topics:
                    response += f"📌 {topic['title']}\n"
                    response += f"💬 {topic['replies']} replies\n"
                    response += f"👁️ {topic['views']} views\n"
                    response += f"⏰ Last activity: {topic['last_activity']}\n\n"
            else:
                response = ("Our forums are organized into several sections:\n"
                          "🎮 General Discussion\n"
                          "🔧 Technical Support\n"
                          "🤝 Looking for Group\n"
                          "📰 News and Announcements\n"
                          "💡 Suggestions and Feedback\n\n"
                          "What topic interests you?")
        
        return response

    def extract_forum_topics(self, message):
        """Extract forum topics from message"""
        # This is a placeholder implementation
        # In a real system, this would query a forum database
        topics = []
        
        # Example topic data
        example_topics = [
            {
                'title': "Weekly Gaming Challenge Discussion",
                'replies': 45,
                'views': 1200,
                'last_activity': "2 hours ago"
            },
            {
                'title': "Looking for Squad Members",
                'replies': 23,
                'views': 500,
                'last_activity': "5 minutes ago"
            }
        ]
        
        # Add logic to match topics based on message content
        keywords = self.extract_keywords(message)
        for topic in example_topics:
            if any(keyword in topic['title'].lower() for keyword in keywords):
                topics.append(topic)
        
        return topics

    def get_tournament_info(self):
        """Get information about current tournaments"""
        # This is a placeholder implementation
        # In a real system, this would query a tournament database
        return [
            {
                'name': "Weekend Warriors Cup",
                'game': "Battle Arena Pro",
                'date': "Next Saturday, 2 PM UTC",
                'prize': "$1000 Prize Pool",
                'slots_available': 16
            },
            {
                'name': "Casual Gaming League",
                'game': "Racing Evolution 2024",
                'date': "Every Sunday, 3 PM UTC",
                'prize': "Premium Gaming Gear",
                'slots_available': 24
            }
        ]

    def load_training_data(self):
        """Load or create training data for intent classification"""
        training_data = {
            'game_recommendation': {
                'en': [
                    "Can you recommend a game?",
                    "What games should I play?",
                    "I'm looking for a new game",
                    "Suggest some games for me",
                    "What's a good game to try?",
                    "Which games are popular right now?",
                    "I need game recommendations",
                    "What games are similar to...",
                    "Best games in this genre?",
                    "What should I play next?"
                ],
                'fr': [
                    "Pouvez-vous recommander un jeu?",
                    "Quels jeux devrais-je jouer?",
                    "Je cherche un nouveau jeu",
                    "Suggérez-moi des jeux",
                    "Quel est un bon jeu à essayer?",
                    "Quels jeux sont populaires en ce moment?",
                    "J'ai besoin de recommandations de jeux",
                    "Quels jeux sont similaires à...",
                    "Meilleurs jeux dans ce genre?",
                    "Que devrais-je jouer ensuite?"
                ]
            },
            'game_info': {
                'en': [
                    "Tell me about this game",
                    "What is this game about?",
                    "Game details please",
                    "Information about...",
                    "What's the gameplay like?",
                    "Is this game multiplayer?",
                    "What are the features of...",
                    "How much does this game cost?",
                    "When was this game released?",
                    "Who developed this game?"
                ],
                'fr': [
                    "Parlez-moi de ce jeu",
                    "De quoi parle ce jeu?",
                    "Détails du jeu s'il vous plaît",
                    "Informations sur...",
                    "Comment est le gameplay?",
                    "Est-ce un jeu multijoueur?",
                    "Quelles sont les caractéristiques de...",
                    "Combien coûte ce jeu?",
                    "Quand ce jeu est-il sorti?",
                    "Qui a développé ce jeu?"
                ]
            },
            'technical_support': {
                'en': [
                    "Game is not working",
                    "How to fix...",
                    "Getting an error",
                    "Game keeps crashing",
                    "Performance issues",
                    "Can't connect to server",
                    "Game won't start",
                    "How to update...",
                    "Installation problems",
                    "Need help with..."
                ],
                'fr': [
                    "Le jeu ne fonctionne pas",
                    "Comment réparer...",
                    "J'ai une erreur",
                    "Le jeu plante",
                    "Problèmes de performance",
                    "Impossible de se connecter au serveur",
                    "Le jeu ne démarre pas",
                    "Comment mettre à jour...",
                    "Problèmes d'installation",
                    "Besoin d'aide avec..."
                ]
            },
            'price': {
                'en': [
                    "How much is...",
                    "Price of the game",
                    "Is it free?",
                    "Cost of...",
                    "Game price",
                    "Any discounts?",
                    "Is it on sale?",
                    "Cheapest price for...",
                    "Where to buy...",
                    "Payment options",
                    "What's the price of...",
                    "Tell me the price of...",
                    "How much does it cost",
                    "Price in english",
                    "Price in french",
                    "Show me the price"
                ],
                'fr': [
                    "Combien coûte...",
                    "Prix du jeu",
                    "Est-ce gratuit?",
                    "Coût de...",
                    "Prix du jeu",
                    "Y a-t-il des réductions?",
                    "Est-ce en solde?",
                    "Prix le plus bas pour...",
                    "Où acheter...",
                    "Options de paiement",
                    "Quel est le prix de...",
                    "Dites-moi le prix de...",
                    "Combien ça coûte",
                    "Prix en français",
                    "Prix en anglais",
                    "Montrez-moi le prix"
                ]
            },
            'community': {
                'en': [
                    "How to join tournaments?",
                    "Looking for players",
                    "Community events",
                    "Active challenges",
                    "Forum discussions",
                    "Find gaming partners",
                    "Team recruitment",
                    "Community guidelines",
                    "Player rankings",
                    "Social features"
                ],
                'fr': [
                    "Comment rejoindre les tournois?",
                    "Recherche de joueurs",
                    "Événements communautaires",
                    "Défis actifs",
                    "Discussions forum",
                    "Trouver des partenaires",
                    "Recrutement d'équipe",
                    "Règles communautaires",
                    "Classement des joueurs",
                    "Fonctionnalités sociales"
                ]
            }
        }
        return training_data

    def train_classifier(self):
        """Train the intent classifier"""
        # Prepare training data
        X = []  # Features (text)
        y = []  # Labels (intents)
        
        for intent, language_data in self.training_data.items():
            for language, examples in language_data.items():
                for example in examples:
                    X.append(example.lower())
                    y.append(intent)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_tfidf = vectorizer.fit_transform(X)
        
        # Train classifier
        classifier = MultinomialNB()
        classifier.fit(X_tfidf, y)
        
        # Create pipeline for easy prediction
        return Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ]) 

    def handle_price(self, message, language='en'):
        """Handle price-related inquiries"""
        message = message.lower()
        
        # Common game title patterns and their variations
        common_patterns = {
            'god of war': ['gow', 'godofwar', 'god war'],
            'fortnite': ['fort', 'fortnight', 'forttnite', 'fortnit'],
            'minecraft': ['mine craft', 'mincraft', 'mcraft'],
            'counter strike': ['cs', 'csgo', 'cs:go', 'counter-strike'],
            'grand theft auto': ['gta', 'grandtheftauto'],
            'call of duty': ['cod', 'callofduty', 'call duty'],
            'league of legends': ['lol', 'leagueoflegends', 'league'],
            'playerunknown\'s battlegrounds': ['pubg', 'battlegrounds'],
            'world of warcraft': ['wow', 'worldofwarcraft']
        }
        
        # Special handling for known free-to-play games
        free_to_play_games = {
            'fortnite': {
                'en': "Fortnite is a free-to-play game! While it offers in-game purchases for cosmetic items and battle passes, the core game is completely free to download and play.",
                'fr': "Fortnite est un jeu gratuit! Bien qu'il propose des achats in-game pour des objets cosmétiques et des pass de combat, le jeu de base est totalement gratuit à télécharger et à jouer."
            }
        }
        
        # Extract potential game title from price query
        # Common price-related words to remove
        price_words = {'price', 'cost', 'how', 'much', 'is', 'the', 'of', 'for', 'prix', 'coût', 'combien', 'coûte', 'le', 'de', 'du', 'give', 'me', 'tell', 'show'}
        words = message.split()
        # Remove price-related words to isolate the game title
        game_title_words = [word for word in words if word.lower() not in price_words]
        potential_title = ' '.join(game_title_words).strip()
        
        # Create variations of the potential title
        message_variations = {
            potential_title,  # Original extracted title
            potential_title.replace(" ", ""),  # No spaces
            potential_title.replace("-", " "),  # Replace hyphens with spaces
            potential_title.replace("_", " "),  # Replace underscores with spaces
        }
        
        # Add common variations based on patterns
        for base_title, variations in common_patterns.items():
            if any(var in potential_title for var in [base_title] + variations):
                message_variations.update(variations)
                message_variations.add(base_title)
        
        # Remove empty strings and normalize
        message_variations = {v.lower() for v in message_variations if v}
        
        # Check for free-to-play games first
        for game, responses in free_to_play_games.items():
            if any(game in variation or variation in game for variation in message_variations):
                return responses[language]
        
        # Try to find the game in our database
        best_match = None
        min_distance = float('inf')
        
        for game in self.games_data:
            game_title = game['title'].lower()
            
            # Create variations of the game title
            game_variations = {
                game_title,  # Original title
                game_title.replace(" ", ""),  # No spaces
                game_title.replace("-", " "),  # Replace hyphens with spaces
                game_title.replace("_", " ")  # Replace underscores with spaces
            }
            
            # Add common variations for this game
            for base_title, variations in common_patterns.items():
                if base_title in game_title or any(var in game_title for var in variations):
                    game_variations.update(variations)
                    game_variations.add(base_title)
            
            # Try exact matches first (including partial matches)
            if any(game_var in msg_var or msg_var in game_var 
                   for game_var in game_variations
                   for msg_var in message_variations):
                best_match = game
                break
            
            # Try partial word matches
            game_words = set(game_title.split())
            for variation in message_variations:
                variation_words = set(variation.split())
                if variation_words.issubset(game_words) or game_words.issubset(variation_words):
                    best_match = game
                    break
            
            if best_match:
                break
            
            # If still no match, try fuzzy matching
            for msg_var in message_variations:
                if len(msg_var) < 3:  # Skip very short variations
                    continue
                for game_var in game_variations:
                    distance = self.calculate_levenshtein_distance(msg_var, game_var)
                    normalized_distance = distance / max(len(msg_var), len(game_var))
                    threshold = 0.4 if len(msg_var) > 5 else 0.3
                    if normalized_distance < min_distance and normalized_distance <= threshold:
                        min_distance = normalized_distance
                        best_match = game
        
        # If we found a match
        if best_match:
            if language == 'fr':
                if best_match['price'] == 0:
                    return f"{best_match['title']} est gratuit!"
                else:
                    return f"{best_match['title']} coûte {best_match['price']:.2f}€."
            else:
                if best_match['price'] == 0:
                    return f"{best_match['title']} is free to play!"
                else:
                    return f"{best_match['title']} costs ${best_match['price']:.2f}."
        
        # If no specific game is found
        if language == 'fr':
            return "Je n'ai pas trouvé le jeu dont vous parlez. Pourriez-vous vérifier l'orthographe du titre?"
        return "I couldn't find the game you're asking about. Could you check the spelling of the title?"

    def calculate_levenshtein_distance(self, s1, s2):
        """Calculate the Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.calculate_levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def handle_greeting(self, message, language='en'):
        """Handle greeting messages"""
        if language == 'fr':
            return "Bonjour! Je suis votre assistant IA et je peux vous aider à trouver des jeux, répondre à vos questions, fournir du support, ou même suggérer des défis amusants. Comment puis-je vous aider aujourd'hui?"
        return "Hello! I'm your AI assistant and I can help you find games, answer questions, provide support, or even suggest fun challenges. How can I help you today?"

    def process_message(self, message):
        """Process user message and return appropriate response"""
        # Store original message for history
        original_message = message
        message_lower = message.lower()
        
        # Detect language
        language = self.detect_language(message)
        
        # First, check for FAQ-related queries
        faq_patterns = [
            'how to', 'comment', 'what is', 'qu\'est-ce que', 'help with', 'aide avec',
            'how do i', 'how can i', 'comment puis-je', 'comment faire pour'
        ]
        
        # Account-related patterns
        account_patterns = [
            'create account', 'créer compte', 'delete account', 'supprimer compte',
            'change password', 'changer mot de passe', 'update email', 'mettre à jour email',
            'payment method', 'méthode de paiement', 'refund', 'remboursement',
            'subscription', 'abonnement', 'download', 'télécharger', 'install', 'installer',
            'create', 'créer', 'sign up', "s'inscrire", 'register', 'inscription'
        ]
        
        # Navigation patterns
        navigation_patterns = [
            'where is', 'où est', 'find', 'trouver', 'access', 'accéder',
            'go to', 'aller à', 'get to', 'navigate to', 'naviguer vers'
        ]
        
        # If it's a FAQ about account/settings/etc.
        if (any(pattern in message_lower for pattern in faq_patterns) and 
            any(pattern in message_lower for pattern in account_patterns)) or \
           any(pattern in message_lower for pattern in ['create account', 'créer compte', 'sign up', "s'inscrire"]):
            return self.handle_platform_faq(message, language)
        
        # If it's a navigation query
        if any(pattern in message_lower for pattern in navigation_patterns) or \
           any(section in message_lower for section in ['profile', 'profil', 'chat', 'store', 'boutique', 'forums', 'library', 'bibliothèque']):
            return self.handle_navigation(message, language)
        
        # Check for challenge-related queries
        if any(word in message_lower for word in ['challenge', 'défi', 'achievement', 'mission']):
            return self.handle_challenges(message, language)
        
        # Check for game-specific queries
        for game in self.games_data:
            if game['title'].lower() in message_lower:
                return self.handle_game_info(message, language)
        
        # Special handling for Fortnite
        if 'fortnite' in message_lower:
            return self.handle_game_info(message, language)
        
        # If no specific intent is detected, provide a general help message
        if language == 'fr':
            return ("Je peux vous aider à naviguer sur la plateforme, trouver des jeux, "
                   "ou répondre à vos questions. Que souhaitez-vous faire?")
        return ("I can help you navigate the platform, find games, "
               "or answer your questions. What would you like to do?")

    def handle_navigation(self, message, language='en'):
        """Handle navigation-related inquiries about the platform"""
        message = message.lower()
        
        # Extract the section the user is looking for
        sections = {
            'profile': ['profile', 'profil', 'account', 'compte'],
            'chat': ['chat', 'message', 'messages', 'messaging'],
            'store': ['store', 'boutique', 'shop', 'magasin'],
            'forums': ['forum', 'forums', 'discussion'],
            'browse': ['browse', 'parcourir', 'discover', 'découvrir'],
            'library': ['library', 'bibliothèque', 'games', 'jeux'],
            'news': ['news', 'actualités', 'updates', 'mises à jour']
        }
        
        # Find which section the user is asking about
        target_section = None
        for section, keywords in sections.items():
            if any(keyword in message for keyword in keywords):
                target_section = section
                break
        
        if target_section and target_section in self.navigation_guides['platform']:
            section_info = self.navigation_guides['platform'][target_section][language]
            if language == 'fr':
                return f"📍 Pour accéder à cette section : {section_info['path']}\n\n💡 {section_info['description']}"
            return f"📍 To access this section: {section_info['path']}\n\n💡 {section_info['description']}"
        
        # If no specific section is found, provide general navigation help
        if language == 'fr':
            return ("Je peux vous aider à naviguer sur la plateforme. Voici les sections principales :\n"
                   "👤 Profil - Gérez votre compte\n"
                   "💬 Messages - Chattez avec d'autres joueurs\n"
                   "🎮 Boutique - Parcourez et achetez des jeux\n"
                   "📢 Forums - Participez aux discussions\n"
                   "📚 Bibliothèque - Accédez à vos jeux\n"
                   "📰 Actualités - Restez informé\n\n"
                   "Quelle section vous intéresse ?")
        return ("I can help you navigate the platform. Here are the main sections:\n"
                "👤 Profile - Manage your account\n"
                "💬 Messages - Chat with other players\n"
                "🎮 Store - Browse and buy games\n"
                "📢 Forums - Join discussions\n"
                "📚 Library - Access your games\n"
                "📰 News - Stay updated\n\n"
                "Which section are you interested in?")

    def handle_platform_faq(self, message, language='en'):
        """Handle platform-related FAQs"""
        message = message.lower()
        
        # Define categories and their keywords
        categories = {
            'account': ['account', 'compte', 'password', 'mot de passe', 'email', 'delete', 'supprimer', 'create', 'créer'],
            'payment': ['payment', 'paiement', 'refund', 'remboursement', 'subscription', 'abonnement'],
            'technical': ['download', 'télécharger', 'install', 'installer', 'update', 'mise à jour']
        }
        
        # Find the category
        category = None
        for cat, keywords in categories.items():
            if any(keyword in message for keyword in keywords):
                category = cat
                break
        
        if not category:
            # If no specific category is found, provide general help
            if language == 'fr':
                return ("Je peux vous aider avec :\n"
                       "👤 Compte - création, suppression, mot de passe\n"
                       "💳 Paiement - méthodes, remboursements, abonnements\n"
                       "🔧 Technique - téléchargement, installation, mises à jour\n\n"
                       "Quelle catégorie vous intéresse ?")
            return ("I can help you with:\n"
                   "👤 Account - creation, deletion, password\n"
                   "💳 Payment - methods, refunds, subscriptions\n"
                   "🔧 Technical - downloads, installation, updates\n\n"
                   "Which category are you interested in?")
        
        # Find the specific topic within the category
        topics = {
            'account': {
                'create': ['create', 'créer', 'sign up', "s'inscrire"],
                'delete': ['delete', 'supprimer', 'remove', 'effacer'],
                'password': ['password', 'mot de passe'],
                'email': ['email', 'e-mail', 'mail']
            },
            'payment': {
                'methods': ['method', 'méthode', 'payment', 'paiement', 'pay', 'payer'],
                'refund': ['refund', 'remboursement', 'return', 'retour'],
                'subscription': ['subscription', 'abonnement', 'subscribe', "s'abonner"]
            },
            'technical': {
                'download': ['download', 'télécharger', 'get game', 'obtenir jeu'],
                'install': ['install', 'installer', 'setup', 'configurer'],
                'update': ['update', 'mise à jour', 'patch', 'upgrade']
            }
        }
        
        topic = None
        for t, keywords in topics[category].items():
            if any(keyword in message for keyword in keywords):
                topic = t
                break
        
        # If no specific topic is found, use the first topic in the category
        if not topic:
            topic = list(topics[category].keys())[0]
        
        # Get the FAQ response
        try:
            return self.platform_faqs[language][category][topic]
        except KeyError:
            # Fallback to English if the language or category/topic combination doesn't exist
            return self.platform_faqs['en'][category][topic]

    def respond(self, session_id, message):
        """Generate a response to the user's message"""
        # Store original message
        original_message = message
        message_lower = message.lower()
        
        # Check for language override first
        if 'in english' in message_lower or 'en anglais' in message_lower:
            language = 'en'
        elif 'in french' in message_lower or 'en français' in message_lower:
            language = 'fr'
        else:
            # Detect language if no explicit override
            language = self.detect_language(message)
        
        # First, check for navigation-related queries
        navigation_patterns = [
            'how to access', 'how do i access', 'where is', 'comment accéder',
            'où est', 'how to find', 'comment trouver', 'access', 'accéder',
            'profile', 'profil', 'chat', 'messages', 'store', 'boutique',
            'forums', 'library', 'bibliothèque', 'news', 'actualités'
        ]
        
        # If it's a navigation query, handle it first
        if any(pattern in message_lower for pattern in navigation_patterns):
            return self.handle_navigation(message, language)
        
        # Check for challenges
        if any(word in message_lower for word in ['challenge', 'défi', 'achievement', 'mission']):
            return self.handle_challenges(message, language)
        
        # Check for game recommendations
        if any(word in message_lower for word in ['recommend', 'suggest', 'suggérer', 'recommander', 'conseiller']):
            return self.handle_game_recommendation(message, language)
        
        # Check for greetings
        if re.match(self.intent_patterns['greeting'], message_lower):
            return self.handle_greeting(message, language)
        
        # Check cache for identical query
        cache_key = f"{message}_{language}"
        if cache_key in self.response_cache:
            cache_time, response = self.response_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return response
        
        # Initialize or get user memory
        if session_id not in self.user_memories:
            self.user_memories[session_id] = {
                'preferences': {},
                'last_context': None,
                'interaction_count': 0
            }
        
        # Update conversation history
        self.conversation_history.append({
            'user': original_message,
            'timestamp': time.time()
        })
        if len(self.conversation_history) > self.context_window:
            self.conversation_history.pop(0)
        
        # Process the message using process_message
        response = self.process_message(message)
        
        # Update cache
        self.response_cache[cache_key] = (time.time(), response)
        
        # Update user interaction count
        self.user_memories[session_id]['interaction_count'] += 1
        
        return response
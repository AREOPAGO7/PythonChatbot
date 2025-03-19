# Game data and related information for the gaming chatbot

# Game catalog
games_data = [
    {
        'title': 'Elden Ring',
        'genre': ['Action RPG', 'Open World'],
        'platforms': ['PC', 'PlayStation 5', 'Xbox Series X/S'],
        'price': 59.99,
        'release_date': '2022-02-25',
        'publisher': 'Bandai Namco',
        'description': 'An action RPG developed by FromSoftware and published by Bandai Namco Entertainment, in collaboration with fantasy novelist George R. R. Martin.',
        'rating': 9.6,
        'tags': ['Souls-like', 'Fantasy', 'Difficult', 'Open World', 'RPG'],
        'multiplayer': True
    },
    {
        'title': 'Red Dead Redemption 2',
        'genre': ['Action Adventure', 'Open World'],
        'platforms': ['PC', 'PlayStation 4', 'Xbox One'],
        'price': 39.99,
        'release_date': '2018-10-26',
        'publisher': 'Rockstar Games',
        'description': "An epic tale of life in America's unforgiving heartland. The game's vast and atmospheric world also provides the foundation for a brand new online multiplayer experience.",
        'rating': 9.7,
        'tags': ['Western', 'Open World', 'Story-Rich', 'Action', 'Adventure'],
        'multiplayer': True
    },
    {
        'title': 'The Legend of Zelda: Breath of the Wild',
        'genre': ['Action Adventure', 'Open World'],
        'platforms': ['Nintendo Switch', 'Wii U'],
        'price': 59.99,
        'release_date': '2017-03-03',
        'publisher': 'Nintendo',
        'description': 'Step into a world of discovery, exploration, and adventure in The Legend of Zelda: Breath of the Wild, a boundary-breaking new game in the acclaimed series.',
        'rating': 9.5,
        'tags': ['Fantasy', 'Open World', 'Adventure', 'Action', 'Puzzles'],
        'multiplayer': False
    },
    {
        'title': 'Cyberpunk 2077',
        'genre': ['RPG', 'Open World'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 49.99,
        'release_date': '2020-12-10',
        'publisher': 'CD Projekt',
        'description': 'Cyberpunk 2077 is an open-world, action-adventure story set in Night City, a megalopolis obsessed with power, glamour and body modification.',
        'rating': 7.2,
        'tags': ['Cyberpunk', 'Open World', 'RPG', 'Futuristic', 'Story-Rich'],
        'multiplayer': False
    },
    {
        'title': 'Minecraft',
        'genre': ['Sandbox', 'Survival'],
        'platforms': ['PC', 'PlayStation', 'Xbox', 'Nintendo Switch', 'Mobile'],
        'price': 29.99,
        'release_date': '2011-11-18',
        'publisher': 'Mojang',
        'description': 'A game about placing blocks and going on adventures. Explore randomly generated worlds and build amazing things from the simplest of homes to the grandest of castles.',
        'rating': 9.0,
        'tags': ['Sandbox', 'Survival', 'Building', 'Crafting', 'Multiplayer'],
        'multiplayer': True
    },
    {
        'title': 'God of War Ragnarok',
        'genre': ['Action', 'Adventure'],
        'platforms': ['PlayStation 4', 'PlayStation 5'],
        'price': 69.99,
        'release_date': '2022-11-09',
        'publisher': 'Sony Interactive Entertainment',
        'description': 'Embark on an epic journey with Kratos and Atreus as they battle Norse gods and monsters.',
        'rating': 9.8,
        'tags': ['Action', 'Adventure', 'Mythology', 'Story-Rich', 'Hack and Slash'],
        'multiplayer': False
    },
    {
        'title': 'Horizon Forbidden West',
        'genre': ['Action RPG', 'Open World'],
        'platforms': ['PlayStation 4', 'PlayStation 5'],
        'price': 69.99,
        'release_date': '2022-02-18',
        'publisher': 'Sony Interactive Entertainment',
        'description': 'Explore distant lands, fight bigger and more awe-inspiring machines, and meet new tribes.',
        'rating': 9.0,
        'tags': ['Action', 'Open World', 'Story-Rich', 'Exploration', 'Adventure'],
        'multiplayer': False
    },
    {
        'title': 'Grand Theft Auto V',
        'genre': ['Action', 'Open World'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 29.99,
        'release_date': '2013-09-17',
        'publisher': 'Rockstar Games',
        'description': 'Experience a life of crime in Los Santos in one of the most critically acclaimed open-world games.',
        'rating': 9.5,
        'tags': ['Crime', 'Open World', 'Action', 'Multiplayer', 'Driving'],
        'multiplayer': True
    }
]

# Game vocabulary
game_vocabulary = {
    'genres': [
        'rpg', 'action', 'adventure', 'strategy', 'simulation', 'sports', 
        'racing', 'puzzle', 'platformer', 'shooter', 'horror', 'survival',
        'open world', 'sandbox', 'battle royale', 'mmorpg', 'indie',
        'role-playing', 'first-person', 'third-person', 'stealth', 'fighting',
        'card game', 'board game', 'casual', 'educational', 'music', 'rhythm'
    ],
    'features': [
        'multiplayer', 'single player', 'co-op', 'competitive', 'story',
        'online', 'offline', 'fantasy', 'sci-fi', 'futuristic', 'historical',
        'realistic', 'casual', 'hardcore', 'difficult', 'easy', 'relaxing',
        'challenging', 'immersive', 'atmospheric', 'graphics', 'soundtrack',
        'voice acting', 'mods', 'dlc', 'expansion', 'update', 'patch'
    ],
    'platforms': [
        'pc', 'playstation', 'ps4', 'ps5', 'xbox', 'nintendo', 'switch',
        'mobile', 'android', 'ios', 'mac', 'linux', 'steam', 'epic',
        'gog', 'origin', 'uplay', 'battle.net', 'stadia', 'cloud gaming'
    ],
    'price_indicators': [
        'free', 'cheap', 'expensive', 'budget', 'premium', 'discount',
        'sale', 'bundle', 'deal', 'offer', 'promotion', 'gratuit',
        'bon marché', 'cher', 'coûteux', 'réduction', 'promo'
    ]
}

# FAQ database
faqs = {
    'return_policy': 'Our return policy allows returns within 14 days of purchase with a valid receipt.',
    'shipping': 'We offer free shipping on orders over $50. Standard shipping takes 3-5 business days.',
    'payment_methods': 'We accept all major credit cards, PayPal, and store gift cards.',
    'account_creation': 'To create an account, click the "Sign Up" button in the top right corner of our website.',
    'order_tracking': 'You can track your order by logging into your account and viewing your order history.',
    'download_games': 'Digital games can be downloaded immediately after purchase from your account library.',
    'system_requirements': 'System requirements for games can be found on the individual game pages.',
    'pre_order': 'Pre-ordering gives you access to exclusive bonuses and ensures you get the game on release day.',
    'refund': 'Digital purchases can be refunded within 7 days if you haven\'t downloaded the game.',
    'gift_cards': 'Gift cards are available in denominations of $25, $50, and $100 and never expire.'
}

# Navigation guides
navigation_guides = {
    'homepage': 'You can return to the homepage by clicking the logo at the top left of any page.',
    'games': 'Browse all games by clicking the "Games" tab in the main navigation bar.',
    'deals': 'Current deals and discounts can be found under the "Special Offers" section.',
    'account': 'Access your account settings by clicking on your username in the top right corner.',
    'cart': 'View your shopping cart by clicking the cart icon in the top right corner.',
    'wishlist': 'Your wishlist can be accessed from your account dashboard.',
    'genres': 'Browse games by genre using the dropdown menu in the Games section.',
    'new_releases': 'New releases are featured on the homepage and in the "New Releases" section.',
    'bestsellers': 'Our most popular games can be found in the "Bestsellers" section.',
    'upcoming': 'See upcoming releases in the "Coming Soon" section of our website.'
}

# Challenges
challenges = {
    'general': [
        {
            'name': {'en': 'Easter Egg Hunter', 'fr': 'Chasseur d\'Oeufs de Pâques'},
            'description': {
                'en': 'Find 5 hidden Easter eggs across our website for a 10% discount.',
                'fr': 'Trouvez 5 œufs de Pâques cachés sur notre site pour obtenir une réduction de 10%.'
            },
            'reward': {
                'en': '10% discount on next purchase',
                'fr': '10% de réduction sur votre prochain achat'
            }
        },
        {
            'name': {'en': 'Game Master', 'fr': 'Maître du Jeu'},
            'description': {
                'en': 'Complete our gaming trivia quiz with a perfect score.',
                'fr': 'Complétez notre quiz de culture gaming avec un score parfait.'
            },
            'reward': {
                'en': 'Exclusive profile badge',
                'fr': 'Badge de profil exclusif'
            }
        }
    ],
    'game_specific': {
        'Fortnite': [
            {
                'name': {'en': 'Victory Royale Master', 'fr': 'Maître de la Victoire Royale'},
                'description': {
                    'en': 'Win a match with at least 10 eliminations.',
                    'fr': 'Gagnez une partie avec au moins 10 éliminations.'
                },
                'difficulty': 'Hard',
                'reward': {
                    'en': 'Exclusive Victory Crown emote',
                    'fr': 'Émote Couronne de Victoire exclusive'
                }
            }
        ],
        'Minecraft': [
            {
                'name': {'en': 'Master Builder', 'fr': 'Maître Bâtisseur'},
                'description': {
                    'en': 'Build a castle using at least 1000 blocks.',
                    'fr': 'Construisez un château en utilisant au moins 1000 blocs.'
                },
                'difficulty': 'Medium',
                'reward': {
                    'en': 'Exclusive building templates',
                    'fr': 'Modèles de construction exclusifs'
                }
            }
        ]
    }
}

# Translations
translations = {
    'greeting': {
        'en': "Hello! Welcome to our gaming store. I'm your AI assistant and can help you find games, answer questions, provide support, or even suggest fun community challenges. How can I help you today?",
        'fr': "Bonjour! Bienvenue dans notre magasin de jeux. Je suis votre assistant IA et je peux vous aider à trouver des jeux, répondre à vos questions, fournir un support ou même suggérer des défis communautaires amusants. Comment puis-je vous aider aujourd'hui?"
    },
    'goodbye': {
        'en': "Thanks for chatting! If you need any more gaming recommendations or help, just message me again. Happy gaming!",
        'fr': "Merci d'avoir discuté! Si vous avez besoin de recommandations de jeux ou d'aide, n'hésitez pas à me recontacter. Bon jeu!"
    },
    'no_games_found': {
        'en': "I'm having trouble finding games that match your preferences. Could you tell me more about what types of games you enjoy?",
        'fr': "J'ai du mal à trouver des jeux qui correspondent à vos préférences. Pourriez-vous me dire quels types de jeux vous aimez?"
    },
    'thanks': {
        'en': "You're welcome! I'm always here to help with your gaming needs. Is there anything else you'd like to know?",
        'fr': "Je vous en prie! Je suis toujours là pour vous aider avec vos besoins en matière de jeux. Y a-t-il autre chose que vous aimeriez savoir?"
    }
}

# Reasoning rules
reasoning_rules = {
    'game_recommendation': {
        'genres': ['rpg', 'action', 'adventure', 'strategy', 'simulation', 'sports', 'racing', 'puzzle', 'platformer', 'shooter'],
        'features': ['multiplayer', 'single player', 'co-op', 'competitive', 'story', 'online', 'offline'],
        'price_ranges': ['free', 'cheap', 'medium', 'expensive', 'premium'],
        'platforms': ['pc', 'playstation', 'xbox', 'nintendo', 'mobile']
    },
    'technical_support': {
        'issues': ['download', 'install', 'launch', 'crash', 'error', 'bug', 'performance'],
        'solutions': ['update', 'patch', 'fix', 'repair', 'reinstall', 'verify']
    },
    'price': {
        'indicators': ['cost', 'price', 'expensive', 'cheap', 'free', 'discount', 'sale'],
        'comparisons': ['more than', 'less than', 'around', 'between', 'under', 'over']
    }
}

# Intent patterns
intent_patterns = {
    'greeting': r'(?i)^(hi|hello|hey|greetings|sup|what\'s up|howdy|bonjour|salut|bonsoir|coucou|yo|hi there|hello there).*',
    'goodbye': r'(?i)^(bye|goodbye|see you|exit|quit|au revoir|à bientôt|à plus tard|bonne journée|à tout à l\'heure|au plaisir).*',
    'game_recommendation': r'(?i).*(recommend|suggest|recommander|suggérer|conseiller|proposer|donner|trouver|chercher|rechercher|aimerais|voudrais|souhaiterais).*(game|games|jeux|jeu|titre|titles).*',
    'game_info': r'(?i).*(info|information|details|détails|savoir|connaître|apprendre|découvrir).*(about|sur|concernant|à propos).*(game|games|jeux|jeu|titre|titles).*',
    'navigation': r'(?i).*(how|where|comment|où).*(find|navigate|go to|trouver|naviguer|aller|accéder|localiser|retrouver).*',
    'technical_support': r'(?i).*(help|fix|issue|problem|error|trouble|not working|aidez-moi|problème|erreur|bug|dysfonctionnement|souci|difficulté|ne marche pas).*',
    'faq': r'(?i).*(faq|question|how do i|what is|when|where|question|comment faire|qu\'est-ce que|quand|où|comment|pourquoi|quel|qui).*',
    'community': r'(?i).*(community|quest|challenge|event|tournament|communauté|défi|événement|tournoi|groupe|rencontre|compétition).*',
    'price': r'(?i).*(price|cost|how much|prix|coût|combien|tarif|valeur|montant|budget|gratuit|cher|bon marché).*',
    'thanks': r'(?i)^(thanks|thank you|thx|merci|merci beaucoup|je vous remercie|merci infiniment|thanks a lot).*',
    'challenges': r'(?i).*(challenge|quest|mission|achievement|défi|quête|mission|succès|objectif|tâche|daily|quotidien|hebdomadaire|weekly).*(what|show|tell|give|voir|montre|dis|donne).*'
}

# Word synonyms
synonyms = {
    'recommend': ['suggest', 'advise', 'propose', 'recommander', 'suggérer', 'conseiller', 'proposer'],
    'help': ['assist', 'support', 'aid', 'aider', 'assister', 'soutenir'],
    'game': ['title', 'video game', 'product', 'jeu', 'titre', 'produit'],
    'community': ['forum', 'group', 'society', 'communauté', 'groupe', 'société'],
    'challenge': ['quest', 'task', 'mission', 'défi', 'quête', 'tâche'],
    'price': ['cost', 'value', 'amount', 'prix', 'coût', 'montant'],
    'problem': ['issue', 'trouble', 'error', 'problème', 'souci', 'erreur'],
    'question': ['query', 'inquiry', 'question', 'demande', 'interrogation'],
    'find': ['search', 'locate', 'discover', 'trouver', 'chercher', 'localiser'],
    'buy': ['purchase', 'get', 'acquire', 'acheter', 'obtenir', 'acquérir']
} 
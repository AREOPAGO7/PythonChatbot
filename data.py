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
        'title': 'God of War Ragnarök',
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
    },
    {
        'title': 'The Witcher 3: Wild Hunt',
        'genre': ['Action RPG', 'Open World'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 39.99,
        'release_date': '2015-05-19',
        'publisher': 'CD Projekt',
        'description': 'An epic role-playing game set in a vast open world, where every choice has far-reaching consequences.',
        'rating': 9.8,
        'tags': ['Fantasy', 'RPG', 'Story-Rich', 'Open World', 'Action'],
        'multiplayer': False
    },
    {
        'title': 'Final Fantasy XVI',
        'genre': ['Action RPG', 'Fantasy'],
        'platforms': ['PlayStation 5'],
        'price': 69.99,
        'release_date': '2023-06-22',
        'publisher': 'Square Enix',
        'description': 'An epic dark fantasy world where fate of the realm depends on powerful Dominants and their Eikon abilities.',
        'rating': 8.8,
        'tags': ['Fantasy', 'Action', 'Story-Rich', 'RPG', 'Single-player'],
        'multiplayer': False
    },
    {
        'title': "Baldur's Gate 3",
        'genre': ['RPG', 'Strategy'],
        'platforms': ['PC', 'PlayStation 5'],
        'price': 59.99,
        'release_date': '2023-08-03',
        'publisher': 'Larian Studios',
        'description': 'Gather your party and return to the Forgotten Realms in a tale of fellowship and betrayal, sacrifice and survival.',
        'rating': 9.7,
        'tags': ['Fantasy', 'RPG', 'Turn-Based', 'Story-Rich', 'Dungeons & Dragons'],
        'multiplayer': True
    },
    {
        'title': 'Resident Evil 4 Remake',
        'genre': ['Horror', 'Action'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox Series X/S'],
        'price': 59.99,
        'release_date': '2023-03-24',
        'publisher': 'Capcom',
        'description': 'A reimagining of the 2005 survival horror classic, featuring modernized gameplay and stunning visuals.',
        'rating': 9.3,
        'tags': ['Horror', 'Survival Horror', 'Action', 'Remake', 'Story-Rich'],
        'multiplayer': False
    },
    {
        'title': 'Spider-Man 2',
        'genre': ['Action Adventure', 'Open World'],
        'platforms': ['PlayStation 5'],
        'price': 69.99,
        'release_date': '2023-10-20',
        'publisher': 'Sony Interactive Entertainment',
        'description': "Swing through Marvel's New York as Peter Parker and Miles Morales in this thrilling superhero adventure.",
        'rating': 9.2,
        'tags': ['Superhero', 'Action', 'Open World', 'Story-Rich', 'Adventure'],
        'multiplayer': False
    },
    {
        'title': 'Starfield',
        'genre': ['RPG', 'Open World'],
        'platforms': ['PC', 'Xbox Series X/S'],
        'price': 69.99,
        'release_date': '2023-09-06',
        'publisher': 'Bethesda',
        'description': "Embark on an epic journey through space in Bethesda Game Studios' first new universe in over 25 years.",
        'rating': 8.5,
        'tags': ['Sci-Fi', 'Space', 'RPG', 'Exploration', 'Open World'],
        'multiplayer': False
    },
    {
        'title': 'Diablo IV',
        'genre': ['Action RPG', 'Hack and Slash'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 69.99,
        'release_date': '2023-06-06',
        'publisher': 'Blizzard Entertainment',
        'description': 'Return to darkness with the newest installment in the genre-defining ARPG series.',
        'rating': 8.7,
        'tags': ['Dark Fantasy', 'Action RPG', 'Multiplayer', 'Hack and Slash', 'Dungeon Crawler'],
        'multiplayer': True
    },
    {
        'title': 'Street Fighter 6',
        'genre': ['Fighting'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox Series X/S'],
        'price': 59.99,
        'release_date': '2023-06-02',
        'publisher': 'Capcom',
        'description': 'The newest edition of the legendary fighting game series featuring new modes and modern controls.',
        'rating': 9.0,
        'tags': ['Fighting', 'Competitive', 'Arcade', 'eSports', 'Multiplayer'],
        'multiplayer': True
    },
    {
        'title': "Assassin's Creed Mirage",
        'genre': ['Action Adventure', 'Stealth'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 49.99,
        'release_date': '2023-10-05',
        'publisher': 'Ubisoft',
        'description': 'Experience the story of Basim, a cunning street thief seeking answers and justice in ninth-century Baghdad.',
        'rating': 8.4,
        'tags': ['Stealth', 'Action', 'Historical', 'Adventure', 'Story-Rich'],
        'multiplayer': False
    },
    {
        'title': 'Hogwarts Legacy',
        'genre': ['Action RPG', 'Open World'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 59.99,
        'release_date': '2023-02-10',
        'publisher': 'Warner Bros. Games',
        'description': 'Embark on a journey through the magical world of Harry Potter as a student at Hogwarts School of Witchcraft and Wizardry.',
        'rating': 8.9,
        'tags': ['Magic', 'Fantasy', 'RPG', 'Open World', 'Story-Rich'],
        'multiplayer': False
    },
    {
        'title': 'Dead Space Remake',
        'genre': ['Horror', 'Action'],
        'platforms': ['PC', 'PlayStation 5', 'Xbox Series X/S'],
        'price': 59.99,
        'release_date': '2023-01-27',
        'publisher': 'Electronic Arts',
        'description': 'A ground-up remake of the classic sci-fi survival horror game with stunning visuals and enhanced gameplay.',
        'rating': 9.0,
        'tags': ['Horror', 'Sci-Fi', 'Survival Horror', 'Space', 'Remake'],
        'multiplayer': False
    },
    {
        'title': 'The Last of Us Part I',
        'genre': ['Action Adventure', 'Survival'],
        'platforms': ['PC', 'PlayStation 5'],
        'price': 59.99,
        'release_date': '2022-09-02',
        'publisher': 'Sony Interactive Entertainment',
        'description': 'Experience the emotional masterpiece that launched a new franchise, completely rebuilt for a new generation.',
        'rating': 9.4,
        'tags': ['Survival', 'Story-Rich', 'Action', 'Post-apocalyptic', 'Remake'],
        'multiplayer': False
    },
    {
        'title': 'FIFA 24',
        'genre': ['Sports', 'Simulation'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 69.99,
        'release_date': '2023-09-29',
        'publisher': 'EA Sports',
        'description': "Experience the world's game with authentic football action and unrivaled realism.",
        'rating': 8.2,
        'tags': ['Sports', 'Football', 'Multiplayer', 'Competitive', 'Simulation'],
        'multiplayer': True
    },
    {
        'title': 'Call of Duty: Modern Warfare III',
        'genre': ['FPS', 'Action'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 69.99,
        'release_date': '2023-11-10',
        'publisher': 'Activision',
        'description': 'The latest installment in the blockbuster FPS series featuring intense multiplayer combat and a thrilling campaign.',
        'rating': 8.0,
        'tags': ['FPS', 'Multiplayer', 'Action', 'Military', 'Competitive'],
        'multiplayer': True
    },
    {
        'title': 'Mortal Kombat 1',
        'genre': ['Fighting'],
        'platforms': ['PC', 'PlayStation 5', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 69.99,
        'release_date': '2023-09-19',
        'publisher': 'Warner Bros. Games',
        'description': 'Experience a reborn Mortal Kombat Universe created by the Fire God Liu Kang, with brutal fighting and fatalities.',
        'rating': 8.8,
        'tags': ['Fighting', 'Gore', 'Competitive', 'Arcade', 'Story-Rich'],
        'multiplayer': True
    },
    {
        'title': 'Sea of Thieves',
        'genre': ['Action Adventure', 'Open World'],
        'platforms': ['PC', 'Xbox One', 'Xbox Series X/S'],
        'price': 39.99,
        'release_date': '2018-03-20',
        'publisher': 'Microsoft Studios',
        'description': 'Be the pirate you want to be in a shared-world adventure game filled with sailing, treasure hunting, and sea battles.',
        'rating': 8.5,
        'tags': ['Pirates', 'Multiplayer', 'Open World', 'Adventure', 'Co-op'],
        'multiplayer': True
    },
    {
        'title': 'Persona 5 Royal',
        'genre': ['JRPG', 'Social Simulation'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 59.99,
        'release_date': '2022-10-21',
        'publisher': 'Atlus',
        'description': 'Don the mask of a Phantom Thief and fight for justice in this critically acclaimed JRPG.',
        'rating': 9.5,
        'tags': ['JRPG', 'Story-Rich', 'Turn-Based', 'Anime', 'Social Sim'],
        'multiplayer': False
    },
    {
        'title': 'Monster Hunter: World',
        'genre': ['Action RPG', 'Adventure'],
        'platforms': ['PC', 'PlayStation 4', 'Xbox One'],
        'price': 29.99,
        'release_date': '2018-01-26',
        'publisher': 'Capcom',
        'description': 'Hunt incredible monsters in a living, breathing ecosystem and use the materials to craft gear and weapons.',
        'rating': 9.0,
        'tags': ['Action', 'RPG', 'Co-op', 'Monster Hunter', 'Multiplayer'],
        'multiplayer': True
    },
    {
        'title': 'Overwatch 2',
        'genre': ['FPS', 'Hero Shooter'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 0.00,
        'release_date': '2022-10-04',
        'publisher': 'Blizzard Entertainment',
        'description': 'Team-based action game set in an optimistic future, where every match is the ultimate 5v5 battlefield brawl.',
        'rating': 7.8,
        'tags': ['FPS', 'Hero Shooter', 'Competitive', 'Team-Based', 'Free-to-Play'],
        'multiplayer': True
    },
    {
        'title': 'Destiny 2',
        'genre': ['FPS', 'Action MMO'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 0.00,
        'release_date': '2017-09-06',
        'publisher': 'Bungie',
        'description': 'A free-to-play online-only multiplayer first-person shooter set in a mythic science fiction world.',
        'rating': 8.5,
        'tags': ['FPS', 'MMO', 'Looter Shooter', 'Sci-Fi', 'Free-to-Play'],
        'multiplayer': True
    },
    {
        'title': 'Stardew Valley',
        'genre': ['Simulation', 'RPG'],
        'platforms': ['PC', 'PlayStation 4', 'Xbox One', 'Nintendo Switch', 'Mobile'],
        'price': 14.99,
        'release_date': '2016-02-26',
        'publisher': 'ConcernedApe',
        'description': "Build the farm of your dreams and create a new life in the valley, where you'll meet unique characters and explore vast caves.",
        'rating': 9.3,
        'tags': ['Farming', 'Life Sim', 'RPG', 'Pixel Graphics', 'Relaxing'],
        'multiplayer': True
    },
    {
        'title': 'Hades',
        'genre': ['Action Roguelike', 'Indie'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 24.99,
        'release_date': '2020-09-17',
        'publisher': 'Supergiant Games',
        'description': 'Defy the god of the dead as you hack and slash out of the Underworld in this rogue-like dungeon crawler.',
        'rating': 9.4,
        'tags': ['Roguelike', 'Action', 'Greek Mythology', 'Indie', 'Story-Rich'],
        'multiplayer': False
    },
    {
        'title': 'Among Us',
        'genre': ['Party', 'Social Deduction'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch', 'Mobile'],
        'price': 4.99,
        'release_date': '2018-06-15',
        'publisher': 'InnerSloth',
        'description': 'An online and local party game of teamwork and betrayal for 4-15 players in space.',
        'rating': 8.7,
        'tags': ['Party', 'Social Deduction', 'Multiplayer', 'Casual', 'Space'],
        'multiplayer': True
    },
    {
        'title': 'Hollow Knight',
        'genre': ['Metroidvania', 'Action'],
        'platforms': ['PC', 'PlayStation 4', 'Xbox One', 'Nintendo Switch'],
        'price': 14.99,
        'release_date': '2017-02-24',
        'publisher': 'Team Cherry',
        'description': 'Forge your own path in an epic action adventure through a vast ruined kingdom of insects and heroes.',
        'rating': 9.4,
        'tags': ['Metroidvania', 'Difficult', 'Atmospheric', 'Indie', 'Story-Rich'],
        'multiplayer': False
    },
    {
        'title': 'Fortnite',
        'genre': ['Battle Royale', 'Third-Person Shooter'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch', 'Mobile'],
        'price': 0.00,
        'release_date': '2017-07-25',
        'publisher': 'Epic Games',
        'description': 'A free-to-play battle royale game where 100 players fight to be the last person standing.',
        'rating': 8.3,
        'tags': ['Battle Royale', 'Multiplayer', 'Building', 'Third-Person Shooter', 'Free-to-Play'],
        'multiplayer': True
    },
    {
        'title': 'Terraria',
        'genre': ['Action Adventure', 'Sandbox'],
        'platforms': ['PC', 'PlayStation 4', 'Xbox One', 'Nintendo Switch', 'Mobile'],
        'price': 9.99,
        'release_date': '2011-05-16',
        'publisher': 'Re-Logic',
        'description': 'Dig, fight, explore, build! Nothing is impossible in this action-packed adventure game.',
        'rating': 9.2,
        'tags': ['Sandbox', 'Adventure', 'Crafting', 'Pixel Graphics', 'Multiplayer'],
        'multiplayer': True
    },
    {
        'title': 'Valheim',
        'genre': ['Survival', 'Open World'],
        'platforms': ['PC', 'Xbox Series X/S'],
        'price': 19.99,
        'release_date': '2021-02-02',
        'publisher': 'Coffee Stain Publishing',
        'description': 'A brutal exploration and survival game for 1-10 players set in a procedurally-generated world inspired by Norse mythology.',
        'rating': 8.9,
        'tags': ['Survival', 'Viking', 'Open World', 'Crafting', 'Co-op'],
        'multiplayer': True
    },
    {
        'title': 'Subnautica',
        'genre': ['Survival', 'Adventure'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 29.99,
        'release_date': '2018-01-23',
        'publisher': 'Unknown Worlds Entertainment',
        'description': 'Descend into the depths of an alien underwater world filled with wonder and peril.',
        'rating': 9.1,
        'tags': ['Survival', 'Underwater', 'Exploration', 'Crafting', 'Story-Rich'],
        'multiplayer': False
    },
    {
        'title': 'Sekiro: Shadows Die Twice',
        'genre': ['Action Adventure', 'Souls-like'],
        'platforms': ['PC', 'PlayStation 4', 'Xbox One'],
        'price': 59.99,
        'release_date': '2019-03-22',
        'publisher': 'Activision',
        'description': 'Carve your own clever path to vengeance in an all-new adventure from developer FromSoftware.',
        'rating': 9.3,
        'tags': ['Souls-like', 'Difficult', 'Action', 'Ninja', 'Story-Rich'],
        'multiplayer': False
    },
    {
        'title': 'Rocket League',
        'genre': ['Sports', 'Racing'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 0.00,
        'release_date': '2015-07-07',
        'publisher': 'Psyonix LLC',
        'description': 'High-powered hybrid of arcade-style soccer and vehicular mayhem with easy-to-understand controls.',
        'rating': 8.8,
        'tags': ['Sports', 'Cars', 'Soccer', 'Competitive', 'Free-to-Play'],
        'multiplayer': True
    },
    {
        'title': 'Celeste',
        'genre': ['Platformer', 'Indie'],
        'platforms': ['PC', 'PlayStation 4', 'Xbox One', 'Nintendo Switch'],
        'price': 19.99,
        'release_date': '2018-01-25',
        'publisher': 'Matt Makes Games',
        'description': 'Help Madeline survive her inner demons on her journey to the top of Celeste Mountain.',
        'rating': 9.2,
        'tags': ['Platformer', 'Difficult', 'Story-Rich', 'Pixel Graphics', 'Indie'],
        'multiplayer': False
    },
    {
        'title': 'Undertale',
        'genre': ['RPG', 'Indie'],
        'platforms': ['PC', 'PlayStation 4', 'Xbox One', 'Nintendo Switch'],
        'price': 9.99,
        'release_date': '2015-09-15',
        'publisher': 'Toby Fox',
        'description': 'A unique RPG where nobody has to die - befriend all of the monsters!',
        'rating': 9.5,
        'tags': ['RPG', 'Story-Rich', 'Pixel Graphics', 'Indie', 'Comedy'],
        'multiplayer': False
    },
    {
        'title': 'Fall Guys',
        'genre': ['Party', 'Battle Royale'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 0.00,
        'release_date': '2020-08-04',
        'publisher': 'Epic Games',
        'description': 'A massively multiplayer party game with up to 60 players online in a free-for-all struggle.',
        'rating': 8.1,
        'tags': ['Party', 'Battle Royale', 'Multiplayer', 'Casual', 'Free-to-Play'],
        'multiplayer': True
    },
    {
        'title': 'Cuphead',
        'genre': ['Platformer', 'Run and Gun'],
        'platforms': ['PC', 'PlayStation 4', 'Xbox One', 'Nintendo Switch'],
        'price': 19.99,
        'release_date': '2017-09-29',
        'publisher': 'Studio MDHR',
        'description': 'A classic run and gun action game heavily focused on boss battles with a unique visual style.',
        'rating': 8.9,
        'tags': ['Difficult', 'Hand-drawn', 'Co-op', 'Run and Gun', 'Indie'],
        'multiplayer': True
    },
    {
        'title': "No Man's Sky",
        'genre': ['Action Adventure', 'Survival'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 59.99,
        'release_date': '2016-08-09',
        'publisher': 'Hello Games',
        'description': 'A science fiction game about exploration and survival in an infinite procedurally generated universe.',
        'rating': 8.4,
        'tags': ['Space', 'Exploration', 'Survival', 'Multiplayer', 'Sci-Fi'],
        'multiplayer': True
    },
    {
        'title': 'Deep Rock Galactic',
        'genre': ['FPS', 'Co-op'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 29.99,
        'release_date': '2020-05-13',
        'publisher': 'Coffee Stain Publishing',
        'description': 'Work together as a team of dwarf miners in this 1-4 player co-op FPS featuring destructible environments.',
        'rating': 9.1,
        'tags': ['Co-op', 'FPS', 'Mining', 'Exploration', 'Team-Based'],
        'multiplayer': True
    },
    {
        'title': 'Factorio',
        'genre': ['Strategy', 'Simulation'],
        'platforms': ['PC', 'Nintendo Switch'],
        'price': 30.00,
        'release_date': '2020-08-14',
        'publisher': 'Wube Software',
        'description': 'Build and maintain automated factories in this complex management and construction game.',
        'rating': 9.6,
        'tags': ['Automation', 'Management', 'Strategy', 'Building', 'Multiplayer'],
        'multiplayer': True
    },
    {
        'title': 'Rimworld',
        'genre': ['Strategy', 'Simulation'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 34.99,
        'release_date': '2018-10-17',
        'publisher': 'Ludeon Studios',
        'description': 'A sci-fi colony sim driven by an intelligent AI storyteller.',
        'rating': 9.4,
        'tags': ['Colony Sim', 'Management', 'Strategy', 'Survival', 'Story Generator'],
        'multiplayer': False
    },
    {
        'title': 'Satisfactory',
        'genre': ['Simulation', 'Open World'],
        'platforms': ['PC'],
        'price': 29.99,
        'release_date': '2019-03-19',
        'publisher': 'Coffee Stain Publishing',
        'description': 'Build massive factories in a first-person open-world exploration and automation game.',
        'rating': 9.0,
        'tags': ['Factory Building', 'Open World', 'Automation', 'Multiplayer', 'Exploration'],
        'multiplayer': True
    },
    {
        'title': 'Outer Wilds',
        'genre': ['Adventure', 'Mystery'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 24.99,
        'release_date': '2019-05-28',
        'publisher': 'Annapurna Interactive',
        'description': 'Explore a mysterious solar system stuck in an endless time loop.',
        'rating': 9.3,
        'tags': ['Space', 'Exploration', 'Mystery', 'Time Loop', 'Story-Rich'],
        'multiplayer': False
    },
    {
        'title': 'Disco Elysium',
        'genre': ['RPG', 'Adventure'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 39.99,
        'release_date': '2019-10-15',
        'publisher': 'ZA/UM',
        'description': 'A groundbreaking role playing game where your skills are your party members.',
        'rating': 9.6,
        'tags': ['RPG', 'Detective', 'Story-Rich', 'Choices Matter', 'Atmospheric'],
        'multiplayer': False
    },
    {
        'title': 'Cult of the Lamb',
        'genre': ['Action Roguelike', 'Management'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 24.99,
        'release_date': '2022-08-11',
        'publisher': 'Devolver Digital',
        'description': 'Start your own cult in a land of false prophets, venturing out into diverse and mysterious regions.',
        'rating': 8.7,
        'tags': ['Roguelike', 'Cute', 'Dark', 'Management', 'Action'],
        'multiplayer': False
    },
    {
        'title': 'Vampire Survivors',
        'genre': ['Action', 'Roguelike'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch', 'Mobile'],
        'price': 4.99,
        'release_date': '2022-10-20',
        'publisher': 'poncle',
        'description': 'A time survival game with minimalistic gameplay and roguelite elements.',
        'rating': 9.1,
        'tags': ['Roguelite', 'Pixel Graphics', 'Casual', 'Action', 'Indie'],
        'multiplayer': False
    },
    {
        'title': 'Lethal Company',
        'genre': ['Horror', 'Co-op'],
        'platforms': ['PC'],
        'price': 9.99,
        'release_date': '2023-10-23',
        'publisher': 'Zeekerss',
        'description': 'A co-op horror game about exploring abandoned moons to salvage resources for profit.',
        'rating': 9.2,
        'tags': ['Horror', 'Co-op', 'Exploration', 'Indie', 'Atmospheric'],
        'multiplayer': True
    },
    {
        'title': 'Palworld',
        'genre': ['Action Adventure', 'Survival'],
        'platforms': ['PC', 'Xbox Series X/S'],
        'price': 29.99,
        'release_date': '2024-01-19',
        'publisher': 'Pocketpair',
        'description': 'A multiplayer creature-collection survival game where you can explore, build, farm and work alongside mysterious creatures called "Pals".',
        'rating': 8.8,
        'tags': ['Survival', 'Multiplayer', 'Open World', 'Crafting', 'Creature Collector'],
        'multiplayer': True
    },
    {
        'title': 'Helldivers 2',
        'genre': ['Third-Person Shooter', 'Co-op'],
        'platforms': ['PC', 'PlayStation 5'],
        'price': 39.99,
        'release_date': '2024-02-08',
        'publisher': 'Sony Interactive Entertainment',
        'description': 'A cooperative shooter where players fight for Super Earth against alien threats.',
        'rating': 8.9,
        'tags': ['Shooter', 'Co-op', 'Action', 'Sci-Fi', 'Multiplayer'],
        'multiplayer': True
    },
    {
        'title': 'Lies of P',
        'genre': ['Action RPG', 'Souls-like'],
        'platforms': ['PC', 'PlayStation 5', 'Xbox Series X/S'],
        'price': 59.99,
        'release_date': '2023-09-19',
        'publisher': 'Neowiz Games',
        'description': 'A souls-like action RPG inspired by the story of Pinocchio, set in a dark Belle Époque world.',
        'rating': 8.6,
        'tags': ['Souls-like', 'Dark Fantasy', 'Action', 'Story-Rich', 'Difficult'],
        'multiplayer': False
    },
    {
        'title': 'Dave the Diver',
        'genre': ['Adventure', 'Management'],
        'platforms': ['PC', 'Nintendo Switch'],
        'price': 19.99,
        'release_date': '2023-06-28',
        'publisher': 'Mintrocket',
        'description': 'Manage a sushi restaurant by day and explore the mysterious Blue Hole by night in this underwater adventure.',
        'rating': 9.0,
        'tags': ['Management', 'Adventure', 'Underwater', 'Indie', 'Relaxing'],
        'multiplayer': False
    },
    {
        'title': 'Hi-Fi Rush',
        'genre': ['Action', 'Rhythm'],
        'platforms': ['PC', 'Xbox Series X/S'],
        'price': 29.99,
        'release_date': '2023-01-25',
        'publisher': 'Bethesda Softworks',
        'description': 'A rhythm-action game where everything moves to the beat, from combat to platforming.',
        'rating': 9.1,
        'tags': ['Rhythm', 'Action', 'Stylish', 'Music', 'Colorful'],
        'multiplayer': False
    },
    {
        'title': 'Remnant II',
        'genre': ['Action RPG', 'Third-Person Shooter'],
        'platforms': ['PC', 'PlayStation 5', 'Xbox Series X/S'],
        'price': 49.99,
        'release_date': '2023-07-25',
        'publisher': 'Gearbox Publishing',
        'description': 'A third-person survival action shooter where players create their own adventure across dangerous worlds.',
        'rating': 8.8,
        'tags': ['Action', 'Co-op', 'Shooter', 'RPG', 'Souls-like'],
        'multiplayer': True
    },
    {
        'title': 'Sea of Stars',
        'genre': ['RPG', 'Turn-Based'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 34.99,
        'release_date': '2023-08-29',
        'publisher': 'Sabotage Studio',
        'description': 'A turn-based RPG inspired by classics, telling the story of two Children of the Solstice.',
        'rating': 9.0,
        'tags': ['RPG', 'Turn-Based', 'Pixel Graphics', 'Story-Rich', 'Indie'],
        'multiplayer': False
    },
    {
        'title': 'Armored Core VI: Fires of Rubicon',
        'genre': ['Action', 'Mecha'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 59.99,
        'release_date': '2023-08-25',
        'publisher': 'Bandai Namco',
        'description': 'A mecha action game where players assemble and pilot their own customized mechs.',
        'rating': 8.7,
        'tags': ['Mecha', 'Action', 'Difficult', 'Customization', 'Sci-Fi'],
        'multiplayer': True
    },
    {
        'title': 'Atomic Heart',
        'genre': ['Action RPG', 'FPS'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 59.99,
        'release_date': '2023-02-21',
        'publisher': 'Focus Entertainment',
        'description': 'An action RPG in an alternate reality Soviet Union where technology has gone rogue.',
        'rating': 8.2,
        'tags': ['FPS', 'Action', 'Sci-Fi', 'Story-Rich', 'Alternate History'],
        'multiplayer': False
    },
    {
        'title': 'Dead Island 2',
        'genre': ['Action RPG', 'Zombie'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 59.99,
        'release_date': '2023-04-21',
        'publisher': 'Deep Silver',
        'description': 'A first-person action RPG set in a zombie-infested Los Angeles.',
        'rating': 8.1,
        'tags': ['Zombie', 'Action', 'Gore', 'Co-op', 'Open World'],
        'multiplayer': True
    },
    {
        'title': 'Octopath Traveler II',
        'genre': ['JRPG', 'Turn-Based'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Nintendo Switch'],
        'price': 59.99,
        'release_date': '2023-02-24',
        'publisher': 'Square Enix',
        'description': 'A HD-2D RPG following eight travelers on their unique journeys across a vast continent.',
        'rating': 8.9,
        'tags': ['JRPG', 'Turn-Based', 'Story-Rich', 'Pixel Graphics', 'Fantasy'],
        'multiplayer': False
    },
    {
        'title': 'Like a Dragon: Infinite Wealth',
        'genre': ['Action RPG', 'Adventure'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S'],
        'price': 69.99,
        'release_date': '2024-01-26',
        'publisher': 'Sega',
        'description': 'An action RPG that follows Ichiban Kasuga and Kazuma Kiryu in a story spanning Japan and Hawaii.',
        'rating': 9.2,
        'tags': ['RPG', 'Story-Rich', 'Action', 'Comedy', 'Mini-games'],
        'multiplayer': False
    },
    {
        'title': 'Enshrouded',
        'genre': ['Action RPG', 'Survival'],
        'platforms': ['PC'],
        'price': 29.99,
        'release_date': '2024-01-24',
        'publisher': 'Keen Games',
        'description': 'A survival action-RPG where players build, craft, and fight in a world covered by a mysterious Shroud.',
        'rating': 8.5,
        'tags': ['Survival', 'RPG', 'Building', 'Co-op', 'Open World'],
        'multiplayer': True
    },
    {
        'title': 'Jusant',
        'genre': ['Adventure', 'Puzzle'],
        'platforms': ['PC', 'PlayStation 5', 'Xbox Series X/S'],
        'price': 24.99,
        'release_date': '2023-10-31',
        'publisher': "DON'T NOD",
        'description': 'A meditative climbing adventure where players scale a mysterious tower to uncover its secrets.',
        'rating': 8.4,
        'tags': ['Climbing', 'Atmospheric', 'Puzzle', 'Adventure', 'Relaxing'],
        'multiplayer': False
    },
    {
        'title': 'Cocoon',
        'genre': ['Puzzle', 'Adventure'],
        'platforms': ['PC', 'PlayStation 4', 'PlayStation 5', 'Xbox One', 'Xbox Series X/S', 'Nintendo Switch'],
        'price': 24.99,
        'release_date': '2023-09-29',
        'publisher': 'Annapurna Interactive',
        'description': 'A unique puzzle adventure from the creator of LIMBO and INSIDE, featuring world-within-worlds mechanics.',
        'rating': 9.0,
        'tags': ['Puzzle', 'Atmospheric', 'Adventure', 'Sci-Fi', 'Indie'],
        'multiplayer': False
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

# Platform sections data
platform_sections = {
    'profile': {
        'en': {
            'path': "Click on the user avatar in the top right of the navbar, then click 'Profile'",
            'description': "Access your profile settings, achievements, and personal information"
        },
        'fr': {
            'path': "Cliquez sur l'avatar de l'utilisateur en haut à droite de la barre de navigation, puis cliquez sur 'Profil'",
            'description': "Accédez à vos paramètres de profil, réalisations et informations personnelles"
        }
    },
    'chat': {
        'en': {
            'path': "Click on 'Messages' in the sidebar",
            'description': "Access your conversations and messages with other users"
        },
        'fr': {
            'path': "Cliquez sur 'Messages' dans la barre latérale",
            'description': "Accédez à vos conversations et messages avec d'autres utilisateurs"
        }
    },
    'store': {
        'en': {
            'path': "Click on 'Store' in the sidebar",
            'description': "Browse and purchase games from our catalog"
        },
        'fr': {
            'path': "Cliquez sur 'Boutique' dans la barre latérale",
            'description': "Parcourez et achetez des jeux de notre catalogue"
        }
    },
    'forums': {
        'en': {
            'path': "Click on 'Forums' in the navbar",
            'description': "Join discussions with other gamers"
        },
        'fr': {
            'path': "Cliquez sur 'Forums' dans la barre de navigation",
            'description': "Rejoignez les discussions avec d'autres joueurs"
        }
    },
    'browse': {
        'en': {
            'path': "Click 'Browse' near Forums",
            'description': "Discover new games and browse different categories"
        },
        'fr': {
            'path': "Cliquez sur 'Parcourir' près de Forums",
            'description': "Découvrez de nouveaux jeux et parcourez différentes catégories"
        }
    },
    'library': {
        'en': {
            'path': "Located near 'Browse' in the navbar",
            'description': "Access your game collection and purchased titles"
        },
        'fr': {
            'path': "Situé près de 'Parcourir' dans la barre de navigation",
            'description': "Accédez à votre collection de jeux et aux titres achetés"
        }
    },
    'news': {
        'en': {
            'path': "Click on 'News' in the navbar",
            'description': "Get updates on gaming trends, platform changes, and events"
        },
        'fr': {
            'path': "Cliquez sur 'Actualités' dans la barre de navigation",
            'description': "Obtenez des mises à jour sur les tendances du jeu, les changements de plateforme et les événements"
        }
    }
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
    'upcoming': 'See upcoming releases in the "Coming Soon" section of our website.',
    'platform': platform_sections
}

# Platform FAQs
platform_faqs = {
    'en': {
        'account': {
            'create': 'To create an account, click "Sign Up" in the top right corner and follow the instructions.',
            'delete': 'To delete your account, go to Profile Settings > Account > Delete Account.',
            'password': 'To change your password, go to Profile Settings > Security > Change Password.',
            'email': 'To update your email, go to Profile Settings > Account > Email Settings.'
        },
        'payment': {
            'methods': 'We accept credit cards, PayPal, and various local payment methods.',
            'refund': 'Refund requests can be submitted within 14 days of purchase if you have played less than 2 hours.',
            'subscription': 'Manage your subscription in Profile Settings > Subscriptions.'
        },
        'technical': {
            'download': 'Games can be downloaded from your Library after purchase.',
            'install': 'Follow the installation instructions provided with each game.',
            'update': 'Games are updated automatically by default. You can manage auto-updates in Settings.'
        }
    },
    'fr': {
        'account': {
            'create': 'Pour créer un compte, cliquez sur "S\'inscrire" en haut à droite et suivez les instructions.',
            'delete': 'Pour supprimer votre compte, allez dans Paramètres du profil > Compte > Supprimer le compte.',
            'password': 'Pour changer votre mot de passe, allez dans Paramètres du profil > Sécurité > Changer le mot de passe.',
            'email': 'Pour mettre à jour votre email, allez dans Paramètres du profil > Compte > Paramètres email.'
        },
        'payment': {
            'methods': 'Nous acceptons les cartes de crédit, PayPal et divers moyens de paiement locaux.',
            'refund': 'Les demandes de remboursement peuvent être soumises dans les 14 jours suivant l\'achat si vous avez joué moins de 2 heures.',
            'subscription': 'Gérez votre abonnement dans Paramètres du profil > Abonnements.'
        },
        'technical': {
            'download': 'Les jeux peuvent être téléchargés depuis votre Bibliothèque après l\'achat.',
            'install': 'Suivez les instructions d\'installation fournies avec chaque jeu.',
            'update': 'Les jeux sont mis à jour automatiquement par défaut. Vous pouvez gérer les mises à jour automatiques dans les Paramètres.'
        }
    }
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
    'refund': "Digital purchases can be refunded within 7 days if you haven't downloaded the game.",
    'gift_cards': 'Gift cards are available in denominations of $25, $50, and $100 and never expire.',
    'platform': {
        'refunds': {
            'en': "Refunds can be requested within 14 days of purchase if the game has been played for less than 2 hours. To request a refund, go to your library, select the game, and click the 'Refund' button.",
            'fr': "Les remboursements peuvent être demandés dans les 14 jours suivant l'achat si le jeu a été joué moins de 2 heures. Pour demander un remboursement, allez dans votre bibliothèque, sélectionnez le jeu et cliquez sur le bouton 'Rembourser'."
        },
        'bonus_points': {
            'en': "Bonus points are earned through purchases (1 point per $1 spent), special events, and achievements. You can redeem them for discounts in the store. Check your profile to see your current points balance.",
            'fr': "Les points bonus sont gagnés grâce aux achats (1 point par 1€ dépensé), événements spéciaux et succès. Vous pouvez les échanger contre des réductions dans la boutique. Consultez votre profil pour voir votre solde de points actuel."
        },
        'username_change': {
            'en': "You can change your username from the profile settings once every 30 days. Go to Profile > Settings > Account to make the change.",
            'fr': "Vous pouvez changer votre nom d'utilisateur dans les paramètres du profil une fois tous les 30 jours. Allez dans Profil > Paramètres > Compte pour effectuer le changement."
        },
        'report_player': {
            'en': "To report a player: 1) Go to their profile, 2) Click the 'Report' button, 3) Select a reason for reporting, 4) Provide additional details if needed. Our moderation team will review the report within 24 hours.",
            'fr': "Pour signaler un joueur : 1) Allez sur son profil, 2) Cliquez sur le bouton 'Signaler', 3) Sélectionnez une raison, 4) Fournissez des détails supplémentaires si nécessaire. Notre équipe de modération examinera le signalement sous 24 heures."
        }
    }
}

# Challenges
challenges = {
    'general': [
        {
            'name': {'en': 'Easter Egg Hunter', 'fr': "Chasseur d'Oeufs de Pâques"},
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